"""Mel-spectrogram provenance images with a spectral AI-generation watermark."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _font_path() -> str | None:
    candidates = [
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simsun.ttf",
        r"C:\Windows\Fonts\simsunb.ttf",
        r"C:\Windows\Fonts\STSONG.TTF",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\Deng.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_audio_mono(audio_path: str | Path, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1).astype(np.float32)
    if sr != target_sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    if mono.size == 0:
        raise ValueError(f"empty audio: {audio_path}")
    return mono, int(sr)


def _render_text_mask(text: str, width: int, height: int) -> np.ndarray:
    """Rasterize watermark text as a frequency-time mask.

    The returned mask is indexed like a mel matrix: row 0 is the lowest
    frequency bin. No text is drawn on the final plot; the mask is converted
    into synthetic mel energy in `_spectral_watermark_panel`.
    """

    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    width = max(int(width), 16)
    height = max(int(height), 16)
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    font_path = _font_path()
    max_w = int(width * 0.90)
    max_h = int(height * 0.70)
    font = ImageFont.load_default()

    for size in range(max_h, 9, -2):
        try:
            candidate = ImageFont.truetype(font_path, size=size) if font_path else ImageFont.load_default()
        except Exception:
            candidate = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=candidate)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= max_w and text_h <= max_h:
            font = candidate
            break

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2 - bbox[0]
    y = (height - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)

    # Slightly thicken strokes so the characters remain legible after imshow
    # resampling and color mapping.
    canvas = canvas.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius=0.45))
    mask = np.asarray(canvas, dtype=np.float32) / 255.0
    return mask[::-1, :]


def _spectral_watermark_panel(text: str, n_mels: int, frames: int) -> np.ndarray:
    """Create a spectrogram-art panel whose negative-space pattern spells text."""

    rng = np.random.default_rng(20260602)
    frames = max(int(frames), 260)
    mask = _render_text_mask(text, frames, n_mels)
    freq = np.linspace(0.0, 1.0, n_mels, dtype=np.float32)[:, None]
    time = np.linspace(0.0, 1.0, frames, dtype=np.float32)[None, :]

    # The reference style is closer to "spectrogram art": a hot noisy field
    # with dark negative-space shapes. Keep this matrix fully mel-like instead
    # of drawing text on top of the plot.
    background = -31.0 + rng.normal(0.0, 6.0, size=(n_mels, frames)).astype(np.float32)
    background += 10.0 * (1.0 - freq)  # lower bands are usually brighter
    background += 2.8 * np.sin(2 * np.pi * (freq * 10.0 + time * 0.8)).astype(np.float32)
    background += 2.0 * np.sin(2 * np.pi * (freq * 31.0 - time * 1.3)).astype(np.float32)

    # Dense horizontal and vertical texture so the area reads like a spectrum,
    # not a flat rectangular watermark.
    for bin_idx in range(5, n_mels, 7):
        background[max(0, bin_idx - 1):min(n_mels, bin_idx + 1), :] += rng.uniform(4.0, 8.5)
    for frame_idx in range(0, frames, 17):
        width = int(rng.integers(1, 4))
        background[:, frame_idx:frame_idx + width] += rng.uniform(3.0, 7.0)

    from scipy.ndimage import binary_dilation, gaussian_filter

    hard_mask = mask > 0.18
    outline = binary_dilation(hard_mask, iterations=2).astype(np.float32) - hard_mask.astype(np.float32)
    outline = gaussian_filter(outline, sigma=0.8)
    soft_mask = np.clip(mask * 1.25, 0.0, 1.0)

    cutout = -73.0 + rng.normal(0.0, 2.6, size=(n_mels, frames)).astype(np.float32)
    cutout += 2.0 * np.sin(2 * np.pi * (freq * 22.0)).astype(np.float32)
    edge_energy = -6.0 + rng.normal(0.0, 1.5, size=(n_mels, frames)).astype(np.float32)

    panel = background * (1.0 - soft_mask) + cutout * soft_mask
    panel = panel * (1.0 - outline) + edge_energy * outline
    return np.clip(panel, -80.0, 0.0).astype(np.float32)


def save_ai_watermarked_mel(
    audio_path: str | Path,
    output_path: str | Path | None = None,
    *,
    watermark_text: str = "AI生成",
    title: str = "完整 Mel 图谱（末尾频谱负形 AI 水印）",
    n_mels: int = 96,
    hop_length: int = 512,
    target_sr: int = 44100,
) -> str:
    """Save a full-song mel spectrogram with an end spectral watermark.

    The watermark is encoded as mel-like energy in appended time frames. It is
    visual provenance for the diagnostic image and does not modify the audio
    waveform.
    """

    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    audio_path = Path(audio_path)
    if output_path is None:
        output_path = audio_path.with_name(f"{audio_path.stem}_mel_ai_watermark.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y, sr = _load_audio_mono(audio_path, target_sr=target_sr)
    duration = max(float(len(y) / sr), 0.001)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=min(sr // 2, 16000),
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    watermark_duration = max(5.0, min(18.0, duration * 0.22))
    watermark_frames = max(96, int(round(watermark_duration * sr / hop_length)))
    watermark_db = _spectral_watermark_panel(watermark_text, n_mels, watermark_frames)
    mel_db_with_watermark = np.concatenate([mel_db, watermark_db], axis=1)
    actual_watermark_duration = watermark_frames * hop_length / sr
    total_duration = duration + actual_watermark_duration
    fig_width = min(18.0, max(10.0, total_duration / 14.0))
    cn_font_path = _font_path()
    cn_font = fm.FontProperties(fname=cn_font_path) if cn_font_path else None

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "axes.linewidth": 0.8,
        "axes.grid": False,
    })
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    extent = [0.0, total_duration, 0, n_mels]
    image = ax.imshow(mel_db_with_watermark, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.set_xlim(0.0, total_duration)
    ax.set_ylim(0, n_mels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bins")
    ax.set_title(title, fontproperties=cn_font, fontsize=13)
    ax.axvline(duration, color="#f2f2f2", linewidth=0.9, alpha=0.9)
    cbar = fig.colorbar(image, ax=ax, pad=0.012, fraction=0.035)
    cbar.set_label("dB")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    sidecar = {
        "audio_path": str(audio_path),
        "mel_path": str(output_path),
        "watermark_text": watermark_text,
        "audio_duration_seconds": round(duration, 4),
        "watermark_duration_seconds": round(actual_watermark_duration, 4),
        "watermark_mode": "spectral_energy_text",
        "watermark_frames": int(watermark_frames),
        "sample_rate": sr,
        "n_mels": int(n_mels),
        "hop_length": int(hop_length),
        "note": "The AI watermark is encoded as appended mel-spectrogram energy; the audio waveform is unchanged.",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(output_path)
