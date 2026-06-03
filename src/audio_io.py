"""Audio I/O utilities: loading, saving, format conversion, and normalization."""

import os
import subprocess
import numpy as np
import soundfile as sf
import librosa


def load_audio(path: str, sr: int = 44100, mono: bool = False) -> tuple[np.ndarray, int]:
    """Load audio file, resample to target sr, return (samples, sr).

    Always returns float32 in [-1, 1] range, shape (channels, samples) for stereo
    or (samples,) for mono.
    """
    y, orig_sr = sf.read(path, dtype="float32")
    # Ensure 2D for stereo
    if y.ndim == 1:
        y = y[np.newaxis, :]  # (1, samples)
    else:
        y = y.T  # (channels, samples)

    # Resample if needed
    if orig_sr != sr:
        y_resampled = []
        for ch in range(y.shape[0]):
            y_resampled.append(librosa.resample(y[ch], orig_sr=orig_sr, target_sr=sr))
        y = np.stack(y_resampled, axis=0)

    if mono:
        y = np.mean(y, axis=0)

    return y, sr


def save_audio(path: str, y: np.ndarray, sr: int = 44100, subtype: str = "PCM_16"):
    """Save audio to file.

    Args:
        y: Audio array, shape (samples,) for mono or (channels, samples) for stereo.
        sr: Sample rate.
        subtype: Sample subtype (PCM_16, PCM_24, FLOAT).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if y.ndim == 2:
        y = y.T  # soundfile expects (samples, channels)
    sf.write(path, y, sr, subtype=subtype)


def normalize_audio(y: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """Peak-normalize audio to specified dB level."""
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y
    target_peak = 10 ** (peak_db / 20.0)
    return y * (target_peak / peak)


def convert_to_standard(input_path: str, output_path: str, sr: int = 44100, channels: int = 2):
    """Convert any audio file to standard WAV format using ffmpeg."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(sr),
        "-ac", str(channels),
        "-sample_fmt", "s16",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def get_audio_duration(path: str) -> float:
    """Get duration of audio file in seconds."""
    info = sf.info(path)
    return info.duration


def check_clipping(y: np.ndarray, threshold: float = 0.99) -> float:
    """Return ratio of samples above clipping threshold."""
    return float(np.mean(np.abs(y) >= threshold))


def check_silence(y: np.ndarray, threshold: float = 0.001) -> float:
    """Return ratio of silent frames (RMS below threshold)."""
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y if y.ndim == 1 else np.mean(y, axis=0),
                              frame_length=frame_length, hop_length=hop_length)[0]
    return float(np.mean(rms < threshold))


# ============================================================
# Post-processing for EDM
# ============================================================

def soft_clip(y: np.ndarray, threshold: float = 0.9, ratio: float = 4.0) -> np.ndarray:
    """Soft-clip audio above threshold using tanh saturation.

    Acts as a limiter: signals above threshold are gently compressed
    instead of hard-clipped, preserving dynamics while controlling peaks.
    """
    sign = np.sign(y)
    abs_y = np.abs(y)

    # Below threshold: linear. Above: tanh curve scaled to threshold
    below = abs_y <= threshold
    result = np.where(
        below,
        abs_y,
        threshold + (1.0 - threshold) * np.tanh((abs_y - threshold) / ((1.0 - threshold) * ratio))
    )
    return result * sign


def apply_eq(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply EDM-friendly EQ: high-pass rumble removal, low-shelf body, presence boost.

    Uses simple IIR-style biquad filters via scipy.
    """
    from scipy.signal import butter, sosfilt

    mono = y.ndim == 1
    if mono:
        y = y[np.newaxis, :]

    result = np.empty_like(y)

    for ch in range(y.shape[0]):
        sig = y[ch]

        # 1. High-pass at 30 Hz — remove sub-rumble
        sos_hp = butter(2, 30.0, btype="high", fs=sr, output="sos")
        sig = sosfilt(sos_hp, sig)

        # 2. Low-shelf boost at 80 Hz (+3 dB) — add body/punch
        # Approximate with a peaking EQ
        sos_low = butter(2, [60.0, 120.0], btype="band", fs=sr, output="sos")
        low_band = sosfilt(sos_low, sig)
        sig = sig + 0.25 * low_band  # gentle boost

        # 3. Presence boost at 3-5 kHz (+2 dB) — add clarity
        sos_pres = butter(2, [2500.0, 5000.0], btype="band", fs=sr, output="sos")
        pres_band = sosfilt(sos_pres, sig)
        sig = sig + 0.15 * pres_band  # gentle boost

        # 4. High-pass at 25 Hz again to clean up any DC offset from boosts
        sig = sosfilt(sos_hp, sig)

        result[ch] = sig

    return result[0] if mono else result


def stereo_widen(y: np.ndarray, width: float = 1.3) -> np.ndarray:
    """Widen stereo image using mid/side processing.

    width > 1.0 = wider, 1.0 = unchanged, < 1.0 = narrower (mono-ish).
    Only works on stereo audio; mono is returned unchanged.
    """
    if y.ndim == 1 or y.shape[0] < 2:
        return y

    left, right = y[0], y[1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    # Boost side signal for wider image
    side *= width

    new_left = mid + side
    new_right = mid - side

    return np.stack([new_left, new_right], axis=0)


def mono_to_stereo(y: np.ndarray) -> np.ndarray:
    """Convert mono audio to stereo by duplicating the channel."""
    if y.ndim == 2 and y.shape[0] >= 2:
        return y
    if y.ndim == 1:
        return np.stack([y, y], axis=0)
    # y is (1, samples)
    return np.concatenate([y, y], axis=0)


def edm_post_process(y: np.ndarray, sr: int) -> np.ndarray:
    """Full EDM post-processing chain: EQ -> stereo widen -> soft clip -> normalize.

    This makes MusicGen output sound significantly more polished.
    """
    # Ensure stereo
    y = mono_to_stereo(y)

    # EQ shaping
    y = apply_eq(y, sr)

    # Stereo widening
    y = stereo_widen(y, width=1.3)

    # Soft-clip limiter (prevents harsh clipping while allowing loudness)
    y = soft_clip(y, threshold=0.85, ratio=3.0)

    # Final peak normalization
    y = normalize_audio(y, peak_db=-0.5)

    return y
