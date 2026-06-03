"""Audio generation: ACE-Step and Stable Audio Open inference."""

from __future__ import annotations

import os
import sys
import threading

import numpy as np
import torch


# ============================================================
# ACE-Step (primary model)
# ============================================================

_acestep_pipeline = None
_acestep_pipeline_cache: dict[str, object] = {}
_acestep_pipeline_cache_lock = threading.RLock()


# 默认模型存放路径（项目目录下）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ACE_STEP_SOURCE_DIR = os.path.join(_PROJECT_ROOT, "ACE-Step")
if os.path.isdir(_ACE_STEP_SOURCE_DIR) and _ACE_STEP_SOURCE_DIR not in sys.path:
    sys.path.insert(0, _ACE_STEP_SOURCE_DIR)

_DEFAULT_CHECKPOINT_DIR = os.path.join(
    _PROJECT_ROOT,
    "models", "ace-step", "ACE-Step-v1-3.5B"
)

REFERENCE_MELODY_PROMPT = (
    "instrumental music that preserves the reference audio's melodic contour, hook motif, "
    "chord progression energy, rhythmic groove, lead phrasing, arrangement density, and mix balance; "
    "clear memorable lead melody, coherent song structure, high quality production"
)

REFERENCE_RECONSTRUCTION_PROMPT = (
    "high fidelity instrumental variation of the reference audio; keep the same main melody contour, "
    "chord movement, hook rhythm, bass groove, lead phrasing, section energy, and mix balance; "
    "clear tonal harmony, stable chords, audible lead melody, no random noise"
)

REFERENCE_STYLE_PROMPT = (
    "instrumental original track inspired by the reference audio's sound palette, synth timbre, drum design, "
    "bass tone, groove density, arrangement energy, stereo width, and mix balance; compose a new melody, "
    "new chord progression, and new hook rhythm; do not copy or interpolate the reference melody; "
    "clear tonal center, stable chord progression, audible lead motif, clean transients, high fidelity mix; "
    "no vocals, no singing, no spoken words"
)


def reset_acestep_model_cache():
    """Clear the cached ACE-Step pipeline so the next load starts from base weights."""
    global _acestep_pipeline, _acestep_pipeline_cache
    with _acestep_pipeline_cache_lock:
        _acestep_pipeline = None
        _acestep_pipeline_cache = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_acestep_model(
    checkpoint_dir: str = "",
    device: str = "auto",
    cpu_offload: bool = True,
    dtype: str = "float32",
    cache_key: str | None = None,
):
    """Load ACE-Step pipeline. Downloads checkpoint on first use.

    Returns the ACEStepPipeline instance.
    """
    global _acestep_pipeline
    with _acestep_pipeline_cache_lock:
        if cache_key is None and _acestep_pipeline is not None:
            return _acestep_pipeline
        if cache_key is not None and cache_key in _acestep_pipeline_cache:
            return _acestep_pipeline_cache[cache_key]

        from acestep.pipeline_ace_step import ACEStepPipeline

        if not checkpoint_dir:
            checkpoint_dir = _DEFAULT_CHECKPOINT_DIR

        # ACE-Step uses torchaudio.load internally when encoding reference audio.
        # The local patch routes torchaudio I/O through soundfile on Windows.
        try:
            scripts_dir = os.path.join(_PROJECT_ROOT, "scripts")
            if os.path.isdir(scripts_dir) and scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            import torchaudio_patch  # noqa: F401
        except Exception:
            pass

        if device == "auto":
            if torch.cuda.is_available():
                device_id = 0
            else:
                device_id = -1  # CPU
        elif device == "cpu":
            device_id = -1
        else:
            device_id = int(device)

        # On CPU, force float32 and enable cpu_offload
        if not torch.cuda.is_available():
            dtype = "float32"
            cpu_offload = True

        print(f"Loading ACE-Step from: {checkpoint_dir}")
        print("(first run will download ~4GB checkpoint)...")
        pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_dir,
            device_id=device_id if device_id >= 0 else 0,
            dtype=dtype,
            torch_compile=False,
            cpu_offload=cpu_offload,
            overlapped_decode=False,
        )
        # Force load checkpoint eagerly
        pipeline.load_checkpoint(pipeline.checkpoint_dir)

        # Monkey-patch: 用 soundfile 代替 torchaudio.save（避免 torchcodec 依赖问题）
        _patch_pipeline_save(pipeline)
        _patch_pipeline_lora_state(pipeline)

        if cache_key is None:
            _acestep_pipeline = pipeline
        else:
            _acestep_pipeline_cache[cache_key] = pipeline
        print("ACE-Step model loaded.")
        return pipeline


def apply_finetuned_weights(pipeline, model_path_or_dir: str) -> dict[str, int | str]:
    """Load full fine-tuned transformer weights from a model.pt file or containing directory."""
    model_path = model_path_or_dir
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到微调权重文件: {model_path}")

    print(f"Loading full fine-tuned weights from: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    prefix = "ace_step_transformer."
    if isinstance(state_dict, dict) and state_dict:
        keys = list(state_dict.keys())
        if all(key.startswith(prefix) for key in keys):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}

    result = pipeline.ace_step_transformer.load_state_dict(state_dict, strict=False)
    return {
        "path": model_path,
        "missing_keys": len(result.missing_keys),
        "unexpected_keys": len(result.unexpected_keys),
    }


def _patch_pipeline_save(pipeline):
    """Patch pipeline to use soundfile for saving (avoids torchcodec/FFmpeg dependency)."""
    import soundfile as sf

    def _save_wav_patched(target_wav, idx, save_path=None, sample_rate=48000, format="wav"):
        import time as _time
        if save_path is None:
            base_path = "./outputs"
            os.makedirs(base_path, exist_ok=True)
            output_path_wav = f"{base_path}/output_{_time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
        else:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            if os.path.isdir(save_path):
                output_path_wav = os.path.join(save_path, f"output_{_time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}")
            else:
                output_path_wav = save_path

        wav_np = target_wav.float().cpu().numpy()
        if wav_np.ndim == 1:
            wav_np = wav_np[np.newaxis, :]
        sf.write(output_path_wav, wav_np.T, sample_rate)
        return output_path_wav

    pipeline.save_wav_file = _save_wav_patched


def _patch_pipeline_lora_state(pipeline):
    """Keep LoRA state accurate when ACE-Step unloads adapters."""
    if getattr(pipeline, "_edm_lora_state_patched", False):
        return

    original_load_lora = pipeline.load_lora

    def _load_lora_with_state(lora_name_or_path, lora_weight):
        result = original_load_lora(lora_name_or_path, lora_weight)
        if lora_name_or_path == "none":
            pipeline.lora_path = "none"
            pipeline.lora_weight = 1.0
        return result

    pipeline.load_lora = _load_lora_with_state
    pipeline._edm_lora_state_patched = True


def _find_generated_audio_path(output_paths, save_dir: str, ignored_paths: set[str] | None = None) -> str | None:
    ignored = {os.path.abspath(path) for path in (ignored_paths or set()) if path}

    if output_paths:
        for path in output_paths:
            if not path or not isinstance(path, str) or not path.lower().endswith(".wav"):
                continue
            abs_path = os.path.abspath(path)
            if abs_path not in ignored and os.path.exists(abs_path):
                return abs_path

    if not save_dir or not os.path.isdir(save_dir):
        return None

    candidates = []
    for name in os.listdir(save_dir):
        if not name.lower().endswith(".wav"):
            continue
        path = os.path.abspath(os.path.join(save_dir, name))
        if path in ignored:
            continue
        candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=os.path.getmtime, reverse=True)[0]


def prepare_reference_audio_for_acestep(
    reference_audio_path: str,
    duration: float,
    output_dir: str,
    reference_start: float = 0.0,
    sample_rate: int = 48000,
    auto_reference_start: bool = True,
) -> tuple[str, dict]:
    """Standardize a reference clip for ACE-Step audio-to-audio latent conditioning."""
    from src.audio_io import load_audio, normalize_audio, save_audio

    if not reference_audio_path or not os.path.exists(reference_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

    duration = max(1.0, float(duration))
    reference_start = max(0.0, float(reference_start or 0.0))
    os.makedirs(output_dir, exist_ok=True)

    audio, sr = load_audio(reference_audio_path, sr=sample_rate, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    original_duration = audio.shape[-1] / float(sr)
    auto_selected_start = None
    if auto_reference_start and reference_start <= 0 and original_duration > duration * 1.5:
        auto_selected_start = _select_reference_start_by_energy(audio, sr, duration)
        reference_start = auto_selected_start

    start_sample = int(reference_start * sr)
    if start_sample >= audio.shape[-1]:
        start_sample = 0
        reference_start = 0.0

    clip = audio[:, start_sample:]
    if clip.shape[-1] == 0:
        raise ValueError("Reference audio is empty after applying the start offset.")

    target_samples = int(round(duration * sr))
    if clip.shape[-1] < target_samples:
        repeats = int(np.ceil(target_samples / max(clip.shape[-1], 1)))
        clip = np.tile(clip, (1, repeats))
    clip = clip[:, :target_samples]
    clip = normalize_audio(clip.astype(np.float32), peak_db=-3.0)

    prepared_path = os.path.join(output_dir, "reference_style_input.wav")
    save_audio(prepared_path, clip, sr=sr, subtype="PCM_16")

    metadata = {
        "original_path": os.path.abspath(reference_audio_path),
        "prepared_path": os.path.abspath(prepared_path),
        "sample_rate": sr,
        "channels": int(clip.shape[0]),
        "original_duration": float(original_duration),
        "reference_start": float(reference_start),
        "auto_reference_start": bool(auto_reference_start),
        "auto_selected_start": None if auto_selected_start is None else float(auto_selected_start),
        "prepared_duration": float(clip.shape[-1] / sr),
        "target_duration": float(duration),
        "looped_or_trimmed": bool(
            original_duration - reference_start < duration
            or original_duration - reference_start > duration
        ),
    }
    return prepared_path, metadata


def _select_reference_start_by_energy(audio: np.ndarray, sr: int, duration: float) -> float:
    """Pick a non-silent/high-energy segment when the user leaves the start at 0."""
    if audio.ndim == 2:
        mono = np.mean(audio, axis=0)
    else:
        mono = audio
    total_samples = mono.shape[-1]
    target_samples = int(max(1.0, duration) * sr)
    if total_samples <= target_samples:
        return 0.0

    frame_seconds = 0.5
    frame = max(1, int(frame_seconds * sr))
    frame_count = max(1, total_samples // frame)
    trimmed = mono[: frame_count * frame]
    framed = trimmed.reshape(frame_count, frame)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-10)
    if np.max(rms) <= 1e-8:
        return 0.0

    rms = rms / (np.max(rms) + 1e-8)
    window_frames = max(1, int(duration / frame_seconds))
    if frame_count <= window_frames:
        return 0.0

    kernel = np.ones(window_frames, dtype=np.float32) / window_frames
    scores = np.convolve(rms.astype(np.float32), kernel, mode="valid")

    # Avoid choosing a sparse intro or fade-out when possible.
    min_start_frame = min(int(8.0 / frame_seconds), max(0, len(scores) - 1))
    last_allowed_seconds = min(total_samples / sr - duration - 3.0, total_samples / sr * 0.85)
    max_start_frame = max(min_start_frame, int(last_allowed_seconds / frame_seconds))
    usable = scores[min_start_frame : max_start_frame + 1]
    if usable.size == 0:
        best_frame = int(np.argmax(scores))
    else:
        best_frame = min_start_frame + int(np.argmax(usable))
    return float(best_frame * frame_seconds)


def build_reference_style_proxy_audio(
    reference_audio_path: str,
    duration: float,
    output_dir: str,
    reference_start: float = 0.0,
    sample_rate: int = 48000,
    auto_reference_start: bool = True,
    use_demucs: bool = False,
) -> tuple[str, dict]:
    """Build a safer reference clip for style/timbre transfer.

    A full mastered song latent entangles melody, harmony, vocals, bass, drums, and
    limiting. For "style timbre, new melody" generation, this proxy keeps mostly
    non-melodic evidence: percussive transients, bass envelope, and a quiet
    residual texture bed. It deliberately suppresses harmonic/melodic content so
    audio2audio conditioning is less likely to copy or smear the source song.
    """
    from scipy.signal import butter, sosfilt
    import librosa

    from src.audio_io import load_audio, normalize_audio, save_audio

    prepared_ref, metadata = prepare_reference_audio_for_acestep(
        reference_audio_path=reference_audio_path,
        duration=float(duration),
        output_dir=output_dir,
        reference_start=float(reference_start or 0.0),
        sample_rate=sample_rate,
        auto_reference_start=auto_reference_start,
    )
    metadata["style_proxy_enabled"] = True
    metadata["style_proxy_method"] = "hpss_percussive_bass_texture"
    metadata["style_proxy_demucs_requested"] = bool(use_demucs)
    metadata["style_proxy_demucs_used"] = False

    source_for_proxy = prepared_ref
    demucs_stems = {}
    if use_demucs:
        try:
            from src.demucs_wrapper import separate_stems

            demucs_out = os.path.join(output_dir, "reference_proxy_demucs")
            demucs_stems = separate_stems(prepared_ref, demucs_out) or {}
            # Use drums/bass/other as a style bed, but keep vocals out. Other may
            # still contain leads, so it is kept quiet below.
            parts = []
            weights = {"drums": 0.95, "bass": 0.70, "other": 0.20}
            for stem, weight in weights.items():
                stem_path = demucs_stems.get(stem)
                if not stem_path or not os.path.exists(stem_path):
                    continue
                stem_audio, _ = load_audio(stem_path, sr=sample_rate, mono=False)
                if stem_audio.ndim == 1:
                    stem_audio = stem_audio[np.newaxis, :]
                if stem_audio.shape[0] == 1:
                    stem_audio = np.repeat(stem_audio, 2, axis=0)
                parts.append(weight * stem_audio[:, : int(round(duration * sample_rate))])
            if parts:
                min_len = min(part.shape[-1] for part in parts)
                proxy_mix = np.sum([part[:, :min_len] for part in parts], axis=0)
                source_for_proxy = os.path.join(output_dir, "reference_proxy_demucs_mix.wav")
                save_audio(source_for_proxy, normalize_audio(proxy_mix, peak_db=-6.0), sr=sample_rate, subtype="PCM_16")
                metadata["style_proxy_method"] = "demucs_drums_bass_other_hpss"
                metadata["style_proxy_demucs_used"] = True
                metadata["style_proxy_demucs_stems"] = demucs_stems
        except Exception as exc:
            metadata["style_proxy_demucs_error"] = str(exc)

    audio, sr = load_audio(source_for_proxy, sr=sample_rate, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    target_samples = int(round(float(duration) * sr))
    if audio.shape[-1] < target_samples:
        repeats = int(np.ceil(target_samples / max(audio.shape[-1], 1)))
        audio = np.tile(audio, (1, repeats))
    audio = audio[:, :target_samples]

    proxy_channels = []
    for ch in range(audio.shape[0]):
        sig = audio[ch].astype(np.float32)
        harmonic, percussive = librosa.effects.hpss(sig, margin=(1.0, 5.0))

        low_sos = butter(4, [45.0, 230.0], btype="bandpass", fs=sr, output="sos")
        low = sosfilt(low_sos, sig)

        high_sos = butter(3, 900.0, btype="highpass", fs=sr, output="sos")
        texture = sosfilt(high_sos, sig)

        rumble_sos = butter(2, 35.0, btype="highpass", fs=sr, output="sos")
        proxy = 0.72 * percussive + 0.28 * low + 0.20 * texture + 0.05 * harmonic
        proxy = sosfilt(rumble_sos, proxy)
        proxy_channels.append(proxy.astype(np.float32))

    proxy_audio = np.stack(proxy_channels, axis=0)
    mono_proxy = np.mean(proxy_audio, axis=0)
    spec = np.abs(np.fft.rfft(mono_proxy))
    freqs = np.fft.rfftfreq(len(mono_proxy), d=1.0 / sr)
    total_energy = float(np.sum(spec ** 2) + 1e-10)
    low_ratio = float(np.sum(spec[(freqs >= 20) & (freqs <= 250)] ** 2) / total_energy)
    metadata["style_proxy_low_freq_ratio_before_tame"] = low_ratio
    if low_ratio > 0.48:
        tame = float(np.clip((low_ratio - 0.48) * 1.2, 0.18, 0.55))
        low_sos = butter(4, [35.0, 260.0], btype="bandpass", fs=sr, output="sos")
        for ch in range(proxy_audio.shape[0]):
            proxy_audio[ch] = proxy_audio[ch] - tame * sosfilt(low_sos, proxy_audio[ch])
        metadata["style_proxy_low_tame_amount"] = tame
    else:
        metadata["style_proxy_low_tame_amount"] = 0.0
    proxy_audio = normalize_audio(proxy_audio, peak_db=-6.0)
    proxy_path = os.path.join(output_dir, "reference_style_proxy_nonmelodic.wav")
    save_audio(proxy_path, proxy_audio, sr=sr, subtype="PCM_16")

    metadata["style_proxy_path"] = os.path.abspath(proxy_path)
    metadata["style_proxy_source"] = os.path.abspath(source_for_proxy)
    metadata["style_proxy_description"] = (
        "HPSS suppresses harmonic/melodic content; proxy keeps percussive transients, "
        "bass envelope, and quiet high-frequency texture for timbre-style conditioning."
    )
    return proxy_path, metadata


def score_reference_candidate(audio: np.ndarray, sr: int, reference_features: dict | None = None) -> dict:
    """Score a generated candidate for clarity and non-muddy EDM usability."""
    import librosa
    from src.audio_features import extract_all_features
    from src.audio_io import check_clipping, check_silence

    y = audio[0] if audio.ndim == 2 else audio
    y = np.asarray(y, dtype=np.float32)
    features = extract_all_features(y, sr)
    centroid = float(features.get("spectral_centroid_mean") or 0.0)
    low_ratio = float(features.get("low_freq_ratio") or 0.0)
    onset = float(features.get("onset_density") or 0.0)
    rms = float(features.get("rms_mean") or 0.0)
    clipping = check_clipping(y)
    silence = check_silence(y)

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) + 1e-8
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    rolloff = float(features.get("spectral_rolloff_mean") or 0.0)

    centroid_score = float(np.clip((centroid - 900.0) / 2600.0, 0.0, 1.0))
    rolloff_score = float(np.clip((rolloff - 3500.0) / 5500.0, 0.0, 1.0))
    onset_score = float(np.exp(-abs(onset - 4.0) / 3.2))
    rms_score = float(np.clip(rms / 0.09, 0.0, 1.0))
    mud_penalty = float(np.clip((low_ratio - 0.48) / 0.22, 0.0, 1.0))
    noise_penalty = float(np.clip((flatness - 0.28) / 0.35, 0.0, 1.0))
    clip_penalty = float(np.clip(clipping / 0.015, 0.0, 1.0))
    silence_penalty = float(np.clip(silence / 0.25, 0.0, 1.0))

    ref_low = float((reference_features or {}).get("low_freq_ratio") or low_ratio)
    target_low = float(np.clip(ref_low, 0.16, 0.42))
    low_match_score = float(np.exp(-abs(low_ratio - target_low) / 0.18))

    score = (
        0.24 * centroid_score
        + 0.13 * rolloff_score
        + 0.20 * onset_score
        + 0.14 * rms_score
        + 0.13 * low_match_score
        - 0.24 * mud_penalty
        - 0.12 * noise_penalty
        - 0.10 * clip_penalty
        - 0.10 * silence_penalty
    )
    return {
        "score": float(score),
        "bpm": float(features.get("bpm") or 0.0),
        "rms": rms,
        "low_freq_ratio": low_ratio,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "onset_density": onset,
        "spectral_flatness": flatness,
        "clipping_ratio": float(clipping),
        "silence_ratio": float(silence),
        "mud_penalty": mud_penalty,
        "noise_penalty": noise_penalty,
    }


def generate_acestep(
    pipeline,
    prompt: str,
    lyrics: str = "",
    duration: float = 60.0,
    seed: int = 42,
    infer_step: int = 60,
    guidance_scale: float = 15.0,
    scheduler_type: str = "euler",
    cfg_type: str = "apg",
    omega_scale: float = 10.0,
    guidance_interval: float = 0.5,
    min_guidance_scale: float = 3.0,
    use_erg: bool = True,
    save_dir: str = None,
    lora_name_or_path: str | None = None,
    lora_weight: float | None = None,
    progress_callback=None,
) -> tuple[np.ndarray, int]:
    """Generate audio using ACE-Step.

    Returns (audio_numpy, sample_rate) where audio is shape (channels, samples).
    """
    if save_dir is None:
        import tempfile
        save_dir = tempfile.mkdtemp(prefix="acestep_")

    os.makedirs(save_dir, exist_ok=True)

    # ACE-Step generates vocals by default; for instrumental-only, use [instrumental] in lyrics
    if not lyrics or lyrics.strip() == "":
        lyrics = "[instrumental]"

    try:
        output_paths = pipeline(
            format="wav",
            audio_duration=duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=[seed],
            guidance_interval=guidance_interval,
            guidance_interval_decay=0.0,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=bool(use_erg),
            use_erg_lyric=bool(use_erg),
            use_erg_diffusion=bool(use_erg),
            lora_name_or_path=lora_name_or_path if lora_name_or_path is not None else getattr(pipeline, "lora_path", "none"),
            lora_weight=float(lora_weight) if lora_weight is not None else getattr(pipeline, "lora_weight", 1.0),
            save_path=save_dir,
            progress_callback=progress_callback,
        )
    except Exception as e:
        raise RuntimeError(f"ACE-Step pipeline 执行失败: {e}") from e

    # 找到生成的音频文件（pipeline 返回 [audio_path, json_path, ...]）
    audio_path = None
    audio_path = _find_generated_audio_path(output_paths, save_dir)

    # 如果 pipeline 返回的路径没找到，搜索 save_dir
    if not audio_path:
        for f in os.listdir(save_dir):
            if f.endswith(".wav"):
                audio_path = os.path.join(save_dir, f)
                break

    if not audio_path:
        raise RuntimeError(
            f"ACE-Step 生成完成但未找到输出文件。"
            f"返回路径: {output_paths}，目录内容: {os.listdir(save_dir) if os.path.exists(save_dir) else '目录不存在'}"
        )

    # Load the generated audio (use soundfile to avoid torchcodec/FFmpeg dependency)
    import soundfile as sf
    wav, sr = sf.read(audio_path, dtype='float32')
    audio = wav.T  # soundfile returns (samples, channels), we need (channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    return audio, sr


def generate_acestep_reference_style(
    pipeline,
    prompt: str,
    reference_audio_path: str,
    lyrics: str = "",
    duration: float = 30.0,
    seed: int = 42,
    infer_step: int = 120,
    guidance_scale: float = 4.5,
    ref_audio_strength: float = 0.90,
    reference_start: float = 0.0,
    auto_reference_start: bool = True,
    use_style_proxy: bool = True,
    use_demucs_proxy: bool = False,
    save_dir: str = None,
    progress_callback=None,
) -> tuple[np.ndarray, int, dict]:
    """Generate with training-free reference-audio latent conditioning."""
    if save_dir is None:
        import tempfile
        save_dir = tempfile.mkdtemp(prefix="acestep_reference_")
    os.makedirs(save_dir, exist_ok=True)

    if not prompt or not prompt.strip():
        prompt = REFERENCE_RECONSTRUCTION_PROMPT
    prompt = prompt.strip()
    lyrics = (lyrics or "").strip() or "[instrumental]"
    strength = float(np.clip(float(ref_audio_strength), 0.05, 0.97))
    sigma_max = max(0.03, 1.0 - strength)
    effective_infer_step = int(np.ceil(int(infer_step) / sigma_max))
    effective_infer_step = int(np.clip(effective_infer_step, int(infer_step), int(infer_step) * 12))

    if use_style_proxy:
        prepared_ref, ref_metadata = build_reference_style_proxy_audio(
            reference_audio_path=reference_audio_path,
            duration=float(duration),
            output_dir=save_dir,
            reference_start=float(reference_start or 0.0),
            sample_rate=48000,
            auto_reference_start=auto_reference_start,
            use_demucs=bool(use_demucs_proxy),
        )
    else:
        prepared_ref, ref_metadata = prepare_reference_audio_for_acestep(
            reference_audio_path=reference_audio_path,
            duration=float(duration),
            output_dir=save_dir,
            reference_start=float(reference_start or 0.0),
            sample_rate=48000,
            auto_reference_start=auto_reference_start,
        )
        ref_metadata["style_proxy_enabled"] = False

    try:
        output_paths = pipeline(
            format="wav",
            audio_duration=float(duration),
            prompt=prompt,
            lyrics=lyrics,
            infer_step=effective_infer_step,
            guidance_scale=float(guidance_scale),
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10.0,
            manual_seeds=[int(seed)],
            guidance_interval=0.5,
            guidance_interval_decay=0.0,
            min_guidance_scale=3.0,
            use_erg_tag=False,
            use_erg_lyric=False,
            use_erg_diffusion=False,
            audio2audio_enable=True,
            ref_audio_input=prepared_ref,
            ref_audio_strength=strength,
            lora_name_or_path="none",
            lora_weight=1.0,
            save_path=save_dir,
            progress_callback=progress_callback,
        )
    except Exception as e:
        raise RuntimeError(f"ACE-Step reference-audio pipeline failed: {e}") from e

    audio_path = _find_generated_audio_path(
        output_paths,
        save_dir,
        ignored_paths={prepared_ref},
    )
    if not audio_path:
        raise RuntimeError(
            f"ACE-Step reference generation finished but no generated WAV was found. "
            f"Returned paths: {output_paths}; directory: {os.listdir(save_dir) if os.path.exists(save_dir) else 'missing'}"
        )

    import soundfile as sf
    wav, sr = sf.read(audio_path, dtype="float32")
    audio = wav.T
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    metadata = {
        "task": "training_free_reference_audio",
        "prompt": prompt,
        "lyrics": lyrics,
        "seed": int(seed),
        "duration": float(duration),
        "infer_step": int(infer_step),
        "guidance_scale": float(guidance_scale),
        "ref_audio_strength": strength,
        "requested_infer_step": int(infer_step),
        "effective_infer_step": int(effective_infer_step),
        "reference": ref_metadata,
        "generated_path": os.path.abspath(audio_path),
        "output_paths": output_paths,
        "lora_name_or_path": "none",
        "use_style_proxy": bool(use_style_proxy),
        "use_demucs_proxy": bool(use_demucs_proxy),
    }
    return audio, sr, metadata


def generate_acestep_sequence(
    pipeline,
    sections: list[dict],
    crossfade_seconds: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Generate a longer track by stitching ACE-Step sections."""
    audio_segments = []
    sample_rate = None

    for section in sections:
        audio, sample_rate = generate_acestep(
            pipeline=pipeline,
            prompt=section["prompt"],
            lyrics=section.get("lyrics", "[instrumental]"),
            duration=float(section["duration"]),
            seed=int(section["seed"]),
            infer_step=section.get("infer_step", 60),
            guidance_scale=section.get("guidance_scale", 15.0),
        )
        audio_segments.append(audio)

    if not audio_segments:
        raise ValueError("No sections were generated.")

    if len(audio_segments) == 1:
        return audio_segments[0], sample_rate

    fade_samples = int(sample_rate * crossfade_seconds)
    stitched = audio_segments[0]
    for segment in audio_segments[1:]:
        stitched = _crossfade_audio(stitched, segment, fade_samples)

    return stitched, sample_rate


def get_acestep_max_duration() -> float:
    """ACE-Step supports up to ~4 minutes per generation."""
    return 240.0


def _crossfade_audio(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    """Crossfade two audio arrays along the sample axis."""
    if fade_samples <= 0:
        return np.concatenate([a, b], axis=-1)

    if fade_samples > min(a.shape[-1], b.shape[-1]):
        fade_samples = max(1, min(a.shape[-1], b.shape[-1]) // 2)

    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

    if a.ndim == 2:
        fade_out = fade_out[np.newaxis, :]
        fade_in = fade_in[np.newaxis, :]

    mixed_tail = a[..., -fade_samples:] * fade_out + b[..., :fade_samples] * fade_in
    return np.concatenate([a[..., :-fade_samples], mixed_tail, b[..., fade_samples:]], axis=-1)


# ============================================================
# Stable Audio Open (secondary / legacy)
# ============================================================

def load_stable_audio_model(device: str = "cuda"):
    """Load Stable Audio Open 1.0 model.

    Returns (model, config) tuple.
    """
    from src.model_store import configure_hf_environment, resolve_model_source

    configure_hf_environment()
    try:
        from stable_audio_tools import get_pretrained_model
    except ImportError:
        raise ImportError(
            "stable-audio-tools not installed. "
            "Install with: pip install stable-audio-tools"
        )

    model, config = get_pretrained_model(resolve_model_source("stabilityai/stable-audio-open-1.0"))
    model = model.to(device)
    return model, config


def generate_stable_audio(
    model,
    config,
    prompt: str,
    duration: float = 10.0,
    seed: int = 42,
    device: str = "cuda",
    steps: int = 100,
    cfg_scale: int = 7,
) -> np.ndarray:
    """Generate audio using Stable Audio Open model.

    Returns numpy array of shape (channels, samples).
    """
    from stable_audio_tools import generate_diffusion_cond

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    sample_rate = config["sample_rate"]
    sample_size = int(duration * sample_rate)

    conditioning = [{"prompt": prompt, "seconds_total": duration}]

    audio = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-2m-sde",
        device=device,
    )

    audio = audio.cpu().numpy()
    if audio.ndim == 3:
        audio = audio[0]

    return audio


def generate_batch(
    model,
    config_or_processor,
    prompts: list[dict],
    output_dir: str,
    model_type: str = "acestep",
    device: str = "cpu",
    seeds: list[int] | None = None,
) -> list[str]:
    """Generate audio for a batch of prompts.

    Args:
        model: Loaded model (ACEStepPipeline or stable-audio model).
        config_or_processor: Model config (stable-audio) or None (acestep).
        prompts: List of prompt dicts with 'text', 'id', 'duration'.
        output_dir: Directory to save outputs.
        model_type: 'acestep' or 'stable-audio'.
        device: Device string.
        seeds: List of seeds (default [42, 123, 456]).

    Returns:
        List of output file paths.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    for prompt_info in prompts:
        prompt_text = prompt_info["text"]
        prompt_id = prompt_info["id"]
        duration = prompt_info.get("duration", 60.0)

        for seed in seeds:
            fname = f"{prompt_id}_seed_{seed:03d}.wav"
            out_path = os.path.join(output_dir, fname)

            if os.path.exists(out_path):
                print(f"  Skipping (exists): {fname}")
                output_files.append(out_path)
                continue

            print(f"  Generating: {fname}")

            if model_type == "acestep":
                from src.audio_io import save_audio, normalize_audio
                audio, sr = generate_acestep(
                    model,
                    prompt=prompt_text,
                    duration=duration,
                    seed=seed,
                    save_dir=os.path.join(output_dir, f"tmp_{prompt_id}_{seed}"),
                )
                audio = normalize_audio(audio, peak_db=-1.0)
                save_audio(out_path, audio, sr=sr)
            else:
                from src.audio_io import save_audio, normalize_audio
                audio = generate_stable_audio(
                    model, config_or_processor, prompt_text,
                    duration=duration, seed=seed, device=device,
                )
                audio = normalize_audio(audio, peak_db=-1.0)
                save_audio(out_path, audio, sr=config_or_processor["sample_rate"])

            output_files.append(out_path)

    return output_files
