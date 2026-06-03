"""Demucs wrapper with torchaudio.save monkey-patch to avoid torchcodec dependency."""

import os
import sys
import subprocess
import json


def patch_torchaudio():
    """Monkey-patch torchaudio.save and torchaudio.load to use soundfile."""
    try:
        import torchaudio
        import soundfile as sf
        import torch
        import numpy as np

        def _save_with_soundfile(filepath, src, sample_rate, **kwargs):
            """Use soundfile to save audio, avoiding torchcodec dependency."""
            wav_np = src.float().cpu().numpy()
            if wav_np.ndim == 1:
                wav_np = wav_np.reshape(1, -1)
            sf.write(filepath, wav_np.T, sample_rate)

        def _load_with_soundfile(filepath, **kwargs):
            """Use soundfile to load audio, avoiding torchcodec dependency."""
            wav, sr = sf.read(filepath, dtype='float32')
            # soundfile returns (samples, channels), torchaudio expects (channels, samples)
            if wav.ndim == 1:
                wav = wav.reshape(1, -1)
            else:
                wav = wav.T
            return torch.from_numpy(wav), sr

        torchaudio.save = _save_with_soundfile
        torchaudio.load = _load_with_soundfile
        return True
    except Exception as e:
        print(f"Failed to patch torchaudio: {e}")
        return False


def _separate_stems_in_process(audio_path: str, output_dir: str, model: str = "htdemucs"):
    """Run Demucs with torchaudio patch to avoid torchcodec issues.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated stems
        model: Demucs model name (default: htdemucs)

    Returns:
        dict of {stem_name: file_path} or None on failure
    """
    # Patch torchaudio before running Demucs
    patch_torchaudio()

    from demucs.pretrained import get_model
    from demucs.audio import save_audio
    from demucs.apply import apply_model
    import torchaudio
    import torch

    print(f"Loading Demucs model: {model}")
    model_obj = get_model(model)
    model_obj.eval()

    # Load audio
    print(f"Loading audio: {audio_path}")
    wav, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != model_obj.samplerate:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, model_obj.samplerate)
        wav = resampler(wav)
        sr = model_obj.samplerate

    # Ensure stereo
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    # Add batch dimension
    wav = wav.unsqueeze(0)

    # Apply model
    print("Separating stems...")
    with torch.no_grad():
        sources = apply_model(model_obj, wav, device="cpu")

    # sources shape: (1, num_sources, channels, samples)
    sources = sources[0]  # Remove batch dimension

    # Get source names
    source_names = model_obj.sources

    # Create output directory
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, model, basename)
    os.makedirs(stem_dir, exist_ok=True)

    # Save each stem
    stems = {}
    for i, name in enumerate(source_names):
        stem_path = os.path.join(stem_dir, f"{name}.wav")
        save_audio(sources[i], stem_path, sr)
        stems[name] = stem_path
        print(f"  Saved: {name}.wav")

    return stems


def _separate_stems_in_conda(audio_path: str, output_dir: str, model: str = "htdemucs"):
    """Run this wrapper inside the configured conda environment when the web Python lacks demucs."""
    conda_exe = os.environ.get("CONDA_EXE") or "conda"
    conda_env = os.environ.get("EDM_DEMUCS_CONDA_ENV", "edm-adapter")
    timeout = int(os.environ.get("EDM_DEMUCS_TIMEOUT_SEC", "3600"))
    cmd = [
        conda_exe,
        "run",
        "-n",
        conda_env,
        "python",
        os.path.abspath(__file__),
        "--json",
        audio_path,
        output_dir,
        "--model",
        model,
    ]
    env = os.environ.copy()
    env["EDM_DEMUCS_SUBPROCESS"] = "1"
    print(f"Demucs not available in {sys.executable}; running via conda env '{conda_env}'.")
    proc = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        print(f"Demucs subprocess failed: returncode={proc.returncode}", file=sys.stderr)
        return None
    marker = "DEMUCS_STEMS_JSON="
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    print("Demucs subprocess finished but did not report stems JSON.", file=sys.stderr)
    return None


def separate_stems(audio_path: str, output_dir: str, model: str = "htdemucs"):
    try:
        return _separate_stems_in_process(audio_path, output_dir, model=model)
    except ModuleNotFoundError as exc:
        if exc.name == "demucs" and os.environ.get("EDM_DEMUCS_SUBPROCESS") != "1":
            return _separate_stems_in_conda(audio_path, output_dir, model=model)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Separate stems with Demucs.")
    parser.add_argument("audio_file")
    parser.add_argument("output_dir", nargs="?", default="test_separation")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    stems = separate_stems(args.audio_file, args.output_dir, model=args.model)
    if stems:
        print("\nSeparated stems:")
        for name, path in stems.items():
            print(f"  {name}: {path}")
        if args.as_json:
            print("DEMUCS_STEMS_JSON=" + json.dumps(stems, ensure_ascii=False))
    else:
        print("Separation failed!")
        sys.exit(1)
