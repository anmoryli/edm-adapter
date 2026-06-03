"""Bridge script for calling Seed-VC from the Gradio app.

The app calls this script as an external process so the Seed-VC model is loaded
only for the conversion task and released when the process exits.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_VC_DIR = PROJECT_ROOT / "external_tools" / "seed-vc"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "1761704195865-bk9wgc-tomori1_e12_s2664"


def parse_optional_timeout(value: str | int | None) -> int | None:
    text = str(value or "").strip().lower()
    if text in {"", "0", "-1", "none", "null", "inf", "infinite", "unlimited", "无限", "无限制"}:
        return None
    seconds = int(float(text))
    return seconds if seconds > 0 else None


def _audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)


def select_reference_audio(model_dir: Path) -> Path:
    refs_dir = model_dir / "参考"
    if not refs_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {refs_dir}")

    candidates: list[tuple[float, Path]] = []
    for path in refs_dir.iterdir():
        if path.suffix.lower() not in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}:
            continue
        try:
            duration = _audio_duration(path)
        except Exception:
            continue
        if 1.0 <= duration <= 30.0:
            candidates.append((duration, path))

    if not candidates:
        raise RuntimeError(f"No usable 1-30 second reference audio found in: {refs_dir}")

    # Longer clean references usually carry more timbre information, while
    # Seed-VC clips references internally to 25 seconds.
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def run_seed_vc(args: argparse.Namespace, reference_audio: Path, inference_output_dir: Path) -> None:
    if not SEED_VC_DIR.exists():
        raise FileNotFoundError(f"Seed-VC directory not found: {SEED_VC_DIR}")

    command = [
        sys.executable,
        "inference.py",
        "--source",
        str(Path(args.source).resolve()),
        "--target",
        str(reference_audio.resolve()),
        "--output",
        str(inference_output_dir.resolve()),
        "--diffusion-steps",
        str(args.diffusion_steps),
        "--length-adjust",
        "1.0",
        "--inference-cfg-rate",
        str(args.inference_cfg_rate),
        "--f0-condition",
        "True" if args.f0_condition else "False",
        "--auto-f0-adjust",
        "False",
        "--semi-tone-shift",
        str(int(args.pitch)),
        "--fp16",
        "False",
    ]
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint

    print("Seed-VC command:")
    print(" ".join(command))
    proc = subprocess.run(
        command,
        cwd=str(SEED_VC_DIR),
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout,
    )
    (Path(args.task_dir) / "seed_vc_stdout.txt").write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    (Path(args.task_dir) / "seed_vc_stderr.txt").write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Seed-VC failed with return code {proc.returncode}")


def latest_wav(output_dir: Path) -> Path:
    wavs = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not wavs:
        raise FileNotFoundError(f"Seed-VC produced no wav files in: {output_dir}")
    return wavs[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--reference", default="")
    parser.add_argument("--diffusion-steps", type=int, default=int(os.environ.get("SEEDVC_DIFFUSION_STEPS", "30")))
    parser.add_argument("--inference-cfg-rate", type=float, default=float(os.environ.get("SEEDVC_CFG_RATE", "0.7")))
    parser.add_argument("--f0-condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=parse_optional_timeout, default=parse_optional_timeout(os.environ.get("SEEDVC_TIMEOUT_SEC", "0")))
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", ""))
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    inference_output_dir = task_dir / "seed_vc_output"
    inference_output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    reference_audio = Path(args.reference) if args.reference else select_reference_audio(model_dir)
    reference_audio = reference_audio.resolve()
    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    (task_dir / "seed_vc_reference.txt").write_text(str(reference_audio), encoding="utf-8")
    print(f"Using reference audio: {reference_audio}")

    run_seed_vc(args, reference_audio, inference_output_dir)
    generated = latest_wav(inference_output_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated, output_path)
    print(f"Copied Seed-VC output: {generated} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
