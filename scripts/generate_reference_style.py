"""Training-free reference-audio style generation with ACE-Step audio-to-audio."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ACE-Step").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Cannot find project root containing ACE-Step/ and src/")


PROJECT_ROOT = find_project_root()
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

import torchaudio_patch  # noqa: E402,F401
from src.audio_io import normalize_audio, save_audio  # noqa: E402
from src.generation import REFERENCE_STYLE_PROMPT, generate_acestep_reference_style, load_acestep_model  # noqa: E402


DEFAULT_PROMPT = REFERENCE_STYLE_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-audio", required=True, help="Path to the uploaded/reference audio file.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Optional content prompt; style comes from reference audio latent.")
    parser.add_argument("--lyrics", default="", help="Optional lyrics. Empty means instrumental/no vocals.")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--reference-start", type=float, default=0.0)
    parser.add_argument("--ref-strength", type=float, default=0.35, help="0.25-0.40 borrows timbre/style with a new melody; 0.85-0.95 keeps melody/chords.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer-step", type=int, default=140)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--no-auto-reference-start", action="store_true")
    parser.add_argument("--no-style-proxy", action="store_true", help="Disable non-melodic reference proxy and use the full reference latent directly.")
    parser.add_argument("--demucs-proxy", action="store_true", help="Try Demucs drums/bass/other separation before building the non-melodic proxy.")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "reference_style" / "generations"),
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"reference_style_seed{args.seed}_{int(args.duration)}s_{stamp}"
    raw_dir = run_dir / "ace_step_raw"
    final_path = run_dir / f"reference_style_seed{args.seed}_{int(args.duration)}s.wav"
    sidecar_path = run_dir / "metadata.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps({"args": vars(args), "run_dir": str(run_dir), "final_path": str(final_path)}, indent=2, ensure_ascii=False))
        return

    pipeline = load_acestep_model(
        checkpoint_dir=args.checkpoint_dir,
        device="auto",
        cpu_offload=True,
        dtype=args.dtype,
        cache_key="reference_style_base_cli",
    )
    pipeline.load_lora("none", 1.0)

    audio, sr, metadata = generate_acestep_reference_style(
        pipeline=pipeline,
        prompt=args.prompt,
        reference_audio_path=args.reference_audio,
        lyrics=args.lyrics,
        duration=args.duration,
        seed=args.seed,
        infer_step=args.infer_step,
        guidance_scale=args.guidance_scale,
        ref_audio_strength=args.ref_strength,
        reference_start=args.reference_start,
        auto_reference_start=not args.no_auto_reference_start,
        use_style_proxy=not args.no_style_proxy,
        use_demucs_proxy=args.demucs_proxy,
        save_dir=str(raw_dir),
    )
    audio = normalize_audio(audio, peak_db=-1.0)
    save_audio(str(final_path), audio, sr=sr)

    metadata["final_path"] = str(final_path)
    metadata["args"] = vars(args)
    sidecar_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
