"""Generate baseline and Avicii LoRA audio with the same seed."""

from __future__ import annotations

import argparse
import glob
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

import torchaudio_patch  # noqa: E402,F401 - patches torchaudio.save/load on Windows
from src.mel_watermark import save_ai_watermarked_mel  # noqa: E402

DEFAULT_PROMPT = (
    "uplifting progressive house, 128 BPM, bright piano chords, emotional melodic lead, "
    "sidechain bass, four-on-the-floor kick, wide supersaw chords, polished festival EDM drop"
)


def find_latest_bundle(root: Path) -> Path:
    candidates = [
        Path(path).parent
        for path in glob.glob(str(root / "**" / "manifest.json"), recursive=True)
    ]
    if not candidates:
        raise FileNotFoundError(f"No Avicii local LoRA manifest found under {root}")

    def sort_key(path: Path) -> tuple[int, float]:
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            step = int(manifest.get("global_step") or 0)
        except Exception:
            step = 0
        return step, path.stat().st_mtime

    return sorted(candidates, key=sort_key, reverse=True)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--infer-step", type=int, default=30)
    parser.add_argument("--lora-weight", type=float, default=1.6)
    parser.add_argument(
        "--adapter-root",
        default=str(PROJECT_ROOT / "outputs" / "avicii_local_lora" / "logs"),
    )
    parser.add_argument("--adapter-bundle", default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "avicii_local_lora" / "generations" / "same_seed_compare"),
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--format", default="wav")
    parser.add_argument("--guidance-scale", type=float, default=15.0)
    parser.add_argument("--omega-scale", type=float, default=10.0)
    parser.add_argument("--scheduler-type", default="euler")
    parser.add_argument("--cfg-type", default="apg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = Path(args.adapter_bundle) if args.adapter_bundle else find_latest_bundle(Path(args.adapter_root))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    adapter_name = manifest.get("adapter_name", "avicii_style")
    adapter_dir = bundle / "adapters" / adapter_name
    if not (adapter_dir / "pytorch_lora_weights.safetensors").exists():
        raise FileNotFoundError(f"Missing adapter weights under {adapter_dir}")

    style_prefix = manifest.get("style_prefix") or manifest.get("trigger_word") or "avicii_adapter_style"
    lora_prompt = args.prompt
    if not lora_prompt.startswith(style_prefix):
        lora_prompt = f"{style_prefix}, {lora_prompt}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_path = output_dir / f"baseline_seed{args.seed}_{int(args.duration)}s_{stamp}.{args.format}"
    lora_path = output_dir / f"avicii_lora_step{manifest.get('global_step')}_w{args.lora_weight:g}_seed{args.seed}_{int(args.duration)}s_{stamp}.{args.format}"

    from acestep.pipeline_ace_step import ACEStepPipeline

    pipe = ACEStepPipeline(checkpoint_dir=args.checkpoint_dir, dtype=args.dtype)
    pipe.load_checkpoint(args.checkpoint_dir)

    common = dict(
        format=args.format,
        audio_duration=args.duration,
        lyrics="[instrumental]",
        infer_step=args.infer_step,
        guidance_scale=args.guidance_scale,
        scheduler_type=args.scheduler_type,
        cfg_type=args.cfg_type,
        omega_scale=args.omega_scale,
        manual_seeds=[args.seed],
        batch_size=1,
        use_erg_tag=False,
        use_erg_lyric=False,
        use_erg_diffusion=False,
    )

    print(f"Generating baseline: {baseline_path}", flush=True)
    baseline_outputs = pipe(
        prompt=args.prompt,
        lora_name_or_path="none",
        lora_weight=1.0,
        save_path=str(baseline_path),
        **common,
    )

    print(f"Generating Avicii LoRA: {lora_path}", flush=True)
    lora_outputs = pipe(
        prompt=lora_prompt,
        lora_name_or_path=str(adapter_dir),
        lora_weight=args.lora_weight,
        save_path=str(lora_path),
        **common,
    )
    baseline_mel = save_ai_watermarked_mel(baseline_path, baseline_path.with_name(f"{baseline_path.stem}_mel_ai_watermark.png"))
    lora_mel = save_ai_watermarked_mel(lora_path, lora_path.with_name(f"{lora_path.stem}_mel_ai_watermark.png"))

    sidecar = {
        "seed": args.seed,
        "duration": args.duration,
        "infer_step": args.infer_step,
        "prompt": args.prompt,
        "lora_prompt": lora_prompt,
        "adapter_bundle": str(bundle),
        "adapter_dir": str(adapter_dir),
        "lora_weight": args.lora_weight,
        "baseline_outputs": baseline_outputs,
        "lora_outputs": lora_outputs,
        "baseline_mel_spectrogram": baseline_mel,
        "lora_mel_spectrogram": lora_mel,
    }
    sidecar_path = output_dir / f"same_seed_compare_seed{args.seed}_{int(args.duration)}s_{stamp}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(sidecar, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
