"""Generate with the latest Avicii local LoRA adapter."""

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

DEFAULT_STYLE_PROMPT = (
    "uplifting progressive house, 128 BPM, bright piano chords, emotional melodic lead, "
    "sidechain bass, four-on-the-floor kick, wide supersaw chords, polished festival EDM drop"
)


def find_latest_bundle(root: Path) -> Path:
    if (root / "manifest.json").exists():
        return root
    candidates = [
        Path(path).parent
        for path in glob.glob(str(root / "**" / "manifest.json"), recursive=True)
    ]
    if not candidates:
        raise FileNotFoundError(f"No Avicii local LoRA manifest found under {root}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def load_manifest(bundle: Path) -> dict:
    manifest_path = bundle / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def build_prompt(prompt: str, manifest: dict, disable_trigger: bool) -> str:
    prompt = prompt.strip()
    if disable_trigger:
        return prompt
    style_prefix = manifest.get("style_prefix") or "avicii_adapter_style"
    if prompt.startswith(style_prefix):
        return prompt
    return f"{style_prefix}, {prompt}" if prompt else style_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_STYLE_PROMPT)
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
        default=str(PROJECT_ROOT / "outputs" / "avicii_local_lora" / "generations"),
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--infer-step", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=15.0)
    parser.add_argument("--omega-scale", type=float, default=10.0)
    parser.add_argument("--scheduler-type", default="euler")
    parser.add_argument("--cfg-type", default="apg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-weight", type=float, default=1.6)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--format", default="wav")
    parser.add_argument("--disable-trigger", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = Path(args.adapter_bundle) if args.adapter_bundle else find_latest_bundle(Path(args.adapter_root))
    manifest = load_manifest(bundle)
    adapter_dir = bundle / "adapters" / manifest.get("adapter_name", "avicii_style")
    weight_file = adapter_dir / "pytorch_lora_weights.safetensors"
    if not weight_file.exists():
        raise FileNotFoundError(f"Missing adapter weights: {weight_file}")

    prompt = build_prompt(args.prompt, manifest, args.disable_trigger)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"avicii_local_lora_{stamp}.{args.format}"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "bundle": str(bundle),
                    "adapter_dir": str(adapter_dir),
                    "prompt": prompt,
                    "save_path": str(save_path),
                    "lora_weight": args.lora_weight,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    from acestep.pipeline_ace_step import ACEStepPipeline

    pipe = ACEStepPipeline(checkpoint_dir=args.checkpoint_dir, dtype=args.dtype)
    pipe.load_checkpoint(args.checkpoint_dir)
    output_paths = pipe(
        format=args.format,
        audio_duration=args.duration,
        prompt=prompt,
        lyrics="[instrumental]",
        infer_step=args.infer_step,
        guidance_scale=args.guidance_scale,
        scheduler_type=args.scheduler_type,
        cfg_type=args.cfg_type,
        omega_scale=args.omega_scale,
        manual_seeds=[args.seed],
        lora_name_or_path=str(adapter_dir),
        lora_weight=args.lora_weight,
        save_path=str(save_path),
        batch_size=1,
        use_erg_tag=False,
        use_erg_lyric=False,
        use_erg_diffusion=False,
    )
    mel_paths = []
    for output_path in output_paths or []:
        if isinstance(output_path, str) and output_path.lower().endswith(".wav") and Path(output_path).exists():
            mel_path = Path(output_path).with_name(f"{Path(output_path).stem}_mel_ai_watermark.png")
            mel_paths.append(save_ai_watermarked_mel(output_path, mel_path))
    sidecar = {
        "bundle": str(bundle),
        "adapter_dir": str(adapter_dir),
        "manifest": manifest,
        "prompt": prompt,
        "output_paths": output_paths,
        "mel_spectrogram_paths": mel_paths,
        "args": vars(args),
    }
    sidecar_path = save_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(sidecar, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
