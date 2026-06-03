"""Generate baseline, previous LoRA, and candidate LoRA comparisons."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


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
from src.mel_watermark import save_ai_watermarked_mel  # noqa: E402


SECTION_PROMPTS = {
    "intro": (
        "progressive house intro, 128 BPM, filtered piano chords, soft pluck motif, "
        "light percussion, gradual sidechain pulse, clean wide EDM mix"
    ),
    "breakdown": (
        "emotional progressive house breakdown, 128 BPM, warm piano chord progression, "
        "sparse drums, uplifting melody, acoustic pluck texture, clean festival mix"
    ),
    "drop": (
        "euphoric progressive house drop, 128 BPM, bright piano chord hits, wide supersaw lead hook, "
        "sidechain bass, four-on-the-floor kick, polished festival EDM drop"
    ),
}


def bundle_sort_key(path: Path) -> tuple[int, float]:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        step = int(manifest.get("global_step") or 0)
    except Exception:
        step = 0
    return step, path.stat().st_mtime


def find_bundles(root: Path) -> list[Path]:
    candidates = [
        Path(path).parent
        for path in glob.glob(str(root / "**" / "manifest.json"), recursive=True)
    ]
    return sorted(candidates, key=bundle_sort_key, reverse=True)


def resolve_bundle(path: str | None, root: Path, fallback_index: int) -> Path:
    if path:
        bundle = Path(path)
        if not bundle.is_absolute():
            bundle = (PROJECT_ROOT / bundle).resolve()
        return bundle
    bundles = find_bundles(root)
    if len(bundles) <= fallback_index:
        raise FileNotFoundError(f"Need at least {fallback_index + 1} LoRA bundle(s) under {root}")
    return bundles[fallback_index]


def load_bundle(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter_name = manifest.get("adapter_name", "avicii_style")
    adapter_dir = bundle / "adapters" / adapter_name
    if not (adapter_dir / "pytorch_lora_weights.safetensors").exists():
        raise FileNotFoundError(f"Missing adapter weights under {adapter_dir}")
    return {
        "bundle": bundle,
        "manifest": manifest,
        "adapter_dir": adapter_dir,
        "style_prefix": manifest.get("style_prefix") or manifest.get("trigger_word") or "avicii_adapter_style",
    }


def with_style_prefix(prompt: str, style_prefix: str) -> str:
    if prompt.startswith(style_prefix):
        return prompt
    return f"{style_prefix}, {prompt}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", default="intro,breakdown,drop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--infer-step", type=int, default=30)
    parser.add_argument("--lora-weight", type=float, default=1.6)
    parser.add_argument("--reference-lora-weight", type=float, default=None)
    parser.add_argument("--candidate-lora-weight", type=float, default=None)
    parser.add_argument("--reference-adapter-bundle", default=None)
    parser.add_argument("--candidate-adapter-bundle", default=None)
    parser.add_argument(
        "--adapter-root",
        default=str(PROJECT_ROOT / "outputs" / "avicii_local_lora" / "logs"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "avicii_local_lora" / "generations" / "section15_triple_compare"),
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
    adapter_root = Path(args.adapter_root)
    candidate_bundle = resolve_bundle(args.candidate_adapter_bundle, adapter_root, fallback_index=0)
    reference_bundle = resolve_bundle(args.reference_adapter_bundle, adapter_root, fallback_index=1)
    candidate = load_bundle(candidate_bundle)
    reference = load_bundle(reference_bundle)

    ref_weight = args.reference_lora_weight if args.reference_lora_weight is not None else args.lora_weight
    cand_weight = args.candidate_lora_weight if args.candidate_lora_weight is not None else args.lora_weight

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    sections = [item.strip() for item in args.sections.split(",") if item.strip()]
    results: list[dict[str, Any]] = []
    for section in sections:
        prompt = SECTION_PROMPTS.get(section, section)
        safe_section = section.replace("-", "")
        baseline_path = output_dir / f"{safe_section}_baseline_seed{args.seed}_{int(args.duration)}s_{stamp}.{args.format}"
        reference_path = output_dir / f"{safe_section}_reference_step{reference['manifest'].get('global_step')}_seed{args.seed}_{int(args.duration)}s_{stamp}.{args.format}"
        candidate_path = output_dir / f"{safe_section}_candidate_step{candidate['manifest'].get('global_step')}_seed{args.seed}_{int(args.duration)}s_{stamp}.{args.format}"

        print(f"Generating {section} baseline: {baseline_path}", flush=True)
        baseline_outputs = pipe(
            prompt=prompt,
            lora_name_or_path="none",
            lora_weight=1.0,
            save_path=str(baseline_path),
            **common,
        )

        print(f"Generating {section} reference LoRA: {reference_path}", flush=True)
        reference_outputs = pipe(
            prompt=with_style_prefix(prompt, reference["style_prefix"]),
            lora_name_or_path=str(reference["adapter_dir"]),
            lora_weight=ref_weight,
            save_path=str(reference_path),
            **common,
        )

        print(f"Generating {section} candidate LoRA: {candidate_path}", flush=True)
        candidate_outputs = pipe(
            prompt=with_style_prefix(prompt, candidate["style_prefix"]),
            lora_name_or_path=str(candidate["adapter_dir"]),
            lora_weight=cand_weight,
            save_path=str(candidate_path),
            **common,
        )
        baseline_mel = save_ai_watermarked_mel(baseline_path, baseline_path.with_name(f"{baseline_path.stem}_mel_ai_watermark.png"))
        reference_mel = save_ai_watermarked_mel(reference_path, reference_path.with_name(f"{reference_path.stem}_mel_ai_watermark.png"))
        candidate_mel = save_ai_watermarked_mel(candidate_path, candidate_path.with_name(f"{candidate_path.stem}_mel_ai_watermark.png"))

        results.append(
            {
                "section": section,
                "prompt": prompt,
                "baseline_path": str(baseline_path),
                "reference_path": str(reference_path),
                "candidate_path": str(candidate_path),
                "baseline_outputs": baseline_outputs,
                "reference_outputs": reference_outputs,
                "candidate_outputs": candidate_outputs,
                "baseline_mel_spectrogram": baseline_mel,
                "reference_mel_spectrogram": reference_mel,
                "candidate_mel_spectrogram": candidate_mel,
            }
        )

    sidecar = {
        "seed": args.seed,
        "duration": args.duration,
        "infer_step": args.infer_step,
        "reference_bundle": str(reference["bundle"]),
        "candidate_bundle": str(candidate["bundle"]),
        "reference_weight": ref_weight,
        "candidate_weight": cand_weight,
        "results": results,
    }
    sidecar_path = output_dir / f"triple_compare_seed{args.seed}_{int(args.duration)}s_{stamp}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(sidecar, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
