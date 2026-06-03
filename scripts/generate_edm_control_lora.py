"""Generate music with an EDM-StructLoRA adapter bundle."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

from src.edm_control.control_curves import ControlCurveConfig, build_control_curve  # noqa: E402
from src.edm_control.lora_router import EDMAdapterRouter, RouterConfig, load_router_manifest  # noqa: E402


def find_manifest(bundle: Path) -> Path:
    if bundle.is_file():
        return bundle
    manifest = bundle / "manifest.json"
    if manifest.exists():
        return manifest
    candidates = sorted(bundle.glob("**/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No manifest.json found under {bundle}")
    return candidates[0]


def build_prompt(args: argparse.Namespace) -> str:
    prompt = args.prompt.strip()
    if args.append_control_tags:
        tags = f"{int(round(args.bpm))} BPM, {args.energy} energy, {args.section}, {args.subgenre}"
        prompt = f"{prompt}, {tags}" if prompt else tags
    return prompt


def route_row(args: argparse.Namespace) -> dict:
    return {
        "section": args.section,
        "energy": args.energy,
        "subgenre": args.subgenre,
        "bpm": args.bpm,
        "bpm_confidence": 1.0,
        "sample_weight": 1.0,
        "tag_confidence": {"section": 1.0, "energy": 1.0, "subgenre": 1.0},
        "duration": args.duration,
        "quality_score": 5,
        "audio_features": {
            "low_freq_ratio": args.low_freq_ratio,
            "onset_density": args.onset_density,
        },
    }


def dry_run(args: argparse.Namespace) -> None:
    manifest_path = None
    manifest = {"router": {}}
    if args.lora_bundle:
        try:
            manifest_path = find_manifest(Path(args.lora_bundle))
            manifest = load_router_manifest(manifest_path)
        except FileNotFoundError:
            config_path = PROJECT_ROOT / "config" / "edm_control_lora.json"
            if config_path.exists():
                manifest = json.loads(config_path.read_text(encoding="utf-8"))
    router = EDMAdapterRouter(RouterConfig.from_dict(manifest.get("router")))
    row = route_row(args)
    print(json.dumps({
        "prompt": build_prompt(args),
        "manifest": str(manifest_path) if manifest_path else None,
        "route_weights": router.weights_for_batch([row]),
        "duration": args.duration,
        "output_dir": args.output_dir,
    }, indent=2, ensure_ascii=False))


def load_adapter_bundle(pipe, manifest_path: Path, args: argparse.Namespace) -> dict[str, float]:
    manifest = load_router_manifest(manifest_path)
    bundle_dir = manifest_path.parent
    router = EDMAdapterRouter(RouterConfig.from_dict(manifest.get("router")))
    for spec in manifest.get("adapters", []):
        name = spec["name"]
        weight_path = bundle_dir / "adapters" / name / "pytorch_lora_weights.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing adapter weights: {weight_path}")
        pipe.ace_step_transformer.load_lora_adapter(
            str(weight_path),
            adapter_name=name,
            with_alpha=True,
            prefix=None,
        )
    weights = router.weights_for_batch([route_row(args)])
    router.apply_to_model(pipe.ace_step_transformer, weights)
    return weights


def install_control_conditioner_patch(pipe, manifest_path: Path, args: argparse.Namespace) -> bool:
    import torch

    from src.edm_control.control_conditioner import EDMControlConditioner

    payload_path = manifest_path.parent / "control_conditioner.pt"
    if not payload_path.exists():
        return False
    payload = torch.load(payload_path, map_location="cpu")
    config = payload["config"]
    conditioner = EDMControlConditioner(
        feature_dim=int(config["feature_dim"]),
        text_embed_dim=int(config.get("text_embed_dim", 768)),
        token_count=int(config.get("token_count", 8)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        dropout=0.0,
    )
    conditioner.load_state_dict(payload["state_dict"], strict=True)
    conditioner.to(device=pipe.device, dtype=pipe.dtype).eval()
    row = route_row(args)
    frame_count = max(1, int(args.duration * 44100 / 512 / 8))
    control, _ = build_control_curve(row, ControlCurveConfig(frame_count=frame_count))

    original_get_text_embeddings = pipe.get_text_embeddings
    original_get_text_embeddings_null = pipe.get_text_embeddings_null

    def patched_get_text_embeddings(texts, text_max_length=256):
        hidden, mask = original_get_text_embeddings(texts, text_max_length=text_max_length)
        batch = hidden.shape[0]
        controls = control.unsqueeze(0).repeat(batch, 1, 1).to(device=pipe.device, dtype=torch.float32)
        with torch.no_grad():
            control_tokens = conditioner(controls).to(device=pipe.device, dtype=hidden.dtype)
        hidden = torch.cat([hidden, control_tokens], dim=1)
        extra_mask = torch.ones(batch, control_tokens.shape[1], device=mask.device, dtype=mask.dtype)
        mask = torch.cat([mask, extra_mask], dim=1)
        return hidden, mask

    def patched_get_text_embeddings_null(texts, text_max_length=256, tau=0.01, l_min=8, l_max=10):
        hidden = original_get_text_embeddings_null(
            texts,
            text_max_length=text_max_length,
            tau=tau,
            l_min=l_min,
            l_max=l_max,
        )
        batch = hidden.shape[0]
        zeros = torch.zeros(
            batch,
            conditioner.token_count,
            conditioner.text_embed_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return torch.cat([hidden, zeros], dim=1)

    pipe.get_text_embeddings = patched_get_text_embeddings
    pipe.get_text_embeddings_null = patched_get_text_embeddings_null
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lora-bundle", required=True)
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "edm_control_lora" / "generations"))
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--infer-step", type=int, default=60)
    parser.add_argument("--guidance-scale", type=float, default=15.0)
    parser.add_argument("--omega-scale", type=float, default=10.0)
    parser.add_argument("--scheduler-type", default="euler")
    parser.add_argument("--cfg-type", default="apg")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--format", default="wav")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--section", default="drop")
    parser.add_argument("--energy", default="high")
    parser.add_argument("--subgenre", default="melodic house")
    parser.add_argument("--bpm", type=float, default=128.0)
    parser.add_argument("--low-freq-ratio", type=float, default=0.12)
    parser.add_argument("--onset-density", type=float, default=16.0)
    parser.add_argument("--append-control-tags", action="store_true")
    parser.add_argument("--use-erg-tag", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    from acestep.pipeline_ace_step import ACEStepPipeline

    manifest_path = find_manifest(Path(args.lora_bundle))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"edm_control_{args.section}_{int(round(args.bpm))}bpm_{stamp}.{args.format}"

    pipe = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_dir,
        device_id=args.device_id,
        dtype=args.dtype,
    )
    pipe.load_checkpoint(args.checkpoint_dir)
    route_weights = load_adapter_bundle(pipe, manifest_path, args)
    control_patched = install_control_conditioner_patch(pipe, manifest_path, args)
    prompt = build_prompt(args)
    output_paths = pipe(
        format=args.format,
        audio_duration=args.duration,
        prompt=prompt,
        lyrics="",
        infer_step=args.infer_step,
        guidance_scale=args.guidance_scale,
        scheduler_type=args.scheduler_type,
        cfg_type=args.cfg_type,
        omega_scale=args.omega_scale,
        manual_seeds=[args.seed],
        lora_name_or_path="none",
        lora_weight=1.0,
        save_path=str(save_path),
        batch_size=args.batch_size,
        use_erg_tag=args.use_erg_tag,
    )
    sidecar = {
        "prompt": prompt,
        "manifest": str(manifest_path),
        "output_paths": output_paths,
        "route_weights": route_weights,
        "control_conditioner_used": control_patched,
        "args": vars(args),
    }
    sidecar_path = save_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(sidecar, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
