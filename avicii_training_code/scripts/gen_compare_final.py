"""Generate baseline vs LoRA comparison with same prompt/seed."""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

from acestep.pipeline_ace_step import ACEStepPipeline

CHECKPOINT_DIR = str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "edm_control_lora" / "generations" / "avicii"
DURATION = 120.0
SEED = 42

PROMPT = "euphoric progressive house, 128 BPM, bright major key piano chords, catchy anthemic synth melody, soaring supersaw lead, sidechain pumping bass, four-on-the-floor kick, uplifting emotional energy, festival mainstage EDM, clean polished studio production, Swedish house music, Tim Bergling style, 128 BPM, high energy, drop, melodic house"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Generate BASELINE (no LoRA)
    print("=" * 50)
    print("1. Generating BASELINE (no LoRA)")
    print("=" * 50)
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)

    baseline_path = str(OUTPUT_DIR / f"baseline_{stamp}.wav")
    pipe(
        format="wav",
        audio_duration=DURATION,
        prompt=PROMPT,
        lyrics="",
        infer_step=100,
        guidance_scale=15.0,
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10.0,
        manual_seeds=[SEED],
        lora_name_or_path="none",
        lora_weight=1.0,
        save_path=baseline_path,
        batch_size=1,
        use_erg_tag=False,
    )
    print(f"Saved: {baseline_path}")

    # 2. Generate with LoRA + Control Conditioner (first training checkpoint)
    print("=" * 50)
    print("2. Generating with LoRA + Control Conditioner")
    print("=" * 50)

    from src.edm_control.control_curves import ControlCurveConfig, build_control_curve
    from src.edm_control.control_conditioner import EDMControlConditioner
    from src.edm_control.lora_router import EDMAdapterRouter, RouterConfig, load_router_manifest
    import torch

    LORA_BUNDLE = PROJECT_ROOT / "outputs" / "edm_control_lora" / "logs" / "lightning_logs" / "2026-05-19_22-43-57_edm_struct_lora" / "checkpoints" / "epoch=3-step=2000_edm_control_lora"

    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)

    # Load all adapters
    manifest_path = LORA_BUNDLE / "manifest.json"
    manifest = load_router_manifest(manifest_path)
    router = EDMAdapterRouter(RouterConfig.from_dict(manifest.get("router")))

    for spec in manifest.get("adapters", []):
        name = spec["name"]
        weight_path = LORA_BUNDLE / "adapters" / name / "pytorch_lora_weights.safetensors"
        if not weight_path.exists():
            continue
        pipe.ace_step_transformer.load_lora_adapter(
            str(weight_path), adapter_name=name, with_alpha=True, prefix=None,
        )

    # Apply routing weights
    row = {
        "section": "drop", "energy": "high", "subgenre": "melodic house",
        "bpm": 128.0, "bpm_confidence": 1.0, "sample_weight": 1.0,
        "tag_confidence": {"section": 1.0, "energy": 1.0, "subgenre": 1.0},
        "duration": DURATION, "quality_score": 5,
        "audio_features": {"low_freq_ratio": 0.12, "onset_density": 16.0},
    }
    weights = router.weights_for_batch([row])
    router.apply_to_model(pipe.ace_step_transformer, weights)
    print(f"Route weights: {weights}")

    # Load control conditioner
    payload_path = LORA_BUNDLE / "control_conditioner.pt"
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
    conditioner.to(device=pipe.device, dtype=torch.float32).eval()

    # Build control curve
    frame_count = max(1, int(DURATION * 44100 / 512 / 8))
    control, _ = build_control_curve(row, ControlCurveConfig(frame_count=frame_count))

    # Patch text embeddings
    original_get_text_embeddings = pipe.get_text_embeddings

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

    pipe.get_text_embeddings = patched_get_text_embeddings

    lora_path = str(OUTPUT_DIR / f"lora_control_{stamp}.wav")
    pipe(
        format="wav",
        audio_duration=DURATION,
        prompt=PROMPT,
        lyrics="",
        infer_step=100,
        guidance_scale=15.0,
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10.0,
        manual_seeds=[SEED],
        lora_name_or_path="none",
        lora_weight=1.0,
        save_path=lora_path,
        batch_size=1,
        use_erg_tag=False,
    )
    print(f"Saved: {lora_path}")

    print("\n" + "=" * 50)
    print("DONE!")
    print(f"Baseline: {baseline_path}")
    print(f"LoRA+Control: {lora_path}")


if __name__ == "__main__":
    main()
