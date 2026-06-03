"""Generate Avicii-style 2min tracks with improved prompt: baseline + LoRA."""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

import torch
from acestep.pipeline_ace_step import ACEStepPipeline
from src.edm_control.control_curves import ControlCurveConfig, build_control_curve
from src.edm_control.control_conditioner import EDMControlConditioner
from src.edm_control.lora_router import EDMAdapterRouter, RouterConfig, load_router_manifest

CHECKPOINT_DIR = str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B")
LORA_BUNDLE = PROJECT_ROOT / "outputs" / "edm_control_lora" / "logs" / "lightning_logs" / "2026-05-19_22-43-57_edm_struct_lora" / "checkpoints" / "epoch=3-step=2000_edm_control_lora"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "edm_control_lora" / "generations" / "avicii"
DURATION = 120.0
SEED = 42

PROMPT = (
    "euphoric progressive house, 128 BPM, bright major key piano chords, "
    "catchy anthemic synth melody, soaring supersaw lead, sidechain pumping bass, "
    "four-on-the-floor kick, uplifting emotional energy, festival mainstage EDM, "
    "clean polished studio production, reverb-drenched plucks, layered synth stacks, "
    "energetic buildup and explosive drop, Swedish house music, Tim Bergling style, "
    "acoustic piano intro leading to electronic drop, feel-good anthem"
)

SECTION = "drop"
ENERGY = "high"
SUBGENRE = "melodic house"
BPM = 128.0


def route_row():
    return {
        "section": SECTION,
        "energy": ENERGY,
        "subgenre": SUBGENRE,
        "bpm": BPM,
        "bpm_confidence": 1.0,
        "sample_weight": 1.0,
        "tag_confidence": {"section": 1.0, "energy": 1.0, "subgenre": 1.0},
        "duration": DURATION,
        "quality_score": 5,
        "audio_features": {"low_freq_ratio": 0.12, "onset_density": 16.0},
    }


def install_control_conditioner(pipe, manifest_path):
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
    conditioner.to(device=pipe.device, dtype=torch.float32).eval()

    row = route_row()
    frame_count = max(1, int(DURATION * 44100 / 512 / 8))
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
        hidden = original_get_text_embeddings_null(texts, text_max_length=text_max_length, tau=tau, l_min=l_min, l_max=l_max)
        batch = hidden.shape[0]
        zeros = torch.zeros(batch, conditioner.token_count, conditioner.text_embed_dim, device=hidden.device, dtype=hidden.dtype)
        return torch.cat([hidden, zeros], dim=1)

    pipe.get_text_embeddings = patched_get_text_embeddings
    pipe.get_text_embeddings_null = patched_get_text_embeddings_null
    return True


def generate_baseline(pipe):
    print("\n=== Generating BASELINE (no LoRA) ===")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = OUTPUT_DIR / f"baseline_2min_v2_{stamp}.wav"

    output_paths = pipe(
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
        save_path=str(save_path),
        batch_size=1,
        use_erg_tag=False,
    )
    print(f"  Baseline saved: {output_paths[0] if isinstance(output_paths, list) else output_paths}")
    return output_paths


def generate_lora(pipe):
    print("\n=== Generating LoRA (edm_control_lora) ===")
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

    row = route_row()
    weights = router.weights_for_batch([row])
    router.apply_to_model(pipe.ace_step_transformer, weights)
    print(f"  Route weights: {weights}")

    install_control_conditioner(pipe, manifest_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = OUTPUT_DIR / f"lora_2min_v2_{stamp}.wav"

    output_paths = pipe(
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
        save_path=str(save_path),
        batch_size=1,
        use_erg_tag=False,
    )
    print(f"  LoRA saved: {output_paths[0] if isinstance(output_paths, list) else output_paths}")
    return output_paths


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ACE-Step model...")
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)
    print("Model loaded.")

    generate_baseline(pipe)

    print("\nReloading model for LoRA generation...")
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)

    generate_lora(pipe)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
