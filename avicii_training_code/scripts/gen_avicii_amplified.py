"""Generate Avicii with amplified LoRA weights to test if LoRA is working."""

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


def route_row():
    return {
        "section": "drop",
        "energy": "high",
        "subgenre": "melodic house",
        "bpm": 128.0,
        "bpm_confidence": 1.0,
        "sample_weight": 1.0,
        "tag_confidence": {"section": 1.0, "energy": 1.0, "subgenre": 1.0},
        "duration": DURATION,
        "quality_score": 5,
        "audio_features": {"low_freq_ratio": 0.12, "onset_density": 16.0},
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)

    # Load LoRA adapters
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

    # Use AMPLIFIED weights to see if LoRA has any effect
    # Normal: edm_shared=1.0, section_drop=0.9, energy_high=0.35, subgenre_melodic_house=0.45
    # Amplified: multiply all by 5x
    amplified_weights = {
        "edm_shared": 5.0,
        "section_drop": 4.5,
        "energy_high": 1.75,
        "subgenre_melodic_house": 2.25,
    }
    print(f"Amplified route weights: {amplified_weights}")

    # Apply amplified weights via router
    router.apply_to_model(pipe.ace_step_transformer, amplified_weights)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = OUTPUT_DIR / f"amplified_lora_2min_{stamp}.wav"

    print(f"Generating to {save_path}...")
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
    print(f"Done: {save_path}")


if __name__ == "__main__":
    main()
