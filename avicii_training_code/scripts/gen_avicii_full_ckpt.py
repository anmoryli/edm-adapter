"""Generate Avicii-style using full checkpoint (not manual LoRA loading)."""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

from acestep.pipeline_ace_step import ACEStepPipeline

CHECKPOINT_DIR = str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B")
CKPT_PATH = str(PROJECT_ROOT / "outputs" / "edm_control_lora" / "logs" / "lightning_logs" / "2026-05-19_22-43-57_edm_struct_lora" / "checkpoints" / "epoch=3-step=2000.ckpt")
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ACE-Step model...")
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)
    print("Base model loaded.")

    # Load the full Lightning checkpoint which has LoRA weights baked in
    print(f"Loading full checkpoint: {CKPT_PATH}")
    import torch
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Check what keys are in the state dict
    lora_keys = [k for k in state_dict.keys() if "lora" in k.lower()]
    print(f"Total state_dict keys: {len(state_dict.keys())}")
    print(f"LoRA-related keys: {len(lora_keys)}")
    if lora_keys:
        print(f"Sample LoRA keys: {lora_keys[:5]}")

    # Load the state dict into the transformer
    transformer_state = {}
    for k, v in state_dict.items():
        if k.startswith("transformers."):
            new_key = k[len("transformers."):]
            transformer_state[new_key] = v

    if transformer_state:
        print(f"Loading {len(transformer_state)} transformer keys from checkpoint...")
        missing, unexpected = pipe.ace_step_transformer.load_state_dict(transformer_state, strict=False)
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Sample missing: {missing[:5]}")
        if unexpected:
            print(f"  Sample unexpected: {unexpected[:5]}")

    # Also load control conditioner if present
    conditioner_state = {}
    for k, v in state_dict.items():
        if k.startswith("control_conditioner."):
            new_key = k[len("control_conditioner."):]
            conditioner_state[new_key] = v

    if conditioner_state:
        print(f"Loading {len(conditioner_state)} control conditioner keys...")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = OUTPUT_DIR / f"full_ckpt_2min_{stamp}.wav"

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
    print(f"Done: {output_paths[0] if isinstance(output_paths, list) else output_paths}")


if __name__ == "__main__":
    main()
