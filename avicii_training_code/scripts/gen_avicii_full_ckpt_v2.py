"""Generate using full checkpoint - clean approach."""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

import torch
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

    # Load checkpoint first to see its structure
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Check transformer keys
    transformer_keys = [k for k in state_dict.keys() if k.startswith("transformers.")]
    print(f"Transformer keys in checkpoint: {len(transformer_keys)}")

    # Load base model
    print("Loading ACE-Step base model...")
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)
    print("Base model loaded.")

    # The checkpoint has keys like: transformers.transformer_blocks.0.attn.to_q.lora_A.edm_shared.weight
    # The model expects: transformer_blocks.0.attn.to_q.lora_A.edm_shared.weight (without transformers. prefix)
    # But the model also has base_layer wrapping due to PEFT

    # Let's try to properly map the keys
    model_state = pipe.ace_step_transformer.state_dict()
    new_state = {}
    matched = 0
    for k, v in state_dict.items():
        if k.startswith("transformers."):
            model_key = k[len("transformers."):]
            # Check if this key exists in the model as-is
            if model_key in model_state:
                new_state[model_key] = v
                matched += 1
            # Also try with base_layer prefix for PEFT-wrapped layers
            elif "lora_" not in model_key:
                # Base weights - try to find the right key
                parts = model_key.split(".")
                # Insert base_layer before the layer name
                for i, part in enumerate(parts):
                    if part in ["to_q", "to_k", "to_v", "to_out", "linear_q", "linear_k", "linear_v", "linear_out"]:
                        new_key = ".".join(parts[:i+1]) + ".base_layer." + ".".join(parts[i+1:])
                        if new_key in model_state:
                            new_state[new_key] = v
                            matched += 1
                            break

    print(f"Matched keys: {matched} out of {len(transformer_keys)}")

    # Load the matched keys
    if new_state:
        missing, unexpected = pipe.ace_step_transformer.load_state_dict(new_state, strict=False)
        print(f"After loading: Missing={len(missing)}, Unexpected={len(unexpected)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = OUTPUT_DIR / f"full_ckpt_v2_2min_{stamp}.wav"

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
