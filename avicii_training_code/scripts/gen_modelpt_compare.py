"""Generate with full fine-tuned model.pt vs baseline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ACE-Step"))

import torch
from acestep.pipeline_ace_step import ACEStepPipeline

CHECKPOINT_DIR = str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B")
MODEL_PT = str(PROJECT_ROOT / "model.pt")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "edm_control_lora" / "generations" / "avicii"

PROMPT = "progressive house, piano chords, uplifting euphoric melody, anthemic synth lead, clean polished production, emotional, energetic drop, sidechain bass, supersaw chords, festival EDM, Swedish house style, 128 BPM, high energy, drop, melodic house"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Baseline
    print("=" * 50)
    print("1. Generating BASELINE")
    print("=" * 50)
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="float32")
    pipe.load_checkpoint(CHECKPOINT_DIR)
    pipe(
        format="wav", audio_duration=120.0, prompt=PROMPT, lyrics="",
        infer_step=100, guidance_scale=15.0, scheduler_type="euler",
        cfg_type="apg", omega_scale=10.0, manual_seeds=[42],
        lora_name_or_path="none", lora_weight=1.0,
        save_path=str(OUTPUT_DIR / "baseline_modelpt_20260520.wav"),
        batch_size=1, use_erg_tag=False,
    )
    print("Baseline saved")

    # 2. Full fine-tuned model
    print("=" * 50)
    print("2. Generating with FULL MODEL (model.pt)")
    print("=" * 50)
    pipe2 = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="float32")
    pipe2.load_checkpoint(CHECKPOINT_DIR)

    print(f"Loading model.pt from: {MODEL_PT}")
    ckpt = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
    missing, unexpected = pipe2.ace_step_transformer.load_state_dict(ckpt, strict=False)
    print(f"Loaded: {len(ckpt)} keys, missing={len(missing)}, unexpected={len(unexpected)}")

    pipe2(
        format="wav", audio_duration=120.0, prompt=PROMPT, lyrics="",
        infer_step=100, guidance_scale=15.0, scheduler_type="euler",
        cfg_type="apg", omega_scale=10.0, manual_seeds=[42],
        lora_name_or_path="none", lora_weight=1.0,
        save_path=str(OUTPUT_DIR / "fullmodel_20260520.wav"),
        batch_size=1, use_erg_tag=False,
    )
    print("Full model saved")

    print("\n" + "=" * 50)
    print("DONE!")
    print(f"Baseline: {OUTPUT_DIR / 'baseline_modelpt_20260520.wav'}")
    print(f"Full model: {OUTPUT_DIR / 'fullmodel_20260520.wav'}")


if __name__ == "__main__":
    main()
