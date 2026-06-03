"""Generate with v3 LoRA at different scales."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

import torch
from acestep.pipeline_ace_step import ACEStepPipeline

CHECKPOINT_DIR = str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B")
ADAPTER_PATH = str(PROJECT_ROOT / "outputs" / "avicii_style_v3" / "logs" / "lightning_logs")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v3_test"

TRIGGER = "avicii_adapter_style"
PROMPT = "progressive house, piano chords, uplifting euphoric melody, anthemic synth lead, clean polished production, emotional, energetic drop, sidechain bass, supersaw chords, festival EDM, Swedish house style"
TRIGGER_PROMPT = f"{TRIGGER}, {PROMPT}"

import glob
adapter_dirs = sorted(glob.glob(ADAPTER_PATH + "/*v3*/checkpoints/step=10000_avicii_style"))
if not adapter_dirs:
    print("No adapter found!")
    sys.exit(1)
ADAPTER_DIR = adapter_dirs[0] + "/adapters/avicii_style/pytorch_lora_weights.safetensors"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline
    print("=== Baseline ===")
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="float32")
    pipe.load_checkpoint(CHECKPOINT_DIR)
    pipe(format="wav", audio_duration=120.0, prompt=PROMPT, lyrics="",
         infer_step=100, guidance_scale=15.0, scheduler_type="euler",
         cfg_type="apg", omega_scale=10.0, manual_seeds=[42],
         lora_name_or_path="none", lora_weight=1.0,
         save_path=str(OUTPUT_DIR / "baseline.wav"), batch_size=1, use_erg_tag=False)

    # LoRA at different scales
    for scale in [1.0, 2.0, 3.0]:
        print(f"\n=== LoRA scale={scale} ===")
        pipe2 = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="float32")
        pipe2.load_checkpoint(CHECKPOINT_DIR)
        pipe2.ace_step_transformer.load_lora_adapter(ADAPTER_DIR, adapter_name="avicii_style", with_alpha=True, prefix=None)
        try:
            pipe2.ace_step_transformer.set_adapters(["avicii_style"], adapter_weights=[scale])
        except:
            pipe2.ace_step_transformer.set_adapters(["avicii_style"], [scale])
        pipe2(format="wav", audio_duration=120.0, prompt=TRIGGER_PROMPT, lyrics="",
              infer_step=100, guidance_scale=15.0, scheduler_type="euler",
              cfg_type="apg", omega_scale=10.0, manual_seeds=[42],
              lora_name_or_path="none", lora_weight=1.0,
              save_path=str(OUTPUT_DIR / f"lora_scale{scale}.wav"), batch_size=1, use_erg_tag=False)

    print("\n=== DONE ===")
    print(f"Baseline: {OUTPUT_DIR / 'baseline.wav'}")
    for s in [1.0, 2.0, 3.0]:
        print(f"LoRA scale={s}: {OUTPUT_DIR / f'lora_scale{s}.wav'}")

if __name__ == "__main__":
    main()
