"""Generate Avicii-style: baseline vs style LoRA comparison."""

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
STYLE_LORA_DIR = str(PROJECT_ROOT / "outputs" / "avicii_style_lora" / "logs" / "lightning_logs" / "2026-05-20_02-59-17_avicii_style" / "checkpoints" / "epoch=7-step=4500_avicii_style" / "adapters" / "avicii_style" / "pytorch_lora_weights.safetensors")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "avicii_style_lora" / "generations"
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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate Baseline
    print("=" * 50)
    print("Generating BASELINE (no LoRA)...")
    print("=" * 50)
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)

    baseline_path = OUTPUT_DIR / f"avicii_baseline_{stamp}.wav"
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
        save_path=str(baseline_path),
        batch_size=1,
        use_erg_tag=False,
    )
    print(f"Baseline saved: {baseline_path}")

    # Generate with Style LoRA
    print("=" * 50)
    print("Generating with STYLE LoRA...")
    print("=" * 50)
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)

    # Load style LoRA
    print(f"Loading style LoRA from: {STYLE_LORA_DIR}")
    pipe.ace_step_transformer.load_lora_adapter(
        STYLE_LORA_DIR, adapter_name="avicii_style", with_alpha=True, prefix=None,
    )
    try:
        pipe.ace_step_transformer.set_adapters(["avicii_style"], adapter_weights=[0.8])
    except TypeError:
        pipe.ace_step_transformer.set_adapters(["avicii_style"], [0.8])

    lora_path = OUTPUT_DIR / f"avicii_style_lora_{stamp}.wav"
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
        save_path=str(lora_path),
        batch_size=1,
        use_erg_tag=False,
    )
    print(f"Style LoRA saved: {lora_path}")

    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"Baseline: {baseline_path}")
    print(f"Style LoRA: {lora_path}")


if __name__ == "__main__":
    main()
