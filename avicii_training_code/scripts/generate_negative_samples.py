"""Generate base model negative samples for preference training."""

import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

import torch
from datasets import load_from_disk
from acestep.pipeline_ace_step import ACEStepPipeline

DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "avicii_preference" / "negative_latents"
CHECKPOINT_DIR = str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    ds = load_from_disk(str(DATASET_PATH))
    print(f"数据集: {len(ds)} 样本")
    
    # Load base model
    print("加载 base model...")
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)
    
    # Generate negative samples for high-value samples first
    # Priority: drop > build-up > chorus > other
    priority_sections = ["drop", "build-up", "chorus"]
    
    samples_to_process = []
    for i, item in enumerate(ds):
        section = item.get("section", "")
        energy = item.get("energy", "")
        
        # Priority score
        if section == "drop" and energy in ["high", "very_high"]:
            priority = 3
        elif section in ["build-up", "chorus"]:
            priority = 2
        elif section == "drop":
            priority = 1
        else:
            priority = 0
        
        samples_to_process.append((i, priority, item))
    
    # Sort by priority (highest first)
    samples_to_process.sort(key=lambda x: -x[1])
    
    # Process top 500 samples (highest priority)
    num_to_process = min(500, len(samples_to_process))
    print(f"生成 {num_to_process} 个负样本...")
    
    processed = 0
    for idx, priority, item in tqdm(samples_to_process[:num_to_process], desc="生成负样本"):
        caption = item.get("caption", "")
        section = item.get("section", "")
        energy = item.get("energy", "")
        subgenre = item.get("subgenre", "")
        bpm = item.get("bpm", 128)
        seed = 42 + idx  # Unique seed per sample
        
        # Build prompt with section/energy/subgenre tags
        prompt = f"{caption}, {int(bpm)} BPM, {energy} energy, {section}, {subgenre}"
        
        try:
            # Generate with base model (no LoRA)
            output = pipe(
                format="wav",
                audio_duration=8.0,
                prompt=prompt,
                lyrics="",
                infer_step=50,  # Faster generation
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=[seed],
                lora_name_or_path="none",
                lora_weight=1.0,
                save_path=str(OUTPUT_DIR / f"negative_{idx:05d}.wav"),
                batch_size=1,
                use_erg_tag=False,
            )
            processed += 1
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            continue
    
    print(f"\n完成! 处理了 {processed} 个负样本")
    print(f"保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
