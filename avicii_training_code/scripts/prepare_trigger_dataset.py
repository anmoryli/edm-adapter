"""Step 1: Create trigger word dataset and canary subset for overfitting test."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

from datasets import load_from_disk

TRIGGER_WORD = "avicii_adapter_style"
DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "avicii_trigger_dataset"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    ds = load_from_disk(str(DATASET_PATH))
    print(f"原始数据集: {len(ds)} 样本")
    
    # Step 1: Select canary samples (drop/chorus, high/very_high energy)
    canary_indices = []
    for i, item in enumerate(ds):
        section = item.get("section", "")
        energy = item.get("energy", "")
        subgenre = item.get("subgenre", "")
        
        #优先选择最能代表 Avicii 风格的片段
        if section == "drop" and energy in ["high", "very_high"]:
            if subgenre in ["melodic house", "progressive house", "festival EDM"]:
                canary_indices.append(i)
    
    # Limit to 64
    canary_indices = canary_indices[:64]
    print(f"Canary 样本数: {len(canary_indices)}")
    
    # Step 2: Create trigger word captions
    # For canary test, we modify the caption to include trigger word
    canary_captions = []
    for idx in canary_indices:
        item = ds[idx]
        original_caption = item.get("caption", "")
        section = item.get("section", "")
        energy = item.get("energy", "")
        subgenre = item.get("subgenre", "")
        bpm = item.get("bpm", 128)
        
        # Add trigger word to caption
        trigger_caption = f"{TRIGGER_WORD}, {original_caption}"
        
        canary_captions.append({
            "index": idx,
            "original_caption": original_caption,
            "trigger_caption": trigger_caption,
            "section": section,
            "energy": energy,
            "subgenre": subgenre,
            "bpm": bpm,
        })
    
    # Save canary config
    canary_config = {
        "trigger_word": TRIGGER_WORD,
        "num_samples": len(canary_indices),
        "indices": canary_indices,
        "captions": canary_captions,
    }
    
    config_path = OUTPUT_DIR / "canary_config.json"
    config_path.write_text(json.dumps(canary_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Canary 配置保存到: {config_path}")
    
    # Step 3: Create full dataset config with trigger word
    full_captions = []
    for i, item in enumerate(ds):
        original_caption = item.get("caption", "")
        trigger_caption = f"{TRIGGER_WORD}, {original_caption}"
        full_captions.append({
            "index": i,
            "trigger_caption": trigger_caption,
        })
    
    full_config = {
        "trigger_word": TRIGGER_WORD,
        "num_samples": len(ds),
        "captions": full_captions[:100],  # Save first 100 for reference
    }
    
    full_path = OUTPUT_DIR / "full_dataset_config.json"
    full_path.write_text(json.dumps(full_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"完整数据集配置保存到: {full_path}")
    
    # Print sample captions
    print("\n=== Canary 样本示例 ===")
    for cap in canary_captions[:5]:
        print(f"#{cap['index']}: {cap['section']}/{cap['energy']}/{cap['subgenre']}")
        print(f"  原始: {cap['original_caption'][:100]}...")
        print(f"  触发: {cap['trigger_caption'][:100]}...")
        print()
    
    print(f"\n=== 下一步 ===")
    print(f"1. 重新计算 text tokens（加入触发词）")
    print(f"2. 运行 canary 测试训练")
    print(f"3. 验证 LoRA 是否能产生可感知差异")


if __name__ == "__main__":
    main()
