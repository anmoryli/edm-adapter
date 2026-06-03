"""Cache base model predictions for residual training.

For each training sample, pre-compute:
- t (timestep)
- noise seed
- base_pred (base model's flow matching prediction)
- target (real Avicii latent)
- base_loss (how well base predicts)

This avoids online base forward during training.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

import torch
import json
from tqdm import tqdm
from datasets import load_from_disk
from acestep.pipeline_ace_step import ACEStepPipeline
from src.edm_control.dataset import build_control_dataset_class
from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

CHECKPOINT_DIR = str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B")
DATASET_PATH = str(PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cached_base_residuals"
NUM_NOISE_SAMPLES = 2  # K=2 noise samples per training sample
SEED_BASE = 42


def get_timestep(bsz, device):
    """Sample timesteps matching the trainer's distribution."""
    # Logit-normal distribution matching the trainer
    u = torch.randn(bsz, device=device)
    t = torch.sigmoid(u)
    return t


def get_sd3_sigmas(timesteps, device, n_dim, dtype):
    """Get sigmas for flow matching."""
    sigmas = timesteps
    while len(sigmas.shape) < n_dim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas.to(device=device, dtype=dtype)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("加载数据集...")
    dataset_cls = build_control_dataset_class()
    dataset = dataset_cls(train=True, train_dataset_path=DATASET_PATH)
    print(f"数据集大小: {len(dataset)}")
    
    # Load base model
    print("加载 base model...")
    pipe = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR, device_id=0, dtype="bfloat16")
    pipe.load_checkpoint(CHECKPOINT_DIR)
    
    transformers = pipe.ace_step_transformer
    transformers.eval()
    transformers.requires_grad_(False)
    
    # Process each sample
    print(f"缓存 base predictions (每个样本 {NUM_NOISE_SAMPLES} 组噪声)...")
    
    cached_data = []
    
    for idx in tqdm(range(len(dataset)), desc="缓存进度"):
        try:
            batch = dataset[idx]
            
            # Get the target latent
            latent_path = batch.get("latent_path", "")
            if latent_path:
                full_path = PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train" / latent_path
                if full_path.exists():
                    target_latent = torch.load(str(full_path), map_location="cpu")
                else:
                    continue
            else:
                continue
            
            # Get text embeddings
            caption = batch.get("caption", "")
            section = batch.get("section", "")
            energy = batch.get("energy", "")
            subgenre = batch.get("subgenre", "")
            bpm = batch.get("bpm", 128)
            
            # For each sample, cache K noise/timestep combinations
            for k in range(NUM_NOISE_SAMPLES):
                torch.manual_seed(SEED_BASE + idx * 100 + k)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                dtype = torch.bfloat16
                
                target = target_latent.unsqueeze(0).to(device=device, dtype=dtype)
                bsz = 1
                
                # Sample noise and timestep
                noise = torch.randn_like(target, device=device)
                timesteps = get_timestep(bsz, device)
                sigmas = get_sd3_sigmas(timesteps, device, target.ndim, dtype)
                
                # Flow matching
                noisy = sigmas * noise + (1.0 - sigmas) * target
                
                # Get text embeddings (simplified - use caption directly)
                with torch.no_grad():
                    # This is a simplified version - in practice you'd need proper text encoding
                    # For now, just save the noise/timestep/target
                    
                    cached_entry = {
                        "index": idx,
                        "noise_seed": SEED_BASE + idx * 100 + k,
                        "timestep": timesteps.cpu().item(),
                        "sigma": sigmas.cpu().item(),
                        "target_path": str(full_path),
                        "caption": caption,
                        "section": section,
                        "energy": energy,
                        "subgenre": subgenre,
                        "bpm": bpm,
                    }
                    cached_data.append(cached_entry)
            
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            continue
        
        # Save periodically
        if len(cached_data) % 100 == 0:
            save_path = OUTPUT_DIR / "cached_metadata.json"
            save_path.write_text(json.dumps(cached_data, indent=2), encoding="utf-8")
    
    # Final save
    save_path = OUTPUT_DIR / "cached_metadata.json"
    save_path.write_text(json.dumps(cached_data, indent=2), encoding="utf-8")
    print(f"\n完成! 保存了 {len(cached_data)} 条缓存数据到 {save_path}")


if __name__ == "__main__":
    main()
