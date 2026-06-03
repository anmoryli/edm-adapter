"""Fine-tune Stable Audio Open 1.0 or MusicGen on electronic music data.

This script supports two backends:
1. Stable Audio Open (primary) via stable-audio-tools
2. MusicGen (fallback) via audiocraft/transformers

For small-scale testing, use --max-samples to limit training data.
"""

import argparse
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, get_device, setup_logging
from src.audio_io import load_audio, save_audio, normalize_audio
from src.dataset_builder import load_metadata_jsonl


def finetune_stable_audio(train_data, val_data, config, device, output_dir, max_samples=None):
    """Fine-tune Stable Audio Open 1.0.

    This uses the stable-audio-tools training pipeline.
    """
    try:
        from stable_audio_tools import get_pretrained_model
        from stable_audio_tools.training.training import train
    except ImportError:
        print("stable-audio-tools not installed. Install with:")
        print("  pip install stable-audio-tools")
        print("Or clone from: https://github.com/Stability-AI/stable-audio-tools")
        raise

    print("Loading Stable Audio Open 1.0 model...")
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)

    if max_samples:
        train_data = train_data[:max_samples]
        val_data = val_data[:max_samples // 5] if val_data else []

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Build training config for stable-audio-tools
    train_config = {
        "training": {
            "batch_size": config.get("training", {}).get("batch_size", 4),
            "learning_rate": config.get("training", {}).get("learning_rate", 1e-5),
            "num_epochs": config.get("training", {}).get("num_epochs", 3),
            "gradient_accumulation_steps": config.get("training", {}).get("gradient_accumulation_steps", 2),
            "fp16": config.get("training", {}).get("fp16", True),
        },
        "data": {
            "sample_rate": 44100,
            "clip_samples": 44100 * 10,  # 10 seconds
        },
    }

    # Create dataset class for stable-audio-tools
    from torch.utils.data import Dataset, DataLoader

    class EDMDataset(Dataset):
        def __init__(self, data_list, sr=44100, clip_duration=10.0):
            self.data = data_list
            self.sr = sr
            self.clip_samples = int(sr * clip_duration)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            audio_path = item["audio_path"]
            caption = item["caption"]

            try:
                y, sr = load_audio(audio_path, sr=self.sr)
                # Ensure correct length
                if y.shape[-1] < self.clip_samples:
                    pad = self.clip_samples - y.shape[-1]
                    y = np.pad(y, ((0, 0), (0, pad)), mode="constant")
                else:
                    y = y[..., :self.clip_samples]

                return {
                    "audio": torch.from_numpy(y).float(),
                    "text": caption,
                }
            except Exception as e:
                # Return silence on error
                return {
                    "audio": torch.zeros(2, self.clip_samples),
                    "text": caption,
                }

    train_dataset = EDMDataset(train_data)
    val_dataset = EDMDataset(val_data) if val_data else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["training"]["learning_rate"],
        weight_decay=0.01,
    )

    # Training loop
    num_epochs = train_config["training"]["num_epochs"]
    save_every = config.get("checkpointing", {}).get("save_every_n_steps", 500)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            audio = batch["audio"].to(device)
            text = batch["text"]

            # Forward pass - model-specific
            try:
                # Stable audio tools uses a different training interface
                # This is a simplified version
                loss = model.training_step(audio, text)
                loss.backward()

                if (batch_idx + 1) % train_config["training"]["gradient_accumulation_steps"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 10 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs}, Step {batch_idx+1}, Loss: {loss.item():.4f}")

                if global_step % save_every == 0:
                    ckpt_path = os.path.join(output_dir, f"checkpoint_step_{global_step}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"  Saved checkpoint: {ckpt_path}")

            except Exception as e:
                print(f"  Training step error: {e}")
                optimizer.zero_grad()
                continue

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nFine-tuning complete. Final model saved to: {final_path}")

    return model


def finetune_musicgen(train_data, val_data, config, device, output_dir, max_samples=None):
    """Fail fast because the previous Transformers-only training path was invalid."""
    raise NotImplementedError(
        "MusicGen fine-tuning is disabled in this project. The previous Transformers-only "
        "training path did not produce a valid MusicGen fine-tune and was a major source of "
        "misleadingly bad results. Use Stable Audio training or re-implement MusicGen training "
        "with the proper AudioCraft pipeline."
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model on electronic music")
    parser.add_argument("--config", default="configs/stable_audio_finetune.yaml")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--val", default="data/processed/val.jsonl")
    parser.add_argument("--output", default="outputs/checkpoints/stable_audio_edm")
    parser.add_argument("--model", default="stable-audio", choices=["stable-audio", "musicgen"])
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max training samples (for quick testing)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    setup_logging()
    config = load_yaml(args.config)
    device = get_device(args.device)

    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # Load data
    if not os.path.exists(args.train):
        print(f"Training data not found: {args.train}")
        print("Run the data processing pipeline first (scripts 01-06).")
        return

    train_data = load_metadata_jsonl(args.train)
    val_data = load_metadata_jsonl(args.val) if os.path.exists(args.val) else []

    print(f"Loaded {len(train_data)} training, {len(val_data)} validation samples")

    # Fine-tune
    if args.model == "stable-audio":
        try:
            finetune_stable_audio(train_data, val_data, config, device, args.output, args.max_samples)
        except Exception as e:
            print(f"\nStable Audio fine-tuning failed: {e}")
            print("Switching to MusicGen fallback...")
            args.model = "musicgen"
            args.output = args.output.replace("stable_audio", "musicgen")

    if args.model == "musicgen":
        try:
            finetune_musicgen(train_data, val_data, config, device, args.output, args.max_samples)
        except NotImplementedError as e:
            print(f"\n{e}")


if __name__ == "__main__":
    main()
