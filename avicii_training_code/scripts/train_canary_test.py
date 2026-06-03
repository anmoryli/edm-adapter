"""Canary test: Overfit 64 samples with aggressive LoRA settings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

TRIGGER_WORD = "avicii_adapter_style"


def build_canary_lora_config():
    """High-gain LoRA config targeting attention + FFN + conditioning."""
    return {
        "r": 32,
        "lora_alpha": 128,  # alpha/r = 4.0, high gain
        "target_modules": [
            # Self-Attention
            "to_q", "to_k", "to_v", "to_out.0",
            # Cross-Attention
            "cross_attn.to_q", "cross_attn.to_k", "cross_attn.to_v",
            "cross_attn.add_q_proj", "cross_attn.add_k_proj", "cross_attn.add_v_proj",
            # Lyric encoder
            "lyric_encoder.encoders.0.self_attn.linear_q",
            "lyric_encoder.encoders.0.self_attn.linear_k",
            "lyric_encoder.encoders.0.self_attn.linear_v",
            "lyric_encoder.encoders.0.self_attn.linear_out",
            "lyric_encoder.encoders.0.feed_forward.w_1",
            "lyric_encoder.encoders.0.feed_forward.w_2",
            # Conditioning
            "t_block.1",
            "speaker_embedder",
            "genre_embedder",
            # Projectors
            "projectors.0.0", "projectors.0.2", "projectors.0.4",
            "projectors.1.0", "projectors.1.2", "projectors.1.4",
        ],
        "use_rslora": True,
        "lora_dropout": 0.0,
    }


def build_lightning_module_class():
    import torch
    import math
    from torch.utils.data import DataLoader, Subset
    from trainer import Pipeline as BasePipeline
    from src.edm_control.dataset import build_control_dataset_class

    class CanaryPipeline(BasePipeline):
        def __init__(self, canary_indices: list[int], **kwargs):
            self.canary_indices = canary_indices
            self.ssl_coeff = 0.0
            kwargs["adapter_name"] = "avicii_canary"
            kwargs["train"] = False
            super().__init__(**kwargs)
            self.is_train = True
            self.transformers.train()

        def train_dataloader(self):
            dataset_cls = build_control_dataset_class()
            full_dataset = dataset_cls(
                train=True,
                train_dataset_path=self.hparams.dataset_path,
            )
            
            # Use only canary indices
            subset = Subset(full_dataset, self.canary_indices)
            
            # Modify captions to include trigger word
            original_collate = full_dataset.collate_fn
            
            def trigger_collate(batch):
                for item in batch:
                    if "prompts" in item:
                        if isinstance(item["prompts"], list):
                            item["prompts"] = [f"{TRIGGER_WORD}, {p}" for p in item["prompts"]]
                        elif isinstance(item["prompts"], str):
                            item["prompts"] = f"{TRIGGER_WORD}, {item['prompts']}"
                return original_collate(batch)
            
            return DataLoader(
                subset,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=trigger_collate,
            )

        def configure_optimizers(self):
            trainable_params = [
                p for _, p in self.transformers.named_parameters() if p.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                [{"params": trainable_params}],
                lr=self.hparams.learning_rate,
                weight_decay=0.0,
                betas=(0.9, 0.999),
            )
            return [optimizer], []

        def on_save_checkpoint(self, checkpoint):
            log_dir = self.logger.log_dir
            step = self.global_step
            bundle_dir = Path(log_dir) / "checkpoints" / f"step={step}_canary"
            adapter_dir = bundle_dir / "adapters" / "avicii_canary"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.transformers.save_lora_adapter(str(adapter_dir), adapter_name="avicii_canary")
            
            manifest = {
                "format": "canary_lora_v1",
                "adapter_name": "avicii_canary",
                "trigger_word": TRIGGER_WORD,
                "lora_config": build_canary_lora_config(),
                "global_step": int(step),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (bundle_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            return {}

    return CanaryPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"))
    parser.add_argument("--logger-dir", default=str(PROJECT_ROOT / "outputs" / "canary_test" / "logs"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "canary_test"))
    parser.add_argument("--canary-config", default=str(PROJECT_ROOT / "outputs" / "avicii_trigger_dataset" / "canary_config.json"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--every-n-train-steps", type=int, default=200)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    args = parser.parse_args()

    # Load canary config
    with open(args.canary_config) as f:
        canary_config = json.load(f)
    canary_indices = canary_config["indices"]
    print(f"Canary 样本数: {len(canary_indices)}")

    # Write LoRA config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_config_path = output_dir / "canary_lora.json"
    lora_config_path.write_text(
        json.dumps(build_canary_lora_config(), indent=2), encoding="utf-8"
    )

    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    lightning_cls = build_lightning_module_class()
    model = lightning_cls(
        canary_indices=canary_indices,
        learning_rate=args.learning_rate,
        num_workers=0,
        train=False,
        T=1000,
        weight_decay=0.0,
        every_plot_step=500,
        shift=3.0,
        ssl_coeff=0.0,
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        warmup_steps=10,
        dataset_path=args.dataset_path,
        lora_config_path=str(lora_config_path),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=None,
        every_n_train_steps=args.every_n_train_steps,
        save_top_k=-1,
    )
    logger_callback = TensorBoardLogger(
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + "canary",
        save_dir=args.logger_dir,
    )
    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="ddp_find_unused_parameters_true",
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    trainer.fit(model)


if __name__ == "__main__":
    main()
