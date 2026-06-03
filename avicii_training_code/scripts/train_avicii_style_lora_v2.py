"""Train Avicii Style LoRA v2: improved parameters for stronger style transfer."""

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


def build_style_lora_config():
    """v2: Lower rank, attention layers only for style."""
    return {
        "r": 8,
        "lora_alpha": 8,
        "target_modules": [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
        "use_rslora": True,
        "lora_dropout": 0.05,
    }


def build_lightning_module_class():
    import torch
    import math
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from trainer import Pipeline as BasePipeline
    from src.edm_control.dataset import build_control_dataset_class

    class AviciiStylePipeline(BasePipeline):
        def __init__(
            self,
            output_dir: str,
            sample_size: int | None = None,
            use_cached_latents: bool = True,
            weighted_sampling: bool = True,
            **kwargs,
        ):
            self.output_dir = output_dir
            self.sample_size = sample_size
            self.use_cached_latents = use_cached_latents
            self.weighted_sampling = weighted_sampling
            self.ssl_coeff = 0.0

            self.adapter_name = "avicii_style"
            kwargs["adapter_name"] = self.adapter_name
            kwargs["train"] = False

            super().__init__(**kwargs)

            self.is_train = True
            self.transformers.train()

        def train_dataloader(self):
            dataset_cls = build_control_dataset_class()
            self.train_dataset = dataset_cls(
                train=True,
                train_dataset_path=self.hparams.dataset_path,
                sample_size=self.sample_size,
            )
            sampler = None
            shuffle = True
            if self.weighted_sampling and hasattr(self.train_dataset, "pretrain_ds"):
                weights = [
                    float(item.get("sample_weight") or 1.0)
                    for item in self.train_dataset.pretrain_ds
                ]
                if weights:
                    sampler = WeightedRandomSampler(
                        weights=torch.DoubleTensor(weights),
                        num_samples=len(weights),
                        replacement=True,
                    )
                    shuffle = False
            return DataLoader(
                self.train_dataset,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                collate_fn=self.train_dataset.collate_fn,
            )

        def configure_optimizers(self):
            trainable_params = [
                parameter
                for _, parameter in self.transformers.named_parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                [{"params": trainable_params}],
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999),
            )

            # Fixed cosine schedule with proper max_steps
            max_steps = self.hparams.max_steps
            warmup_steps = self.hparams.warmup_steps

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        def on_save_checkpoint(self, checkpoint):
            log_dir = self.logger.log_dir
            epoch = self.current_epoch
            step = self.global_step
            bundle_dir = Path(log_dir) / "checkpoints" / f"epoch={epoch}-step={step}_avicii_style_v2"
            adapters_dir = bundle_dir / "adapters"
            adapters_dir.mkdir(parents=True, exist_ok=True)

            adapter_dir = adapters_dir / self.adapter_name
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.transformers.save_lora_adapter(str(adapter_dir), adapter_name=self.adapter_name)

            manifest = {
                "format": "avicii_style_lora_v2",
                "adapter_name": self.adapter_name,
                "lora_config": build_style_lora_config(),
                "global_step": int(step),
                "epoch": int(epoch),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (bundle_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return {}

    return AviciiStylePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"))
    parser.add_argument("--logger-dir", default=str(PROJECT_ROOT / "outputs" / "avicii_style_lora_v2" / "logs"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "avicii_style_lora_v2"))
    parser.add_argument("--exp-name", default="avicii_style_v2")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=4832)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--every-n-train-steps", type=int, default=500)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--disable-cached-latents", action="store_true")
    parser.add_argument("--disable-weighted-sampling", action="store_true")
    args = parser.parse_args()

    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = output_dir / "runtime_config"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lora_config_path = runtime_dir / "style_lora_v2.json"
    lora_config_path.write_text(
        json.dumps(build_style_lora_config(), indent=2), encoding="utf-8"
    )

    lightning_cls = build_lightning_module_class()
    model = lightning_cls(
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        use_cached_latents=not args.disable_cached_latents,
        weighted_sampling=not args.disable_weighted_sampling,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        train=False,
        T=1000,
        weight_decay=args.weight_decay,
        every_plot_step=2000,
        shift=3.0,
        ssl_coeff=0.0,
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        dataset_path=args.dataset_path,
        lora_config_path=str(lora_config_path),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=None,
        every_n_train_steps=args.every_n_train_steps,
        save_top_k=-1,
    )
    logger_callback = TensorBoardLogger(
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + args.exp_name,
        save_dir=args.logger_dir,
    )
    accelerator = "cpu" if args.use_cpu or not torch.cuda.is_available() else "gpu"
    precision = "32" if accelerator == "cpu" else args.precision
    trainer = Trainer(
        accelerator=accelerator,
        devices=1 if accelerator == "cpu" else args.devices,
        num_nodes=args.num_nodes,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="auto" if accelerator == "cpu" else "ddp_find_unused_parameters_true",
        max_epochs=args.max_epochs if args.max_epochs > 0 else -1,
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        reload_dataloaders_every_n_epochs=1,
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    trainer.fit(model)


if __name__ == "__main__":
    main()
