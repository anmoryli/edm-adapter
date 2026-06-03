"""Avicii style LoRA with preference loss: base-negative contrast."""

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


def build_lora_config():
    return {
        "r": 32,
        "lora_alpha": 128,
        "target_modules": [
            "to_q", "to_k", "to_v", "to_out.0",
            "cross_attn.to_q", "cross_attn.to_k", "cross_attn.to_v",
            "cross_attn.add_q_proj", "cross_attn.add_k_proj", "cross_attn.add_v_proj",
            "lyric_encoder.encoders.0.self_attn.linear_q",
            "lyric_encoder.encoders.0.self_attn.linear_k",
            "lyric_encoder.encoders.0.self_attn.linear_v",
            "lyric_encoder.encoders.0.self_attn.linear_out",
            "lyric_encoder.encoders.0.feed_forward.w_1",
            "lyric_encoder.encoders.0.feed_forward.w_2",
            "t_block.1",
            "speaker_embedder",
            "genre_embedder",
            "projectors.0.0", "projectors.0.2", "projectors.0.4",
            "projectors.1.0", "projectors.1.2", "projectors.1.4",
        ],
        "use_rslora": True,
        "lora_dropout": 0.0,
    }


def build_lightning_module_class():
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from trainer import Pipeline as BasePipeline
    from src.edm_control.dataset import build_control_dataset_class

    class PreferencePipeline(BasePipeline):
        def __init__(self, preference_beta=0.5, **kwargs):
            self.preference_beta = preference_beta
            self.ssl_coeff = 0.0
            kwargs["adapter_name"] = "avicii_style"
            kwargs["train"] = False
            super().__init__(**kwargs)
            self.is_train = True
            self.transformers.train()

        def train_dataloader(self):
            dataset_cls = build_control_dataset_class()
            self.train_dataset = dataset_cls(
                train=True,
                train_dataset_path=self.hparams.dataset_path,
            )
            
            sampler = None
            shuffle = True
            if hasattr(self.train_dataset, "pretrain_ds"):
                weights = []
                for item in self.train_dataset.pretrain_ds:
                    section = item.get("section", "")
                    energy = item.get("energy", "")
                    w = float(item.get("sample_weight") or 1.0)
                    if section == "drop" and energy in ["high", "very_high"]:
                        w *= 3.0
                    elif section in ["build-up", "chorus"]:
                        w *= 2.0
                    elif section == "drop":
                        w *= 1.5
                    elif section in ["intro", "outro"]:
                        w *= 0.5
                    weights.append(w)
                
                if weights:
                    sampler = WeightedRandomSampler(
                        weights=torch.DoubleTensor(weights),
                        num_samples=len(weights),
                        replacement=True,
                    )
                    shuffle = False
            
            original_collate = self.train_dataset.collate_fn
            
            def trigger_collate(batch):
                for item in batch:
                    if "prompts" in item:
                        if isinstance(item["prompts"], list):
                            item["prompts"] = [f"{TRIGGER_WORD}, {p}" for p in item["prompts"]]
                        elif isinstance(item["prompts"], str):
                            item["prompts"] = f"{TRIGGER_WORD}, {item['prompts']}"
                return original_collate(batch)
            
            return DataLoader(
                self.train_dataset,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=0,
                pin_memory=True,
                collate_fn=trigger_collate,
            )

        def configure_optimizers(self):
            import math
            trainable_params = [
                p for _, p in self.transformers.named_parameters() if p.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                [{"params": trainable_params}],
                lr=self.hparams.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999),
            )
            
            max_steps = self.hparams.max_steps
            warmup_steps = self.hparams.warmup_steps
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        def run_step(self, batch, batch_idx):
            """Preference loss: LoRA should be closer to Avicii than base."""
            (
                keys, target_latents, attention_mask,
                encoder_text_hidden_states, text_attention_mask,
                speaker_embds, lyric_token_ids, lyric_mask,
                mert_ssl_hidden_states, mhubert_ssl_hidden_states,
            ) = self.preprocess(batch)

            target_image = target_latents
            device = target_image.device
            dtype = target_image.dtype
            bsz = target_image.shape[0]

            noise = torch.randn_like(target_image, device=device)
            timesteps = self.get_timestep(bsz, device)
            sigmas = self.get_sd3_sigmas(
                timesteps=timesteps, device=device, n_dim=target_image.ndim, dtype=dtype
            )
            noisy_image = sigmas * noise + (1.0 - sigmas) * target_image

            x = noisy_image
            all_ssl = []
            if mert_ssl_hidden_states is not None:
                all_ssl.append(mert_ssl_hidden_states)
            if mhubert_ssl_hidden_states is not None:
                all_ssl.append(mhubert_ssl_hidden_states)

            # LoRA prediction (with trigger word in prompt)
            transformer_output = self.transformers(
                hidden_states=x,
                attention_mask=attention_mask,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
                timestep=timesteps.to(device).to(dtype),
                ssl_hidden_states=all_ssl,
            )
            lora_pred = transformer_output.sample * (-sigmas) + noisy_image

            # Base prediction (disable LoRA)
            with torch.no_grad():
                self.transformers.disable_adapters()
                base_output = self.transformers(
                    hidden_states=x,
                    attention_mask=attention_mask,
                    encoder_text_hidden_states=encoder_text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    speaker_embeds=speaker_embds,
                    lyric_token_idx=lyric_token_ids,
                    lyric_mask=lyric_mask,
                    timestep=timesteps.to(device).to(dtype),
                    ssl_hidden_states=all_ssl,
                )
                base_pred = base_output.sample * (-sigmas) + noisy_image
                self.transformers.enable_adapters()

            # Mask
            mask = (
                attention_mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, target_image.shape[1], target_image.shape[2], -1)
            )

            lora_flat = (lora_pred * mask).reshape(bsz, -1).contiguous()
            base_flat = (base_pred * mask).reshape(bsz, -1).contiguous()
            target_flat = (target_image * mask).reshape(bsz, -1).contiguous()

            # Preference loss
            lora_recon = F.mse_loss(lora_flat, target_flat, reduction="none").mean(1)
            base_recon = F.mse_loss(base_flat, target_flat, reduction="none").mean(1)
            preference_gap = base_recon - lora_recon  # positive = LoRA is better

            loss = lora_recon.mean() - self.preference_beta * preference_gap.mean()
            loss = loss * mask.reshape(bsz, -1).mean(1).mean()

            self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log("train/lora_recon", lora_recon.mean(), on_step=True, on_epoch=False)
            self.log("train/base_recon", base_recon.mean(), on_step=True, on_epoch=False)
            self.log("train/gap", preference_gap.mean(), on_step=True, on_epoch=False)

            return loss

        def on_save_checkpoint(self, checkpoint):
            log_dir = self.logger.log_dir
            step = self.global_step
            bundle_dir = Path(log_dir) / "checkpoints" / f"step={step}_preference"
            adapter_dir = bundle_dir / "adapters" / "avicii_style"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.transformers.save_lora_adapter(str(adapter_dir), adapter_name="avicii_style")
            
            manifest = {
                "format": "preference_lora_v1",
                "adapter_name": "avicii_style",
                "trigger_word": TRIGGER_WORD,
                "lora_config": build_lora_config(),
                "preference_beta": self.preference_beta,
                "global_step": int(step),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (bundle_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            return {}

    return PreferencePipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"))
    parser.add_argument("--logger-dir", default=str(PROJECT_ROOT / "outputs" / "avicii_preference" / "logs"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "avicii_preference"))
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--every-n-train-steps", type=int, default=2000)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--preference-beta", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_config_path = output_dir / "preference_lora.json"
    lora_config_path.write_text(json.dumps(build_lora_config(), indent=2), encoding="utf-8")

    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    lightning_cls = build_lightning_module_class()
    model = lightning_cls(
        preference_beta=args.preference_beta,
        learning_rate=args.learning_rate,
        num_workers=0,
        train=False,
        T=1000,
        weight_decay=0.01,
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
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + "preference",
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
