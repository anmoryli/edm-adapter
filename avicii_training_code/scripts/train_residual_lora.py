"""Avicii style LoRA with cached base residual training.

Core idea: LoRA learns (Avicii - base) residual, not Avicii itself.
This forces the model to learn what makes Avicii different from base.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ACE-Step").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Cannot find project root containing ACE-Step/ and src/")


PROJECT_ROOT = find_project_root()
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACE_STEP_ROOT))

TRIGGER_WORD = "avicii_adapter_style"


def build_lora_config():
    return {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": [
            "to_q", "to_k", "to_v", "to_out.0",
            "cross_attn.to_q", "cross_attn.to_k", "cross_attn.to_v",
            "cross_attn.add_q_proj", "cross_attn.add_k_proj", "cross_attn.add_v_proj",
            "timestep_embedder.linear_1", "timestep_embedder.linear_2",
            "lyric_encoder.encoders.0.self_attn.linear_q",
            "lyric_encoder.encoders.0.self_attn.linear_k",
            "lyric_encoder.encoders.0.self_attn.linear_v",
            "lyric_encoder.encoders.0.self_attn.linear_out",
            "lyric_encoder.encoders.0.feed_forward.w_1",
            "lyric_encoder.encoders.0.feed_forward.w_2",
            "lyric_proj",
            "t_block.1",
            "speaker_embedder",
            "genre_embedder",
            "projectors.0.0", "projectors.0.2", "projectors.0.4",
            "projectors.1.0", "projectors.1.2", "projectors.1.4",
            "final_layer.linear",
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

    class ResidualPipeline(BasePipeline):
        def __init__(
            self,
            residual_weight=0.45,
            direction_weight=0.25,
            advantage_weight=0.15,
            delta_norm_weight=0.05,
            style_gain=1.15,
            residual_margin=0.02,
            residual_temperature=0.05,
            residual_every_n_steps=4,
            **kwargs,
        ):
            self.residual_weight = residual_weight
            self.direction_weight = direction_weight
            self.advantage_weight = advantage_weight
            self.delta_norm_weight = delta_norm_weight
            self.style_gain = style_gain
            self.residual_margin = residual_margin
            self.residual_temperature = residual_temperature
            self.residual_every_n_steps = max(1, int(residual_every_n_steps))
            self.ssl_coeff = 0.0
            kwargs["adapter_name"] = "avicii_style"
            kwargs["train"] = False
            super().__init__(**kwargs)
            self.is_train = True
            self.transformers.train()
            self._print_lora_summary()

        def _print_lora_summary(self):
            trainable = [(name, p) for name, p in self.transformers.named_parameters() if p.requires_grad]
            lora_params = [(name, p) for name, p in trainable if "lora_" in name]
            if not lora_params:
                raise RuntimeError("No trainable LoRA parameters were created. Check target_modules.")
            total = sum(p.numel() for _, p in trainable)
            lora_total = sum(p.numel() for _, p in lora_params)
            print(f"Trainable params: {total:,}; LoRA params: {lora_total:,}; tensors: {len(lora_params)}", flush=True)

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
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                collate_fn=trigger_collate,
            )

        def preprocess(self, batch, train=True):
            """Use cached latents when available.

            The earlier Avicii scripts inherited ACE-Step's default preprocess,
            which re-encoded audio every step and ignored cached latents. That is
            slower and makes debugging harder, but the conditioning path remains
            the same as ACE-Step training.
            """
            if "target_latents" in batch:
                target_latents = batch["target_latents"]
                device = target_latents.device
                dtype = target_latents.dtype
                bs = target_latents.shape[0]
                target_wavs = batch.get("target_wavs")
                wav_lengths = batch.get("wav_lengths")
            else:
                return super().preprocess(batch, train=train)

            mert_ssl_hidden_states = None
            mhubert_ssl_hidden_states = None
            if train and self.is_train and target_wavs is not None and wav_lengths is not None and hasattr(self, "mert_model"):
                dev_type = "cuda" if target_wavs.is_cuda else "cpu"
                with torch.amp.autocast(device_type=dev_type, dtype=dtype):
                    mert_ssl_hidden_states = self.infer_mert_ssl(target_wavs, wav_lengths)
                    mhubert_ssl_hidden_states = self.infer_mhubert_ssl(target_wavs, wav_lengths)

            texts = batch["prompts"]
            encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(texts, device)
            encoder_text_hidden_states = encoder_text_hidden_states.to(dtype)

            target_latents = target_latents.to(device=device, dtype=dtype)
            latent_lengths = batch.get("latent_lengths")
            if latent_lengths is None:
                attention_mask = torch.ones(bs, target_latents.shape[-1], device=device, dtype=dtype)
            else:
                positions = torch.arange(target_latents.shape[-1], device=device).unsqueeze(0)
                attention_mask = (positions < latent_lengths.to(device).unsqueeze(1)).to(dtype)

            speaker_embds = batch["speaker_embs"].to(device=device, dtype=dtype)
            keys = batch["keys"]
            lyric_token_ids = batch["lyric_token_ids"].to(device)
            lyric_mask = batch["lyric_masks"].to(device)

            if train:
                keep_text = (torch.rand(size=(bs,), device=device) >= 0.05).long()
                encoder_text_hidden_states = torch.where(
                    keep_text.unsqueeze(1).unsqueeze(1).bool(),
                    encoder_text_hidden_states,
                    torch.zeros_like(encoder_text_hidden_states),
                )

                keep_speaker = (torch.rand(size=(bs,), device=device) >= 0.35).long()
                speaker_embds = torch.where(
                    keep_speaker.unsqueeze(1).bool(),
                    speaker_embds,
                    torch.zeros_like(speaker_embds),
                )

                keep_lyrics = (torch.rand(size=(bs,), device=device) >= 0.15).long()
                lyric_token_ids = torch.where(
                    keep_lyrics.unsqueeze(1).bool(),
                    lyric_token_ids,
                    torch.zeros_like(lyric_token_ids),
                )
                lyric_mask = torch.where(
                    keep_lyrics.unsqueeze(1).bool(),
                    lyric_mask,
                    torch.zeros_like(lyric_mask),
                )

            return (
                keys,
                target_latents,
                attention_mask,
                encoder_text_hidden_states,
                text_attention_mask,
                speaker_embds,
                lyric_token_ids,
                lyric_mask,
                mert_ssl_hidden_states,
                mhubert_ssl_hidden_states,
            )

        def configure_optimizers(self):
            import math
            trainable_params = [
                p for _, p in self.transformers.named_parameters() if p.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                [{"params": trainable_params}],
                lr=self.hparams.learning_rate,
                weight_decay=0.0,
                betas=(0.8, 0.9),
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

        @staticmethod
        def masked_mse(pred, target, mask):
            squared = (pred.float() - target.float()).pow(2) * mask.float()
            denom = mask.float().reshape(mask.shape[0], -1).sum(1).clamp_min(1.0)
            return squared.reshape(squared.shape[0], -1).sum(1) / denom

        def run_step(self, batch, batch_idx):
            """Train LoRA to beat the base prediction and move along the base->Avicii residual.

            The previous implementation used:
                mse(lora_pred - base_pred, target - base_pred)
            That algebraically equals mse(lora_pred, target), so it was just ordinary
            denoising loss with an extra base forward. This version uses non-degenerate
            direction, advantage, and amplified residual terms.
            """
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

            # LoRA prediction
            lora_output = self.transformers(
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
            lora_pred = lora_output.sample * (-sigmas) + noisy_image

            # Mask
            mask = (
                attention_mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, target_image.shape[1], target_image.shape[2], -1)
            )

            recon_loss = self.masked_mse(lora_pred, target_image, mask)
            loss = recon_loss.mean()
            use_residual = (int(self.global_step) % self.residual_every_n_steps) == 0

            if use_residual:
                # Base prediction (disable LoRA). Keep the exact same noisy latent and
                # conditioning, including the trigger token, so the delta is only LoRA.
                try:
                    self.transformers.disable_adapters()
                    with torch.no_grad():
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
                finally:
                    self.transformers.enable_adapters()

                base_pred = base_pred.detach()
                base_recon = self.masked_mse(base_pred, target_image, mask)

                target_delta = (target_image.float() - base_pred.float()) * mask.float()
                lora_delta = (lora_pred.float() - base_pred.float()) * mask.float()
                amplified_target = (base_pred.float() + self.style_gain * target_delta).detach()

                residual_loss = self.masked_mse(lora_pred.float(), amplified_target, mask)

                target_flat = target_delta.reshape(bsz, -1)
                lora_flat = lora_delta.reshape(bsz, -1)
                direction_loss = 1.0 - F.cosine_similarity(lora_flat, target_flat, dim=1, eps=1e-6)

                lora_delta_norm = lora_flat.norm(dim=1)
                target_delta_norm = target_flat.norm(dim=1).clamp_min(1e-6)
                delta_ratio = lora_delta_norm / target_delta_norm
                delta_norm_loss = F.smooth_l1_loss(
                    delta_ratio,
                    torch.full_like(delta_ratio, float(self.style_gain)),
                    reduction="mean",
                )

                temperature = max(float(self.residual_temperature), 1e-4)
                advantage_loss = (
                    F.softplus((recon_loss - base_recon.detach() + self.residual_margin) / temperature)
                    * temperature
                ).mean()

                sigma_weight = sigmas.reshape(bsz, -1)[:, 0].float().clamp(0.15, 0.85)
                sigma_weight = ((sigma_weight - 0.15) / 0.70).clamp(0.25, 1.0)
                energy_weight = (
                    target_delta_norm.detach() / target_delta_norm.detach().mean().clamp_min(1e-6)
                ).clamp(0.25, 2.0)
                residual_weight = sigma_weight * energy_weight

                residual_term = (residual_loss * residual_weight).mean()
                direction_term = (direction_loss * residual_weight).mean()
                loss = (
                    (1.0 - self.residual_weight) * recon_loss.mean()
                    + self.residual_weight * residual_term
                    + self.direction_weight * direction_term
                    + self.advantage_weight * advantage_loss
                    + self.delta_norm_weight * delta_norm_loss
                )

                self.log("train/base_recon", base_recon.mean(), on_step=True, on_epoch=False)
                self.log("train/residual", residual_term, on_step=True, on_epoch=False)
                self.log("train/direction", direction_term, on_step=True, on_epoch=False)
                self.log("train/advantage", advantage_loss, on_step=True, on_epoch=False)
                self.log("train/delta_ratio", delta_ratio.mean(), on_step=True, on_epoch=False)
                self.log("train/residual_energy", target_delta_norm.mean(), on_step=True, on_epoch=False)

            if not torch.isfinite(loss):
                self.log("train/nonfinite_loss", torch.ones((), device=device), on_step=True, on_epoch=False)
                loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=1.0)

            self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log("train/recon", recon_loss.mean(), on_step=True, on_epoch=False)
            self.log("train/used_residual", float(use_residual), on_step=True, on_epoch=False)

            return loss

        def on_save_checkpoint(self, checkpoint):
            log_dir = self.logger.log_dir
            step = self.global_step
            bundle_dir = Path(log_dir) / "checkpoints" / f"step={step}_residual"
            adapter_dir = bundle_dir / "adapters" / "avicii_style"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.transformers.save_lora_adapter(str(adapter_dir), adapter_name="avicii_style")
            
            manifest = {
                "format": "residual_lora_v1",
                "adapter_name": "avicii_style",
                "trigger_word": TRIGGER_WORD,
                "lora_config": build_lora_config(),
                "residual_weight": self.residual_weight,
                "direction_weight": self.direction_weight,
                "advantage_weight": self.advantage_weight,
                "delta_norm_weight": self.delta_norm_weight,
                "style_gain": self.style_gain,
                "residual_margin": self.residual_margin,
                "residual_temperature": self.residual_temperature,
                "residual_every_n_steps": self.residual_every_n_steps,
                "global_step": int(step),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (bundle_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            return {}

    return ResidualPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=str(PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"))
    parser.add_argument("--logger-dir", default=str(PROJECT_ROOT / "outputs" / "avicii_residual" / "logs"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "avicii_residual"))
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--every-n-train-steps", type=int, default=2000)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--accumulate-grad-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-clip-val", type=float, default=0.3)
    parser.add_argument("--residual-weight", type=float, default=0.45)
    parser.add_argument("--direction-weight", type=float, default=0.25)
    parser.add_argument("--advantage-weight", type=float, default=0.15)
    parser.add_argument("--delta-norm-weight", type=float, default=0.05)
    parser.add_argument("--style-gain", type=float, default=1.15)
    parser.add_argument("--residual-margin", type=float, default=0.02)
    parser.add_argument("--residual-temperature", type=float, default=0.05)
    parser.add_argument("--residual-every-n-steps", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_config_path = output_dir / "residual_lora.json"
    lora_config_path.write_text(json.dumps(build_lora_config(), indent=2), encoding="utf-8")

    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    lightning_cls = build_lightning_module_class()
    model = lightning_cls(
        residual_weight=args.residual_weight,
        direction_weight=args.direction_weight,
        advantage_weight=args.advantage_weight,
        delta_norm_weight=args.delta_norm_weight,
        style_gain=args.style_gain,
        residual_margin=args.residual_margin,
        residual_temperature=args.residual_temperature,
        residual_every_n_steps=args.residual_every_n_steps,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        train=False,
        T=1000,
        weight_decay=0.0,
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
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + "residual",
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
