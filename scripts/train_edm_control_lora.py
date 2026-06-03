"""Train EDM-StructLoRA: section/attribute routed LoRA for ACE-Step."""

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

from src.edm_control.lora_router import EDMAdapterRouter, RouterConfig, save_router_manifest  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_runtime_lora_config(output_dir: Path, name: str, config: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_lightning_module_class():
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from peft import LoraConfig
    from trainer import Pipeline as BasePipeline

    from src.edm_control.control_conditioner import EDMControlConditioner
    from src.edm_control.dataset import build_control_dataset_class

    class EDMControlPipeline(BasePipeline):
        def __init__(
            self,
            method_config: dict,
            output_dir: str,
            sample_size: int | None = None,
            use_cached_latents: bool = True,
            weighted_sampling: bool = True,
            enable_ssl_loss: bool = False,
            **kwargs,
        ):
            self.method_config = method_config
            self.output_dir = output_dir
            self.sample_size = sample_size
            self.use_cached_latents = use_cached_latents
            self.weighted_sampling = weighted_sampling
            self.enable_ssl_loss = enable_ssl_loss
            self.router = EDMAdapterRouter(RouterConfig.from_dict(method_config.get("router")))
            runtime_dir = Path(output_dir) / "runtime_config"
            shared_config_path = write_runtime_lora_config(
                runtime_dir,
                "shared_lora",
                method_config["shared_lora"],
            )
            kwargs["lora_config_path"] = str(shared_config_path)
            kwargs["adapter_name"] = self.router.config.shared_adapter
            kwargs["train"] = bool(enable_ssl_loss)
            super().__init__(**kwargs)

            self.is_train = True
            self.transformers.train()
            self.ssl_coeff = float(kwargs.get("ssl_coeff", 0.0)) if enable_ssl_loss else 0.0
            self.all_adapter_names = [self.router.config.shared_adapter]
            self._add_expert_adapters(method_config)

            conditioner_cfg = method_config.get("control_conditioner", {})
            self.use_control_conditioner = bool(conditioner_cfg.get("enabled", True))
            self.control_conditioner = None
            if self.use_control_conditioner:
                self.control_conditioner = EDMControlConditioner(
                    feature_dim=int(conditioner_cfg["feature_dim"]),
                    text_embed_dim=int(conditioner_cfg.get("text_embed_dim", 768)),
                    token_count=int(conditioner_cfg.get("token_count", 8)),
                    hidden_dim=int(conditioner_cfg.get("hidden_dim", 512)),
                    dropout=float(conditioner_cfg.get("dropout", 0.05)),
                )

        def _add_expert_adapters(self, method_config: dict) -> None:
            expert_config = LoraConfig(**method_config["expert_lora"])
            existing = set(self.all_adapter_names)
            for spec in self.router.adapter_specs():
                name = spec["name"]
                if name in existing:
                    continue
                self.transformers.add_adapter(adapter_config=expert_config, adapter_name=name)
                self.all_adapter_names.append(name)
                existing.add(name)

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
            import math

            trainable_params = [
                parameter
                for _, parameter in self.transformers.named_parameters()
                if parameter.requires_grad
            ]
            if self.control_conditioner is not None:
                trainable_params.extend(
                    parameter
                    for parameter in self.control_conditioner.parameters()
                    if parameter.requires_grad
                )
            optimizer = torch.optim.AdamW(
                [{"params": trainable_params}],
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.8, 0.9),
            )

            def lr_lambda(current_step):
                warmup_steps = self.hparams.warmup_steps
                max_steps = self.hparams.max_steps
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        def _apply_route(self, batch) -> dict[str, float]:
            rows = batch.get("route_metadatas", [])
            weights = self.router.weights_for_batch(rows)
            self.router.apply_to_model(self.transformers, weights)
            for name, value in weights.items():
                if name == self.router.config.shared_adapter or name.startswith("section_"):
                    self.log(f"router/{name}", value, on_step=True, on_epoch=False, prog_bar=False)
            return weights

        def preprocess(self, batch, train=True):
            if self.use_cached_latents and "target_latents" in batch:
                target_latents = batch["target_latents"]
                device = target_latents.device
                dtype = target_latents.dtype
                bs = target_latents.shape[0]
                target_wavs = batch.get("target_wavs")
                wav_lengths = batch.get("wav_lengths")
            else:
                target_wavs = batch["target_wavs"]
                wav_lengths = batch["wav_lengths"]
                device = target_wavs.device
                dtype = target_wavs.dtype
                bs = target_wavs.shape[0]
                target_latents = None

            mert_ssl_hidden_states = None
            mhubert_ssl_hidden_states = None
            if (
                train
                and self.enable_ssl_loss
                and target_wavs is not None
                and wav_lengths is not None
                and hasattr(self, "mert_model")
            ):
                dev_type = "cuda" if target_wavs.is_cuda else "cpu"
                with torch.amp.autocast(device_type=dev_type, dtype=dtype):
                    mert_ssl_hidden_states = self.infer_mert_ssl(target_wavs, wav_lengths)
                    mhubert_ssl_hidden_states = self.infer_mhubert_ssl(target_wavs, wav_lengths)

            texts = batch["prompts"]
            encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(texts, device)
            encoder_text_hidden_states = encoder_text_hidden_states.to(dtype)

            if self.control_conditioner is not None and "control_curves" in batch:
                controls = batch["control_curves"].to(device=device, dtype=torch.float32)
                control_tokens = self.control_conditioner(controls).to(device=device, dtype=dtype)
                encoder_text_hidden_states = torch.cat([encoder_text_hidden_states, control_tokens], dim=1)
                control_mask = torch.ones(
                    bs,
                    control_tokens.shape[1],
                    device=device,
                    dtype=text_attention_mask.dtype,
                )
                text_attention_mask = torch.cat([text_attention_mask, control_mask], dim=1)

            if target_latents is None:
                target_latents, _ = self.dcae.encode(target_wavs, wav_lengths)
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
                full_cfg_condition_mask = torch.where(
                    torch.rand(size=(bs,), device=device) < 0.15,
                    torch.zeros(size=(bs,), device=device),
                    torch.ones(size=(bs,), device=device),
                ).long()
                encoder_text_hidden_states = torch.where(
                    full_cfg_condition_mask.unsqueeze(1).unsqueeze(1).bool(),
                    encoder_text_hidden_states,
                    torch.zeros_like(encoder_text_hidden_states),
                )

                full_cfg_condition_mask = torch.where(
                    torch.rand(size=(bs,), device=device) < 0.50,
                    torch.zeros(size=(bs,), device=device),
                    torch.ones(size=(bs,), device=device),
                ).long()
                speaker_embds = torch.where(
                    full_cfg_condition_mask.unsqueeze(1).bool(),
                    speaker_embds,
                    torch.zeros_like(speaker_embds),
                )

                full_cfg_condition_mask = torch.where(
                    torch.rand(size=(bs,), device=device) < 0.15,
                    torch.zeros(size=(bs,), device=device),
                    torch.ones(size=(bs,), device=device),
                ).long()
                lyric_token_ids = torch.where(
                    full_cfg_condition_mask.unsqueeze(1).bool(),
                    lyric_token_ids,
                    torch.zeros_like(lyric_token_ids),
                )
                lyric_mask = torch.where(
                    full_cfg_condition_mask.unsqueeze(1).bool(),
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

        def run_step(self, batch, batch_idx):
            self._apply_route(batch)
            return super().run_step(batch, batch_idx)

        def on_save_checkpoint(self, checkpoint):
            state = {}
            log_dir = self.logger.log_dir
            epoch = self.current_epoch
            step = self.global_step
            bundle_dir = Path(log_dir) / "checkpoints" / f"epoch={epoch}-step={step}_edm_control_lora"
            adapters_dir = bundle_dir / "adapters"
            adapters_dir.mkdir(parents=True, exist_ok=True)
            for adapter_name in self.all_adapter_names:
                adapter_dir = adapters_dir / adapter_name
                adapter_dir.mkdir(parents=True, exist_ok=True)
                self.transformers.save_lora_adapter(str(adapter_dir), adapter_name=adapter_name)
            conditioner_payload = None
            if self.control_conditioner is not None:
                conditioner_payload = {
                    "state_dict": self.control_conditioner.state_dict(),
                    "config": self.method_config.get("control_conditioner", {}),
                }
                torch.save(conditioner_payload, bundle_dir / "control_conditioner.pt")
            save_router_manifest(
                bundle_dir / "manifest.json",
                self.router,
                extra={
                    "global_step": int(step),
                    "epoch": int(epoch),
                    "method_config": self.method_config,
                    "control_conditioner": conditioner_payload is not None,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            return state

    return EDMControlPipeline


def dry_run(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    router = EDMAdapterRouter(RouterConfig.from_dict(config.get("router")))
    row = {
        "section": args.test_section,
        "energy": args.test_energy,
        "subgenre": args.test_subgenre,
        "bpm": args.test_bpm,
        "bpm_confidence": 0.9,
        "sample_weight": 1.0,
        "tag_confidence": {"section": 0.9, "energy": 0.85, "subgenre": 0.8},
    }
    print(json.dumps({
        "config": args.config,
        "dataset_path": args.dataset_path,
        "checkpoint_dir": args.checkpoint_dir,
        "adapter_count": len(router.adapter_names()),
        "route_weights": router.weights_for_batch([row]),
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "edm_control_lora.json"))
    parser.add_argument("--dataset-path", default=str(PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"))
    parser.add_argument("--logger-dir", default=str(PROJECT_ROOT / "outputs" / "edm_control_lora" / "logs"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "edm_control_lora"))
    parser.add_argument("--exp-name", default="edm_struct_lora")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--every-n-train-steps", type=int, default=1000)
    parser.add_argument("--every-plot-step", type=int, default=2000)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--disable-cached-latents", action="store_true")
    parser.add_argument("--disable-weighted-sampling", action="store_true")
    parser.add_argument("--enable-ssl-loss", action="store_true")
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-section", default="drop")
    parser.add_argument("--test-energy", default="high")
    parser.add_argument("--test-subgenre", default="melodic house")
    parser.add_argument("--test-bpm", type=float, default=128.0)
    args = parser.parse_args()

    if args.warmup_steps <= 0:
        args.warmup_steps = max(1, int(args.max_steps * 0.05))

    if args.dry_run:
        dry_run(args)
        return

    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    method_config = load_config(Path(args.config))
    lightning_cls = build_lightning_module_class()
    model = lightning_cls(
        method_config=method_config,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        use_cached_latents=not args.disable_cached_latents,
        weighted_sampling=not args.disable_weighted_sampling,
        enable_ssl_loss=args.enable_ssl_loss,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        train=args.enable_ssl_loss,
        T=1000,
        weight_decay=args.weight_decay,
        every_plot_step=args.every_plot_step,
        shift=3.0,
        ssl_coeff=1.0 if args.enable_ssl_loss else 0.0,
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        dataset_path=args.dataset_path,
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
        max_epochs=-1,
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        reload_dataloaders_every_n_epochs=1,
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    trainer.fit(model, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
