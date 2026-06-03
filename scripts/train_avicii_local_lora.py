"""Low-resource Avicii-focused ACE-Step LoRA training.

This is a real ACE-Step LoRA fine-tune path: it loads the ACE-Step base
transformer, attaches a LoRA adapter, and optimizes the adapter on cached
Avicii latents. For CPU machines it freezes most LoRA tensors and only trains
the last N transformer blocks plus the final conditioning/output adapters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import glob
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
STYLE_PREFIX = (
    "avicii_adapter_style, uplifting progressive house, bright piano chords, "
    "emotional melodic lead, sidechain bass, polished festival EDM"
)


def configure_low_cpu(cpu_threads: int) -> None:
    cpu_threads = max(1, int(cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def load_lora_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_lora_bundle(root: Path) -> Path | None:
    candidates = [
        Path(path).parent
        for path in glob.glob(str(root / "**" / "manifest.json"), recursive=True)
    ]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, float]:
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            step = int(manifest.get("global_step") or 0)
        except Exception:
            step = 0
        return step, path.stat().st_mtime

    return sorted(candidates, key=sort_key, reverse=True)[0]


def build_lightning_module_class():
    import math

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from trainer import Pipeline as BasePipeline

    from src.edm_control.dataset import build_control_dataset_class

    class AviciiLocalLoRAPipeline(BasePipeline):
        def __init__(
            self,
            output_dir: str,
            train_last_n_blocks: int = 2,
            train_conditioning: bool = True,
            train_final_layer: bool = True,
            sample_size: int | None = None,
            style_prompt_dropout: float = 0.05,
            init_lora_bundle: str | None = None,
            **kwargs,
        ):
            self.output_dir = output_dir
            self.train_last_n_blocks = max(0, int(train_last_n_blocks))
            self.train_conditioning = bool(train_conditioning)
            self.train_final_layer = bool(train_final_layer)
            self.sample_size = sample_size
            self.style_prompt_dropout = float(style_prompt_dropout)
            self.init_lora_bundle = init_lora_bundle or ""
            self.step_offset = 0
            self.continued_from = ""
            self.ssl_coeff = 0.0
            kwargs["adapter_name"] = "avicii_style"
            kwargs["train"] = False
            super().__init__(**kwargs)
            self.is_train = True
            self.transformers.train()
            self._load_initial_lora_bundle()
            self._restrict_trainable_lora()
            self._print_trainable_summary()

        def _resolve_initial_adapter_dir(self) -> tuple[Path | None, dict]:
            if not self.init_lora_bundle:
                return None, {}
            bundle = Path(self.init_lora_bundle)
            if bundle.is_file():
                bundle = bundle.parent
            if (bundle / "pytorch_lora_weights.safetensors").exists():
                return bundle, {}
            manifest_path = bundle / "manifest.json"
            manifest = {}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            adapter_name = manifest.get("adapter_name", "avicii_style")
            adapter_dir = bundle / "adapters" / adapter_name
            if not (adapter_dir / "pytorch_lora_weights.safetensors").exists():
                raise FileNotFoundError(f"Missing initial LoRA weights under: {adapter_dir}")
            return adapter_dir, manifest

        def _load_initial_lora_bundle(self) -> None:
            adapter_dir, manifest = self._resolve_initial_adapter_dir()
            if adapter_dir is None:
                return
            from safetensors.torch import load_file

            weight_path = adapter_dir / "pytorch_lora_weights.safetensors"
            lora_state = load_file(str(weight_path), device="cpu")
            named_params = dict(self.transformers.named_parameters())
            copied = 0
            skipped = []
            with torch.no_grad():
                for key, value in lora_state.items():
                    candidates = [
                        key,
                        key.replace(".lora_A.weight", ".lora_A.avicii_style.weight"),
                        key.replace(".lora_B.weight", ".lora_B.avicii_style.weight"),
                    ]
                    target_key = next((candidate for candidate in candidates if candidate in named_params), None)
                    if target_key is None:
                        skipped.append(key)
                        continue
                    target = named_params[target_key]
                    if tuple(target.shape) != tuple(value.shape):
                        skipped.append(key)
                        continue
                    target.copy_(value.to(dtype=target.dtype, device=target.device))
                    copied += 1

            self.step_offset = int(manifest.get("global_step") or 0)
            self.continued_from = str(adapter_dir.parent.parent if manifest else adapter_dir)
            print(
                json.dumps(
                    {
                        "init_lora_bundle": self.continued_from,
                        "init_global_step": self.step_offset,
                        "loaded_lora_tensors": copied,
                        "skipped_lora_tensors": len(skipped),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if skipped:
                print(
                    "WARNING: skipped initial LoRA tensors: "
                    + ", ".join(skipped[:12])
                    + (" ..." if len(skipped) > 12 else ""),
                    flush=True,
                )

        def _is_trainable_lora_name(self, name: str) -> bool:
            if "lora_" not in name:
                return False
            if self.train_final_layer and "final_layer.linear" in name:
                return True
            if self.train_conditioning and any(
                part in name
                for part in (
                    "genre_embedder",
                    "speaker_embedder",
                    "t_block.1",
                )
            ):
                return True
            if self.train_last_n_blocks <= 0:
                return False
            num_layers = int(getattr(self.transformers.config, "num_layers", 24))
            first_trainable = max(0, num_layers - self.train_last_n_blocks)
            for idx in range(first_trainable, num_layers):
                if f"transformer_blocks.{idx}." in name:
                    return True
            return False

        def _restrict_trainable_lora(self) -> None:
            for name, parameter in self.transformers.named_parameters():
                parameter.requires_grad_(self._is_trainable_lora_name(name))

        def _print_trainable_summary(self) -> None:
            trainable = [
                (name, parameter)
                for name, parameter in self.transformers.named_parameters()
                if parameter.requires_grad
            ]
            if not trainable:
                raise RuntimeError("No trainable LoRA parameters were left enabled.")
            total = sum(parameter.numel() for _, parameter in trainable)
            print(
                json.dumps(
                    {
                        "adapter": "avicii_style",
                        "trainable_tensors": len(trainable),
                        "trainable_params": total,
                        "init_global_step": self.step_offset,
                        "continued_from": self.continued_from,
                        "train_last_n_blocks": self.train_last_n_blocks,
                        "train_conditioning": self.train_conditioning,
                        "train_final_layer": self.train_final_layer,
                    },
                    indent=2,
                ),
                flush=True,
            )

        def train_dataloader(self):
            dataset_cls = build_control_dataset_class()
            self.train_dataset = dataset_cls(
                train=True,
                train_dataset_path=self.hparams.dataset_path,
                sample_size=self.sample_size,
            )

            weights = []
            if hasattr(self.train_dataset, "pretrain_ds"):
                for item in self.train_dataset.pretrain_ds:
                    section = item.get("section", "")
                    energy = item.get("energy", "")
                    subgenre = item.get("subgenre", "")
                    weight = float(item.get("sample_weight") or 1.0)
                    if section == "drop" and energy in {"high", "very_high"}:
                        weight *= 3.0
                    elif section == "drop":
                        weight *= 2.0
                    elif section in {"build-up", "loop"}:
                        weight *= 1.5
                    if subgenre in {"progressive house", "melodic house", "festival EDM", "tropical house"}:
                        weight *= 1.25
                    weights.append(weight)

            sampler = None
            shuffle = True
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
                        if torch.rand(()) < self.style_prompt_dropout:
                            continue
                        if isinstance(item["prompts"], list):
                            item["prompts"] = [
                                f"{STYLE_PREFIX}, {prompt}" for prompt in item["prompts"]
                            ]
                        elif isinstance(item["prompts"], str):
                            item["prompts"] = f"{STYLE_PREFIX}, {item['prompts']}"
                return original_collate(batch)

            return DataLoader(
                self.train_dataset,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=False,
                collate_fn=trigger_collate,
            )

        def preprocess(self, batch, train=True):
            if "target_latents" not in batch:
                return super().preprocess(batch, train=train)

            target_latents = batch["target_latents"]
            device = target_latents.device
            dtype = target_latents.dtype
            bs = target_latents.shape[0]

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
                keep_text = (torch.rand(size=(bs,), device=device) >= 0.08).long()
                encoder_text_hidden_states = torch.where(
                    keep_text.unsqueeze(1).unsqueeze(1).bool(),
                    encoder_text_hidden_states,
                    torch.zeros_like(encoder_text_hidden_states),
                )
                keep_speaker = (torch.rand(size=(bs,), device=device) >= 0.50).long()
                speaker_embds = torch.where(
                    keep_speaker.unsqueeze(1).bool(),
                    speaker_embds,
                    torch.zeros_like(speaker_embds),
                )
                keep_lyrics = (torch.rand(size=(bs,), device=device) >= 0.20).long()
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
                None,
                None,
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
                betas=(0.8, 0.9),
            )
            max_steps = self.hparams.max_steps
            warmup_steps = self.hparams.warmup_steps

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        @staticmethod
        def masked_mse(pred, target, mask):
            selected_model_pred = (pred * mask).reshape(pred.shape[0], -1).contiguous()
            selected_target = (target * mask).reshape(target.shape[0], -1).contiguous()
            loss = F.mse_loss(selected_model_pred, selected_target, reduction="none")
            loss = loss.mean(1)
            loss = loss * mask.reshape(mask.shape[0], -1).mean(1)
            return loss.mean()

        def run_step(self, batch, batch_idx):
            (
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
            ) = self.preprocess(batch)

            target_image = target_latents
            device = target_image.device
            dtype = target_image.dtype
            noise = torch.randn_like(target_image, device=device)
            timesteps = self.get_timestep(target_image.shape[0], device)
            sigmas = self.get_sd3_sigmas(
                timesteps=timesteps,
                device=device,
                n_dim=target_image.ndim,
                dtype=dtype,
            )
            noisy_image = sigmas * noise + (1.0 - sigmas) * target_image

            output = self.transformers(
                hidden_states=noisy_image,
                attention_mask=attention_mask,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
                timestep=timesteps.to(device).to(dtype),
                ssl_hidden_states=[],
            )
            pred = output.sample * (-sigmas) + noisy_image
            mask = (
                attention_mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, target_image.shape[1], target_image.shape[2], -1)
            )
            loss = self.masked_mse(pred, target_image, mask)
            if not torch.isfinite(loss):
                loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=1.0)

            self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            if self.lr_schedulers() is not None:
                self.log(
                    "train/lr",
                    self.lr_schedulers().get_last_lr()[0],
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )
            return loss

        def save_adapter_bundle(self, step: int | None = None) -> Path:
            log_dir = Path(self.logger.log_dir)
            local_step = self.global_step if step is None else step
            cumulative_step = int(self.step_offset + int(local_step))
            bundle_dir = log_dir / "checkpoints" / f"step={cumulative_step}_avicii_local_lora"
            adapter_dir = bundle_dir / "adapters" / "avicii_style"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.transformers.save_lora_adapter(str(adapter_dir), adapter_name="avicii_style")
            manifest = {
                "format": "avicii_local_lora_v1",
                "adapter_name": "avicii_style",
                "trigger_word": TRIGGER_WORD,
                "style_prefix": STYLE_PREFIX,
                "global_step": int(cumulative_step),
                "local_step": int(local_step),
                "init_global_step": int(self.step_offset),
                "continued_from": self.continued_from,
                "train_last_n_blocks": self.train_last_n_blocks,
                "train_conditioning": self.train_conditioning,
                "train_final_layer": self.train_final_layer,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (bundle_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return bundle_dir

    return AviciiLocalLoRAPipeline


def build_adapter_checkpoint_callback(every_n_train_steps: int):
    from pytorch_lightning.callbacks import Callback

    class AdapterCheckpointCallback(Callback):
        def __init__(self, every_n: int):
            self.every_n = max(1, int(every_n))
            self.saved_steps: set[int] = set()

        def _save(self, trainer, pl_module) -> None:
            step = int(trainer.global_step)
            if step <= 0 or step in self.saved_steps:
                return
            bundle_dir = pl_module.save_adapter_bundle(step=step)
            self.saved_steps.add(step)
            print(f"Saved Avicii LoRA bundle: {bundle_dir}", flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if int(trainer.global_step) % self.every_n == 0:
                self._save(trainer, pl_module)

        def on_train_end(self, trainer, pl_module):
            self._save(trainer, pl_module)

    return AdapterCheckpointCallback(every_n_train_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"),
    )
    parser.add_argument(
        "--lora-config-path",
        default=str(PROJECT_ROOT / "config" / "avicii_local_lora.json"),
    )
    parser.add_argument(
        "--logger-dir",
        default=str(PROJECT_ROOT / "outputs" / "avicii_local_lora" / "logs"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "avicii_local_lora"),
    )
    parser.add_argument("--exp-name", default="avicii_local_lora")
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--every-n-train-steps", type=int, default=40)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=512, help="Training subset size; use 0 to scan the full training split.")
    parser.add_argument("--train-last-n-blocks", type=int, default=1)
    parser.add_argument("--no-train-conditioning", action="store_true")
    parser.add_argument("--no-train-final-layer", action="store_true")
    parser.add_argument("--style-prompt-dropout", type=float, default=0.05)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--init-lora-bundle", default=None)
    parser.add_argument("--auto-init-latest-lora", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_low_cpu(args.cpu_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_lora_bundle = args.init_lora_bundle
    if args.auto_init_latest_lora and not init_lora_bundle:
        latest = find_latest_lora_bundle(Path(args.logger_dir))
        if latest is not None:
            init_lora_bundle = str(latest)
            print(f"Auto-selected latest LoRA bundle for continuation: {init_lora_bundle}", flush=True)
        else:
            print("No existing LoRA bundle found; training will start from a fresh adapter.", flush=True)
    lora_config = load_lora_config(Path(args.lora_config_path))
    runtime_config = output_dir / "runtime_lora_config.json"
    runtime_config.write_text(json.dumps(lora_config, indent=2), encoding="utf-8")

    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger

    lightning_cls = build_lightning_module_class()
    model = lightning_cls(
        output_dir=args.output_dir,
        train_last_n_blocks=args.train_last_n_blocks,
        train_conditioning=not args.no_train_conditioning,
        train_final_layer=not args.no_train_final_layer,
        sample_size=None if int(args.sample_size) <= 0 else args.sample_size,
        style_prompt_dropout=args.style_prompt_dropout,
        init_lora_bundle=init_lora_bundle,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        train=False,
        T=1000,
        weight_decay=args.weight_decay,
        every_plot_step=max(args.max_steps + 1, 1000000),
        shift=3.0,
        ssl_coeff=0.0,
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        dataset_path=args.dataset_path,
        lora_config_path=str(runtime_config),
    )

    if args.init_only:
        print("Initialization succeeded; no training was run.", flush=True)
        return

    checkpoint_callback = build_adapter_checkpoint_callback(args.every_n_train_steps)
    logger_callback = TensorBoardLogger(
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + args.exp_name,
        save_dir=args.logger_dir,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="32",
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy="auto",
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        enable_checkpointing=False,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(model, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
