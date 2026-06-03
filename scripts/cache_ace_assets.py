"""Cache ACE-Step training-side assets for the cleaned EDM dataset.

This script does not train anything. It adds:
- dataset/latents/*.pt: ACE-Step MusicDCAE f8c8 audio latents.
- dataset/tokens/*.pt: ACE text-conditioning tokenizer outputs for captions.

ACE-Step is a latent diffusion / flow-matching model, not an autoregressive
audio-token model. Therefore this script intentionally does not create EnCodec
or SoundStream audio tokens, and leaves audio_token_path empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
ACE_STEP_ROOT = PROJECT_ROOT / "ACE-Step"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "ace-step" / "ACE-Step-v1-3.5B"

LATENTS_DIR = DATASET_ROOT / "latents"
TOKENS_DIR = DATASET_ROOT / "tokens"
SPLITS_DIR = DATASET_ROOT / "splits"
REPORTS_DIR = DATASET_ROOT / "reports"
METADATA_JSONL = DATASET_ROOT / "metadata.jsonl"
METADATA_CSV = DATASET_ROOT / "metadata.csv"

sys.path.insert(0, str(ACE_STEP_ROOT))

import torch  # noqa: E402
import soundfile as sf  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE  # noqa: E402


def posix_rel(path: Path, base: Path | None = None) -> str:
    if base is None:
        base = DATASET_ROOT
    return path.relative_to(base).as_posix()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_safe) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(flat)


def resolve_dataset_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return DATASET_ROOT / value


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_audio_for_ace(path: Path) -> tuple[torch.Tensor, int]:
    audio_np, sr = sf.read(str(path), always_2d=True, dtype="float32")
    audio = torch.from_numpy(audio_np.T)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    if audio.shape[0] > 2:
        audio = audio[:2]
    audio = torch.clamp(audio.float(), -1.0, 1.0)
    return audio, int(sr)


def load_dcae(device: torch.device) -> MusicDCAE:
    dcae_path = CHECKPOINT_DIR / "music_dcae_f8c8"
    vocoder_path = CHECKPOINT_DIR / "music_vocoder"
    if not dcae_path.exists():
        raise FileNotFoundError(f"Missing ACE DCAE checkpoint: {dcae_path}")
    if not vocoder_path.exists():
        raise FileNotFoundError(f"Missing ACE vocoder checkpoint: {vocoder_path}")
    model = MusicDCAE(
        source_sample_rate=44100,
        dcae_checkpoint_path=str(dcae_path),
        vocoder_checkpoint_path=str(vocoder_path),
    )
    model.eval().to(device=device, dtype=torch.float32)
    model.requires_grad_(False)
    return model


def cache_text_tokens(row: dict[str, Any], tokenizer: AutoTokenizer, token_path: Path, force: bool) -> dict[str, Any]:
    if token_path.exists() and not force:
        return torch.load(str(token_path), map_location="cpu")

    caption = row.get("caption") or ""
    encoded = tokenizer(
        caption,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=256,
    )
    payload = {
        "clip_id": row["clip_id"],
        "tokenizer": "ACE-Step umt5-base",
        "token_type": "text_conditioning",
        "source_field": "caption",
        "input_ids": encoded["input_ids"][0].cpu(),
        "attention_mask": encoded["attention_mask"][0].cpu(),
    }
    token_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(token_path))
    return payload


@torch.no_grad()
def encode_audio_batch(
    dcae: MusicDCAE,
    audios: list[torch.Tensor],
    sample_rate: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([audio.shape[-1] for audio in audios], device=device)
    max_len = int(max(audio.shape[-1] for audio in audios))
    batch = torch.zeros((len(audios), 2, max_len), dtype=torch.float32, device=device)
    for index, audio in enumerate(audios):
        batch[index, :, : audio.shape[-1]] = audio.to(device=device, dtype=torch.float32)

    if sample_rate != 44100:
        # The cleaned dataset clips are 44.1 kHz. Keep a safe fallback for any
        # future mixed-rate rows without duplicating ACE's resampling logic here.
        return dcae.encode(batch, audio_lengths=lengths, sr=sample_rate)

    max_audio_len = batch.shape[-1]
    if max_audio_len % (8 * 512) != 0:
        batch = torch.nn.functional.pad(batch, (0, 8 * 512 - max_audio_len % (8 * 512)))

    mels = dcae.forward_mel(batch)
    mels = (mels - dcae.min_mel_value) / (dcae.max_mel_value - dcae.min_mel_value)
    mels = dcae.transform(mels)
    try:
        latents = dcae.dcae.encoder(mels)
    except RuntimeError:
        # Conservative fallback for environments where the DCAE encoder cannot
        # batch this tensor shape.
        latents = torch.cat([dcae.dcae.encoder(mel.unsqueeze(0)) for mel in mels], dim=0)
    latent_lengths = (lengths / sample_rate * 44100 / 512 / dcae.time_dimention_multiple).long()
    latents = (latents - dcae.shift_factor) * dcae.scale_factor
    return latents, latent_lengths


@torch.no_grad()
def cache_latent_batch(
    rows: list[dict[str, Any]],
    dcae: MusicDCAE,
    device: torch.device,
    force: bool,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], Path, torch.Tensor, int]] = []

    for row in rows:
        latent_path = LATENTS_DIR / f"{row['clip_id']}.pt"
        if latent_path.exists() and not force:
            results[row["clip_id"]] = torch.load(str(latent_path), map_location="cpu")
            continue
        audio_path = resolve_dataset_path(row["audio_path"])
        audio, sr = load_audio_for_ace(audio_path)
        pending.append((row, latent_path, audio, sr))

    if not pending:
        return results

    sample_rates = {sr for _, _, _, sr in pending}
    if len(sample_rates) != 1:
        # Mixed sample rates are not expected for this cleaned dataset. Grouping
        # keeps the batch encoder correct if future data changes.
        for sr in sorted(sample_rates):
            group_rows = [item[0] for item in pending if item[3] == sr]
            results.update(cache_latent_batch(group_rows, dcae, device, force))
        return results

    sample_rate = pending[0][3]
    audios = [item[2] for item in pending]
    latents, latent_lengths = encode_audio_batch(dcae, audios, sample_rate, device)

    for index, (row, latent_path, audio, sr) in enumerate(pending):
        latent = latents[index].detach().cpu().float()
        payload = {
            "clip_id": row["clip_id"],
            "source_audio_path": row["audio_path"],
            "encoder": "ACE-Step MusicDCAE f8c8",
            "latent_type": "flow_matching_diffusion_target",
            "input_sample_rate": sr,
            "internal_sample_rate": 44100,
            "input_channels": int(audio.shape[0]),
            "input_samples": int(audio.shape[-1]),
            "latent_length": int(latent_lengths[index].detach().cpu().item()),
            "latent_shape": list(latent.shape),
            "latent_dtype": str(latent.dtype).replace("torch.", ""),
            "latent": latent,
        }
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(latent_path))
        results[row["clip_id"]] = payload

    return results


@torch.no_grad()
def cache_latent(
    row: dict[str, Any],
    dcae: MusicDCAE,
    device: torch.device,
    latent_path: Path,
    force: bool,
) -> dict[str, Any]:
    if latent_path.exists() and not force:
        return torch.load(str(latent_path), map_location="cpu")

    audio_path = resolve_dataset_path(row["audio_path"])
    audio, sr = load_audio_for_ace(audio_path)
    batch = audio.unsqueeze(0).to(device=device, dtype=torch.float32)
    latents, latent_lengths = dcae.encode(batch, sr=sr)
    latent = latents[0].detach().cpu().float()
    latent_length = int(latent_lengths[0].detach().cpu().item())

    payload = {
        "clip_id": row["clip_id"],
        "source_audio_path": row["audio_path"],
        "encoder": "ACE-Step MusicDCAE f8c8",
        "latent_type": "flow_matching_diffusion_target",
        "input_sample_rate": sr,
        "internal_sample_rate": 44100,
        "input_channels": int(audio.shape[0]),
        "input_samples": int(audio.shape[-1]),
        "latent_length": latent_length,
        "latent_shape": list(latent.shape),
        "latent_dtype": str(latent.dtype).replace("torch.", ""),
        "latent": latent,
    }
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(latent_path))
    return payload


def update_row_metadata(
    row: dict[str, Any],
    latent_rel: str,
    token_rel: str,
    latent_payload: dict[str, Any],
    token_payload: dict[str, Any],
) -> None:
    row["latent_path"] = latent_rel
    row["text_token_path"] = token_rel
    row["audio_token_path"] = ""
    row["audio_token_status"] = "not_applicable_for_ace_step_latent_diffusion"
    row["codec"] = "ACE-Step MusicDCAE f8c8"
    row["codec_sample_rate"] = 44100
    row["codebook_count"] = 0
    row["latent_config"] = {
        "encoder": "ACE-Step MusicDCAE f8c8",
        "latent_type": "flow_matching_diffusion_target",
        "input_sample_rate": latent_payload["input_sample_rate"],
        "internal_sample_rate": latent_payload["internal_sample_rate"],
        "latent_shape": latent_payload["latent_shape"],
        "latent_length": latent_payload["latent_length"],
        "latent_dtype": latent_payload["latent_dtype"],
    }
    row["text_token_config"] = {
        "tokenizer": token_payload["tokenizer"],
        "token_type": token_payload["token_type"],
        "source_field": token_payload["source_field"],
        "max_length": 256,
        "token_count": int(token_payload["input_ids"].numel()),
    }
    notes = list(row.get("processing_notes") or [])
    for note in [
        "ACE-Step MusicDCAE f8c8 latent cached in latents/",
        "ACE text-conditioning tokens cached in tokens/",
        "No EnCodec/SoundStream audio tokens were generated because ACE-Step is latent diffusion, not autoregressive audio-token generation.",
    ]:
        if note not in notes:
            notes.append(note)
    row["processing_notes"] = notes


def write_manifests(rows: list[dict[str, Any]], started_at: float, device: torch.device) -> None:
    elapsed = time.time() - started_at
    latent_rows = [row for row in rows if row.get("latent_path")]
    token_rows = [row for row in rows if row.get("text_token_path")]

    latent_manifest = {
        "asset_type": "ACE-Step MusicDCAE f8c8 latents",
        "model_family": "latent_diffusion_flow_matching",
        "count": len(latent_rows),
        "device_used": str(device),
        "elapsed_seconds": round(elapsed, 2),
        "note": "These latents match ACE-Step's DCAE representation and are not EnCodec audio tokens.",
    }
    token_manifest = {
        "asset_type": "ACE-Step text-conditioning tokens",
        "tokenizer": "umt5-base",
        "count": len(token_rows),
        "audio_tokens": "not_applicable_for_ace_step_latent_diffusion",
        "note": "For ACE-Step, tokens are text-conditioning tokens. Autoregressive audio tokens are intentionally not generated.",
    }
    LATENTS_DIR.mkdir(parents=True, exist_ok=True)
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    (LATENTS_DIR / "manifest.json").write_text(
        json.dumps(latent_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (TOKENS_DIR / "manifest.json").write_text(
        json.dumps(token_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (TOKENS_DIR / "README.md").write_text(
        "# ACE-Step Tokens\n\n"
        "This directory stores ACE-Step text-conditioning tokenizer outputs for captions.\n\n"
        "ACE-Step does not use EnCodec/SoundStream audio tokens for training. It uses MusicDCAE latents in `../latents/`.\n",
        encoding="utf-8",
    )


def write_cache_report(rows: list[dict[str, Any]], failures: list[dict[str, str]], started_at: float, device: torch.device) -> None:
    elapsed = time.time() - started_at
    report = f"""# ACE Asset Cache Report

## Summary

- Metadata rows: {len(rows)}
- Cached ACE latents: {sum(1 for row in rows if row.get("latent_path"))}
- Cached text token files: {sum(1 for row in rows if row.get("text_token_path"))}
- Failures: {len(failures)}
- Device used: {device}
- Elapsed seconds: {elapsed:.2f}

## Representation

- Latents: ACE-Step MusicDCAE f8c8, flow-matching diffusion target.
- Text tokens: ACE-Step `umt5-base` tokenizer outputs for captions.
- Audio tokens: not applicable for ACE-Step; EnCodec/SoundStream tokens were intentionally not generated.

## Failure List

{json.dumps(failures[:200], ensure_ascii=False, indent=2)}
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "ace_cache_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    global DATASET_ROOT, CHECKPOINT_DIR, LATENTS_DIR, TOKENS_DIR, SPLITS_DIR, REPORTS_DIR, METADATA_JSONL, METADATA_CSV

    parser = argparse.ArgumentParser(description="Cache ACE-Step latents and text tokens for dataset clips.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="Dataset root containing metadata.jsonl and clips.")
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR), help="ACE-Step checkpoint directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--max-items", type=int, default=None, help="Debug limit.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of clips to encode per batch.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing cache files.")
    parser.add_argument("--no-update-metadata", action="store_true", help="Do not rewrite metadata/splits after caching.")
    args = parser.parse_args()

    DATASET_ROOT = Path(args.dataset_root).resolve()
    CHECKPOINT_DIR = Path(args.checkpoint_dir).resolve()
    LATENTS_DIR = DATASET_ROOT / "latents"
    TOKENS_DIR = DATASET_ROOT / "tokens"
    SPLITS_DIR = DATASET_ROOT / "splits"
    REPORTS_DIR = DATASET_ROOT / "reports"
    METADATA_JSONL = DATASET_ROOT / "metadata.jsonl"
    METADATA_CSV = DATASET_ROOT / "metadata.csv"

    started_at = time.time()
    rows = load_jsonl(METADATA_JSONL)
    if args.max_items is not None:
        work_rows = rows[: args.max_items]
    else:
        work_rows = rows

    device = choose_device(args.device)
    tokenizer_path = CHECKPOINT_DIR / "umt5-base"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing ACE text tokenizer: {tokenizer_path}")

    print(f"Rows: {len(rows)}; work rows: {len(work_rows)}", flush=True)
    print(f"Device: {device}", flush=True)
    print("Loading ACE MusicDCAE...", flush=True)
    dcae = load_dcae(device)
    print("Loading ACE text tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

    failures: list[dict[str, str]] = []
    processed = 0

    batch_size = max(1, int(args.batch_size))
    for batch_start in range(0, len(work_rows), batch_size):
        batch_rows = work_rows[batch_start : batch_start + batch_size]
        try:
            latent_payloads = cache_latent_batch(batch_rows, dcae, device, args.force)
        except Exception as batch_exc:
            print(f"[batch {batch_start + 1}] batch encode failed, falling back row-by-row: {batch_exc}", flush=True)
            latent_payloads = {}
            for row in batch_rows:
                clip_id = row["clip_id"]
                try:
                    latent_path = LATENTS_DIR / f"{clip_id}.pt"
                    latent_payloads[clip_id] = cache_latent(row, dcae, device, latent_path, args.force)
                except Exception as exc:
                    failures.append({"clip_id": clip_id, "reason": type(exc).__name__, "message": str(exc)})
                    print(f"[{batch_start + 1}/{len(work_rows)}] failed {clip_id}: {exc}", flush=True)

        for offset, row in enumerate(batch_rows):
            index = batch_start + offset + 1
            clip_id = row["clip_id"]
            latent_path = LATENTS_DIR / f"{clip_id}.pt"
            token_path = TOKENS_DIR / f"{clip_id}.pt"
            try:
                if clip_id not in latent_payloads:
                    continue
                latent_payload = latent_payloads[clip_id]
                token_payload = cache_text_tokens(row, tokenizer, token_path, args.force)
                update_row_metadata(
                    row,
                    posix_rel(latent_path),
                    posix_rel(token_path),
                    latent_payload,
                    token_payload,
                )
                processed += 1
                if index == 1 or index % 25 == 0 or index == len(work_rows):
                    print(
                        f"[{index}/{len(work_rows)}] cached {clip_id} latent_shape={latent_payload['latent_shape']}",
                        flush=True,
                    )
            except Exception as exc:
                failures.append({"clip_id": clip_id, "reason": type(exc).__name__, "message": str(exc)})
                print(f"[{index}/{len(work_rows)}] failed {clip_id}: {exc}", flush=True)

    if args.max_items is None and not args.no_update_metadata:
        write_jsonl(METADATA_JSONL, rows)
        write_csv(METADATA_CSV, rows)
        for split in ["train", "val", "test"]:
            write_jsonl(SPLITS_DIR / f"{split}.jsonl", [row for row in rows if row.get("split") == split])

    write_manifests(rows, started_at, device)
    write_cache_report(rows, failures, started_at, device)

    print("\nACE asset cache complete.", flush=True)
    print(f"processed: {processed}", flush=True)
    print(f"failures: {len(failures)}", flush=True)
    print(f"latents_dir: {LATENTS_DIR}", flush=True)
    print(f"tokens_dir: {TOKENS_DIR}", flush=True)

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
