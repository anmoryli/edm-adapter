"""Convert cleaned EDM metadata into an ACE-Step/HuggingFace training dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.edm_control.metadata_utils import confidence_weight, read_jsonl, resolve_dataset_path  # noqa: E402


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _compact_tags(row: dict) -> list[str]:
    tags: list[str] = []
    for key in ["genre", "subgenre", "section", "energy", "bass", "vocal"]:
        value = row.get(key)
        if value:
            tags.append(str(value))
    bpm = row.get("bpm")
    if bpm:
        tags.append(f"{int(round(float(bpm)))} BPM")
    for key in ["mood", "instruments", "drums", "production"]:
        tags.extend(_as_list(row.get(key)))
    seen = set()
    unique: list[str] = []
    for tag in tags:
        clean = tag.strip()
        if clean and clean not in seen:
            unique.append(clean)
            seen.add(clean)
    return unique[:32]


def _recaptions(row: dict) -> dict:
    bpm = row.get("bpm") or "unknown"
    subgenre = row.get("subgenre") or "EDM"
    section = row.get("section") or "unknown"
    energy = row.get("energy") or "medium"
    mood = ", ".join(_as_list(row.get("mood"))[:2]) or "electronic"
    instruments = ", ".join(_as_list(row.get("instruments"))[:5])
    production = ", ".join(_as_list(row.get("production"))[:4])
    return {
        "caption_full": row.get("caption", ""),
        "caption_short": f"{subgenre}, {bpm} BPM, {section}, {energy} energy.",
        "caption_structure": f"A {section} section in {subgenre} with {energy} energy and {mood} mood.",
        "caption_production": f"An EDM clip with {instruments} and {production}.",
    }


def _path_value(dataset_root: Path, row: dict, field: str, path_mode: str) -> str:
    path = resolve_dataset_path(dataset_root, row.get(field))
    if path is None:
        return ""
    if path_mode == "project-relative":
        return path.relative_to(PROJECT_ROOT).as_posix()
    return str(path.resolve())


def _project_rel_value(dataset_root: Path, row: dict, field: str, path_mode: str) -> str:
    value = _path_value(dataset_root, row, field, path_mode)
    if path_mode == "project-relative":
        return value
    try:
        return Path(value).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return value


def build_examples(
    dataset_root: Path,
    split: str,
    path_mode: str,
    min_quality: int,
    max_examples: int | None = None,
) -> list[dict]:
    dataset_root = dataset_root.resolve()
    split_path = dataset_root / "splits" / f"{split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    rows = read_jsonl(split_path)
    examples: list[dict] = []
    for row in rows:
        if int(row.get("quality_score") or 0) < min_quality:
            continue
        filename = _path_value(dataset_root, row, "audio_path", path_mode)
        latent_path = _path_value(dataset_root, row, "latent_path", path_mode)
        control_path = _path_value(dataset_root, row, "control_path", path_mode)
        text_token_path = _path_value(dataset_root, row, "text_token_path", path_mode)
        if not filename or not latent_path or not control_path:
            continue
        example = {
            "keys": row["clip_id"],
            "filename": filename,
            "project_rel_filename": _project_rel_value(dataset_root, row, "audio_path", path_mode),
            "tags": _compact_tags(row),
            "speaker_emb_path": "",
            "norm_lyrics": "[instrumental]",
            "recaption": _recaptions(row),
            "caption": row.get("caption", ""),
            "section": row.get("section", "unknown"),
            "energy": row.get("energy", "medium"),
            "subgenre": row.get("subgenre", "unknown"),
            "bpm": float(row.get("bpm") or 128.0),
            "bpm_confidence": float(row.get("bpm_confidence") or 0.0),
            "quality_score": int(row.get("quality_score") or 4),
            "sample_weight": float(row.get("sample_weight") or confidence_weight(row)),
            "tag_confidence": row.get("tag_confidence") or {},
            "latent_path": latent_path,
            "control_path": control_path,
            "text_token_path": text_token_path,
            "split": split,
            "source_id": row.get("source_id", ""),
            "clip_id": row["clip_id"],
        }
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "datasets" / "edm_control_lora_train"))
    parser.add_argument("--path-mode", default="absolute", choices=["absolute", "project-relative"])
    parser.add_argument("--min-quality", type=int, default=4)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    examples = build_examples(
        dataset_root=Path(args.dataset_root),
        split=args.split,
        path_mode=args.path_mode,
        min_quality=args.min_quality,
        max_examples=args.max_examples,
    )
    if args.repeat_count > 1:
        examples = examples * args.repeat_count

    summary = {
        "examples": len(examples),
        "split": args.split,
        "output": args.output,
        "path_mode": args.path_mode,
        "sample": examples[0] if examples else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit("Please install datasets: pip install datasets") from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(examples).save_to_disk(str(output_path))
    (output_path / "edm_control_dataset_info.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
