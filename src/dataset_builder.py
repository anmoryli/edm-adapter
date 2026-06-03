"""Dataset building: metadata loading, filtering, splitting."""

import json
import os
import random
from pathlib import Path

import pandas as pd


def load_metadata_csv(path: str) -> pd.DataFrame:
    """Load metadata from CSV."""
    return pd.read_csv(path)


def load_metadata_jsonl(path: str) -> list[dict]:
    """Load metadata from JSONL."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_metadata_jsonl(records: list[dict], path: str):
    """Save metadata to JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_metadata_csv(records: list[dict], path: str):
    """Save metadata to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)


def filter_electronic_tracks(
    df: pd.DataFrame,
    electronic_tags: list[str],
    exclude_tags: list[str],
    tag_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter dataframe for electronic music tracks.

    Args:
        df: DataFrame with track metadata.
        electronic_tags: Tags indicating electronic music.
        exclude_tags: Tags to exclude.
        tag_columns: Column names containing tags (comma-separated strings or lists).
    """
    if tag_columns is None:
        # Try to auto-detect tag columns
        tag_columns = [c for c in df.columns if "tag" in c.lower() or "genre" in c.lower() or "label" in c.lower()]
        if not tag_columns:
            tag_columns = [c for c in df.columns if c not in ("track_id", "file_path", "duration")]

    electronic_tags_lower = [t.lower() for t in electronic_tags]
    exclude_tags_lower = [t.lower() for t in exclude_tags]

    def has_electronic_tag(row):
        all_tags = []
        for col in tag_columns:
            val = row.get(col, "")
            if isinstance(val, str):
                all_tags.extend([t.strip().lower() for t in val.split(",") if t.strip()])
            elif isinstance(val, list):
                all_tags.extend([t.lower() for t in val])

        # Check for exclude tags
        for et in exclude_tags_lower:
            if et in " ".join(all_tags):
                return False

        # Check for electronic tags
        for et in electronic_tags_lower:
            for tag in all_tags:
                if et in tag:
                    return True
        return False

    mask = df.apply(has_electronic_tag, axis=1)
    return df[mask].reset_index(drop=True)


def split_dataset(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    group_key: str = "track_id",
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split dataset into train/val/test, grouped by track_id to avoid data leakage.

    Args:
        records: List of record dicts.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        group_key: Key to group by (segments from same track stay together).
        seed: Random seed.
    """
    random.seed(seed)

    # Group by track_id
    groups: dict[str, list[dict]] = {}
    for rec in records:
        key = rec.get(group_key, rec.get("audio_path", ""))
        groups.setdefault(key, []).append(rec)

    group_ids = list(groups.keys())
    random.shuffle(group_ids)

    n = len(group_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(group_ids[:n_train])
    val_ids = set(group_ids[n_train:n_train + n_val])
    test_ids = set(group_ids[n_train + n_val:])

    train_records = [r for gid in train_ids for r in groups[gid]]
    val_records = [r for gid in val_ids for r in groups[gid]]
    test_records = [r for gid in test_ids for r in groups[gid]]

    return train_records, val_records, test_records
