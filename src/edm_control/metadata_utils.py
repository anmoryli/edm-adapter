"""Helpers for reading metadata and resolving dataset-relative paths."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def to_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def resolve_dataset_path(dataset_root: str | Path, value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(dataset_root) / Path(str(value).replace("\\", "/"))


def confidence_weight(row: dict, floor: float = 0.25) -> float:
    """A conservative sample weight from quality and automatic tag confidence."""

    quality = float(row.get("quality_score") or 4.0)
    quality_weight = max(floor, min(1.0, quality / 5.0))
    bpm_conf = float(row.get("bpm_confidence") or 0.0)
    tag_conf = row.get("tag_confidence") or {}
    if isinstance(tag_conf, dict) and tag_conf:
        tag_weight = sum(float(v) for v in tag_conf.values()) / len(tag_conf)
    else:
        tag_weight = 0.60
    return max(floor, min(1.0, 0.50 * quality_weight + 0.25 * bpm_conf + 0.25 * tag_weight))
