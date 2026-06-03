"""Build a section-aware 15 second Avicii LoRA dataset.

The script creates a separate dataset root with only the derived 15 second
clips and metadata. It does not modify the existing ``dataset/`` directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "avicii_section15_v2"

STYLE_TRIGGER = "avicii_adapter_style"

SECTION_RATIOS = {
    "drop": 0.36,
    "breakdown": 0.24,
    "build-up": 0.18,
    "intro": 0.10,
    "loop": 0.07,
    "outro": 0.05,
}

SECTION_PRIORITY = ["drop", "breakdown", "build-up", "intro", "loop", "outro"]
ENERGY_RANK = {"low": 0, "medium": 1, "high": 2, "very_high": 3}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def resolve_project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        DATASET_ROOT / value.replace("\\", "/"),
        PROJECT_ROOT / value.replace("\\", "/"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def safe_slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "clip"


def hash_bucket(value: str, modulo: int = 100) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def split_for_source(source_id: str) -> str:
    bucket = hash_bucket(source_id)
    if bucket < 84:
        return "train"
    if bucket < 92:
        return "val"
    return "test"


def normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def source_matches(row: dict[str, Any], keywords: list[str]) -> bool:
    if not keywords:
        return True
    text = " ".join(
        str(row.get(key) or "")
        for key in ["source_id", "source_file", "processed_audio_path"]
    ).lower()
    return any(keyword.lower() in text for keyword in keywords)


def row_audio_features(row: dict[str, Any]) -> dict[str, float]:
    features = row.get("audio_features") or {}
    return {
        "rms": as_float(features.get("rms"), 0.0),
        "peak": as_float(features.get("peak"), 0.0),
        "onset_density": as_float(features.get("onset_density"), 0.0),
        "low_freq_ratio": as_float(features.get("low_freq_ratio"), 0.0),
        "spectral_centroid": as_float(features.get("spectral_centroid"), 0.0),
        "zero_crossing_rate": as_float(features.get("zero_crossing_rate"), 0.0),
    }


def overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def aggregate_window(source_rows: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    overlaps: list[tuple[float, dict[str, Any]]] = []
    for row in source_rows:
        row_start = as_float(row.get("start_time"), 0.0)
        row_end = as_float(row.get("end_time"), row_start + as_float(row.get("duration"), 8.0))
        overlap = overlap_seconds(start, end, row_start, row_end)
        if overlap > 0:
            overlaps.append((overlap, row))

    if not overlaps:
        nearest = min(source_rows, key=lambda item: abs(as_float(item.get("start_time"), 0.0) - start))
        overlaps = [(1.0, nearest)]

    total_weight = sum(weight for weight, _ in overlaps) or 1.0
    features: dict[str, float] = {}
    for key in ["rms", "peak", "onset_density", "low_freq_ratio", "spectral_centroid", "zero_crossing_rate"]:
        features[key] = sum(row_audio_features(row)[key] * weight for weight, row in overlaps) / total_weight

    sections: Counter[str] = Counter()
    subgenres: Counter[str] = Counter()
    energies: Counter[str] = Counter()
    bpms: list[float] = []
    quality_scores: list[int] = []
    moods: list[str] = []
    instruments: list[str] = []
    drums: list[str] = []
    production: list[str] = []
    tag_confidences: list[dict[str, Any]] = []

    for weight, row in overlaps:
        sections.update({str(row.get("section") or "loop"): weight})
        subgenres.update({str(row.get("subgenre") or "progressive house"): weight})
        energies.update({str(row.get("energy") or "medium"): weight})
        bpm = as_float(row.get("bpm"), 0.0)
        if 60 <= bpm <= 180:
            bpms.append(bpm)
        quality_scores.append(int(as_float(row.get("quality_score"), 4.0)))
        moods.extend(as_list(row.get("mood")))
        instruments.extend(as_list(row.get("instruments")))
        drums.extend(as_list(row.get("drums")))
        production.extend(as_list(row.get("production")))
        if isinstance(row.get("tag_confidence"), dict):
            tag_confidences.append(row["tag_confidence"])

    tag_confidence: dict[str, float] = {}
    for key in ["subgenre", "mood", "energy", "section", "instruments", "drums", "bass", "production"]:
        vals = [as_float(conf.get(key), 0.6) for conf in tag_confidences if key in conf]
        tag_confidence[key] = round(sum(vals) / len(vals), 3) if vals else 0.62

    return {
        "raw_section": sections.most_common(1)[0][0],
        "raw_energy": energies.most_common(1)[0][0],
        "subgenre": subgenres.most_common(1)[0][0],
        "bpm": median(bpms) if bpms else 128.0,
        "quality_score": max(1, min(5, round(sum(quality_scores) / len(quality_scores)))),
        "mood": unique(moods)[:3],
        "instruments": unique(instruments)[:8],
        "drums": unique(drums)[:4],
        "production": unique(production)[:5],
        "tag_confidence": tag_confidence,
        "audio_features": {key: round(value, 6) for key, value in features.items()},
    }


def unique(values: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def classify_sections(candidates: list[dict[str, Any]], original_duration: float) -> None:
    rms_norm = normalize_values([candidate["audio_features"]["rms"] for candidate in candidates])
    low_norm = normalize_values([candidate["audio_features"]["low_freq_ratio"] for candidate in candidates])
    onset_norm = normalize_values([candidate["audio_features"]["onset_density"] for candidate in candidates])

    for idx, candidate in enumerate(candidates):
        energy_score = 0.52 * rms_norm[idx] + 0.28 * low_norm[idx] + 0.20 * onset_norm[idx]
        prev_energy = rms_norm[max(0, idx - 1)]
        next_energy = rms_norm[min(len(candidates) - 1, idx + 1)]
        rel_pos = candidate["start_time"] / max(1.0, original_duration)
        trend = next_energy - prev_energy

        if rel_pos < 0.10 and energy_score < 0.58:
            section = "intro"
        elif rel_pos > 0.88 and energy_score < 0.62:
            section = "outro"
        elif trend > 0.18 and energy_score >= 0.42 and onset_norm[idx] >= 0.35:
            section = "build-up"
        elif energy_score <= 0.43 and 0.12 <= rel_pos <= 0.86:
            section = "breakdown"
        elif energy_score >= 0.60 and onset_norm[idx] >= 0.30:
            section = "drop"
        elif energy_score >= 0.50:
            section = "drop"
        else:
            section = "loop"

        if energy_score >= 0.78:
            energy = "very_high"
        elif energy_score >= 0.58:
            energy = "high"
        elif energy_score >= 0.35:
            energy = "medium"
        else:
            energy = "low"

        candidate["section"] = section
        candidate["energy"] = energy
        candidate["section_confidence"] = round(
            max(0.52, min(0.92, 0.56 + abs(energy_score - 0.50) * 0.45 + abs(trend) * 0.20)),
            3,
        )
        candidate["energy_score"] = round(energy_score, 5)
        candidate["energy_trend"] = round(trend, 5)


def section_score(candidate: dict[str, Any]) -> float:
    section = candidate["section"]
    energy = candidate["energy_score"]
    trend = candidate["energy_trend"]
    rms = candidate["audio_features"]["rms"]
    onset = candidate["audio_features"]["onset_density"]
    quality = candidate["quality_score"] / 5.0
    if section == "drop":
        return 2.2 * energy + 0.35 * onset + 0.4 * quality
    if section == "breakdown":
        return 1.4 * (1.0 - energy) + 0.25 * quality + 0.1 * rms
    if section == "build-up":
        return 1.3 * max(0.0, trend) + 0.9 * energy + 0.25 * onset + 0.25 * quality
    if section == "intro":
        return 1.0 - min(1.0, candidate["start_time"] / 45.0) + 0.3 * quality
    if section == "outro":
        return candidate["start_time"] / max(1.0, candidate["original_duration"]) + 0.25 * quality
    return 0.7 * energy + 0.25 * quality


def windows_overlap(left: dict[str, Any], right: dict[str, Any], min_gap: float) -> bool:
    return abs(left["start_time"] - right["start_time"]) < min_gap


def select_per_source(candidates: list[dict[str, Any]], max_clips: int, duration: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    quotas = {"drop": 2, "breakdown": 1, "build-up": 1, "intro": 1, "loop": 1, "outro": 1}
    min_gap = duration * 0.65

    for section in SECTION_PRIORITY:
        section_candidates = [
            candidate for candidate in candidates if candidate["section"] == section
        ]
        section_candidates.sort(key=section_score, reverse=True)
        for candidate in section_candidates:
            if len([item for item in selected if item["section"] == section]) >= quotas[section]:
                break
            if any(windows_overlap(candidate, item, min_gap) for item in selected):
                continue
            selected.append(candidate)
            if len(selected) >= max_clips:
                return sorted(selected, key=lambda item: item["start_time"])

    remaining = sorted(candidates, key=section_score, reverse=True)
    for candidate in remaining:
        if len(selected) >= max_clips:
            break
        if candidate in selected:
            continue
        if any(windows_overlap(candidate, item, min_gap) for item in selected):
            continue
        selected.append(candidate)
    return sorted(selected[:max_clips], key=lambda item: item["start_time"])


def global_balance(candidates: list[dict[str, Any]], max_total: int) -> list[dict[str, Any]]:
    if max_total <= 0 or len(candidates) <= max_total:
        return sorted(candidates, key=lambda item: (item["source_id"], item["start_time"]))
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    quotas = {
        section: max(1, int(round(max_total * ratio)))
        for section, ratio in SECTION_RATIOS.items()
    }
    for section in SECTION_PRIORITY:
        pool = [item for item in candidates if item["section"] == section]
        pool.sort(key=section_score, reverse=True)
        for item in pool[: quotas.get(section, 0)]:
            if item["candidate_id"] not in used_ids and len(selected) < max_total:
                selected.append(item)
                used_ids.add(item["candidate_id"])
    if len(selected) < max_total:
        pool = sorted(candidates, key=section_score, reverse=True)
        for item in pool:
            if item["candidate_id"] in used_ids:
                continue
            selected.append(item)
            used_ids.add(item["candidate_id"])
            if len(selected) >= max_total:
                break
    return sorted(selected, key=lambda item: (item["source_id"], item["start_time"]))


def build_candidates_for_source(
    source_id: str,
    source_rows: list[dict[str, Any]],
    duration: float,
    max_clips_per_source: int,
) -> list[dict[str, Any]]:
    first = source_rows[0]
    processed_path = resolve_project_path(first.get("processed_audio_path"))
    if processed_path is None or not processed_path.exists():
        return []

    original_duration = max(as_float(row.get("original_duration"), 0.0) for row in source_rows)
    if original_duration <= duration + 1.0:
        return []

    valid_bpms = [as_float(row.get("bpm"), 0.0) for row in source_rows]
    valid_bpms = [bpm for bpm in valid_bpms if 70 <= bpm <= 160]
    bpm = median(valid_bpms) if valid_bpms else 128.0
    bar_sec = 240.0 / bpm
    hop = max(bar_sec * 4.0, duration * 0.45)
    max_start = max(0.0, original_duration - duration)

    starts: list[float] = []
    current = 0.0
    while current <= max_start:
        starts.append(round(current, 3))
        current += hop
    for row in source_rows:
        row_start = as_float(row.get("start_time"), 0.0)
        aligned = round(max(0.0, min(max_start, round(row_start / bar_sec) * bar_sec)), 3)
        starts.append(aligned)
    starts = sorted(set(starts))

    candidates: list[dict[str, Any]] = []
    for start in starts:
        end = start + duration
        window = aggregate_window(source_rows, start, end)
        candidate_id = f"{source_id}_{int(round(start * 1000)):010d}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source_id": source_id,
                "source_file": first.get("source_file", ""),
                "processed_audio_path": str(processed_path),
                "processed_audio_rel": first.get("processed_audio_path", ""),
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "duration": duration,
                "original_duration": round(original_duration, 3),
                "sample_rate": int(first.get("sample_rate") or 44100),
                "channels": int(first.get("channels") or 2),
                "bpm": round(float(window["bpm"]), 2),
                "bpm_confidence": 0.74,
                "subgenre": window["subgenre"],
                "quality_score": window["quality_score"],
                "mood": window["mood"],
                "instruments": window["instruments"],
                "drums": window["drums"],
                "production": window["production"],
                "tag_confidence": window["tag_confidence"],
                "audio_features": window["audio_features"],
            }
        )

    classify_sections(candidates, original_duration)
    return select_per_source(candidates, max_clips_per_source, duration)


def caption_for(candidate: dict[str, Any]) -> str:
    bpm = int(round(float(candidate["bpm"])))
    subgenre = candidate.get("subgenre") or "progressive house"
    section = candidate["section"]
    energy = candidate["energy"]
    mood = ", ".join(candidate.get("mood") or ["uplifting", "emotional"])
    if section == "drop":
        body = (
            "euphoric progressive house drop with bright piano chord hits, wide supersaw lead hook, "
            "sidechain bass, four-on-the-floor kick, clean clap and festival EDM mix"
        )
    elif section == "breakdown":
        body = (
            "emotional breakdown with warm piano progression, sparse drums, melodic vocal-space arrangement, "
            "acoustic pluck texture and uplifting harmonic tension"
        )
    elif section == "build-up":
        body = (
            "rising build-up with snare roll, filtered chords, melodic lead anticipation, white noise riser "
            "and increasing sidechain movement"
        )
    elif section == "intro":
        body = (
            "clean intro with filtered piano chords, soft pluck motif, light percussion, gradual low-end entry "
            "and polished stereo width"
        )
    elif section == "outro":
        body = (
            "outro with reduced drums, fading piano chords, controlled sidechain pulse and clean festival mix tail"
        )
    else:
        body = (
            "progressive house groove loop with piano chords, pluck synth, sidechain bass, punchy kick and warm mix"
        )
    return (
        f"A {bpm} BPM {subgenre} {section}, {energy} energy, {mood} mood, "
        f"Avicii-inspired arrangement: {body}."
    )


def sample_weight_for(candidate: dict[str, Any]) -> float:
    section = candidate["section"]
    base = {
        "drop": 1.45,
        "breakdown": 1.35,
        "build-up": 1.25,
        "intro": 1.05,
        "loop": 1.00,
        "outro": 0.90,
    }.get(section, 1.0)
    quality = candidate["quality_score"] / 5.0
    confidence = candidate["section_confidence"]
    return round(max(0.55, min(1.75, base * (0.65 + 0.20 * quality + 0.15 * confidence))), 4)


def load_audio_clip(path: Path, start: float, duration: float) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    start_sample = int(round(start * sr))
    clip_samples = int(round(duration * sr))
    end_sample = start_sample + clip_samples
    clip = audio[start_sample:end_sample]
    if clip.shape[0] < clip_samples:
        pad = np.zeros((clip_samples - clip.shape[0], clip.shape[1]), dtype=np.float32)
        clip = np.concatenate([clip, pad], axis=0)
    if clip.shape[1] == 1:
        clip = np.repeat(clip, 2, axis=1)
    if clip.shape[1] > 2:
        clip = clip[:, :2]
    fade_samples = min(int(0.015 * sr), clip.shape[0] // 8)
    if fade_samples > 0:
        fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        clip[:fade_samples] *= fade[:, None]
        clip[-fade_samples:] *= fade[::-1, None]
    peak = float(np.max(np.abs(clip))) if clip.size else 0.0
    if peak > 0.98:
        clip = clip / peak * 0.98
    return np.clip(clip, -1.0, 1.0).astype(np.float32), sr


def materialize_dataset(candidates: list[dict[str, Any]], output_root: Path, force: bool) -> list[dict[str, Any]]:
    clips_dir = output_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        source_slug = safe_slug(candidate["source_id"])
        section = candidate["section"]
        start_s = int(round(candidate["start_time"]))
        end_s = int(round(candidate["end_time"]))
        clip_id = f"{source_slug}_{section.replace('-', '')}_{start_s:04d}s_{end_s:04d}s_15s"
        audio_rel = Path("clips") / f"{clip_id}.wav"
        audio_path = output_root / audio_rel
        if force or not audio_path.exists():
            clip, sr = load_audio_clip(Path(candidate["processed_audio_path"]), candidate["start_time"], candidate["duration"])
            sf.write(str(audio_path), clip, sr, subtype="PCM_16")
        else:
            sr = candidate["sample_rate"]

        row = {
            "clip_id": clip_id,
            "source_id": candidate["source_id"],
            "source_file": candidate.get("source_file", ""),
            "processed_audio_path": candidate.get("processed_audio_rel", ""),
            "audio_path": audio_rel.as_posix(),
            "clip_path": audio_rel.as_posix(),
            "start_time": candidate["start_time"],
            "end_time": candidate["end_time"],
            "duration": round(candidate["duration"], 3),
            "original_duration": candidate["original_duration"],
            "sample_rate": int(sr),
            "channels": 2,
            "genre": "EDM",
            "subgenre": candidate["subgenre"],
            "bpm": candidate["bpm"],
            "bpm_confidence": candidate["bpm_confidence"],
            "key": "unknown",
            "key_confidence": 0.0,
            "mood": candidate["mood"],
            "energy": candidate["energy"],
            "section": candidate["section"],
            "section_confidence": candidate["section_confidence"],
            "energy_score": candidate["energy_score"],
            "energy_trend": candidate["energy_trend"],
            "instruments": candidate["instruments"],
            "drums": candidate["drums"],
            "bass": "sidechain bass",
            "vocal": "instrumental_or_vocal_space",
            "production": unique(candidate["production"] + ["clean master", "wide stereo", "sidechain movement"])[:7],
            "quality_score": candidate["quality_score"],
            "quality": "good",
            "audio_defects": [],
            "split": split_for_source(candidate["source_id"]),
            "caption": caption_for(candidate),
            "trigger_word": STYLE_TRIGGER,
            "sample_weight": sample_weight_for(candidate),
            "tags_source": {
                "segmentation": "15s_bar_aligned_source_metadata_features",
                "section": "energy_trend_position_heuristic",
                "caption": "section_specific_avicii_lora_caption",
            },
            "tag_confidence": candidate["tag_confidence"],
            "audio_features": candidate["audio_features"],
            "processing_notes": [
                "Derived as an independent 15 second section-aware LoRA clip.",
                "Start times are aligned to the estimated four-beat bar grid when source BPM metadata is available.",
                "Original source files are not copied into this dataset root.",
            ],
        }
        rows.append(row)
        if index == 1 or index % 50 == 0 or index == len(candidates):
            print(f"[{index}/{len(candidates)}] wrote {clip_id}", flush=True)
    return rows


def write_report(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    section_counts = Counter(row["section"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    source_counts = Counter(row["source_id"] for row in rows)
    report = {
        "dataset_root": str(output_root),
        "clips": len(rows),
        "sources": len(source_counts),
        "duration_seconds": args.duration,
        "section_counts": dict(section_counts),
        "split_counts": dict(split_counts),
        "max_total_clips": args.max_total_clips,
        "max_clips_per_source": args.max_clips_per_source,
        "artist_keywords": args.artist_keywords,
    }
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# Avicii Section 15s Dataset Report",
        "",
        f"- Clips: {len(rows)}",
        f"- Sources: {len(source_counts)}",
        f"- Duration: {args.duration:g} seconds",
        f"- Section counts: {json.dumps(dict(section_counts), ensure_ascii=False)}",
        f"- Split counts: {json.dumps(dict(split_counts), ensure_ascii=False)}",
        "",
        "## Next Commands",
        "",
        "```powershell",
        f"python scripts\\cache_ace_assets.py --dataset-root \"{output_root}\" --device cpu --batch-size 1",
        f"python scripts\\build_edm_control_assets.py --dataset-root \"{output_root}\"",
        f"python scripts\\prepare_ace_control_dataset.py --dataset-root \"{output_root}\" --output outputs\\datasets\\avicii_section15_lora_train --path-mode project-relative",
        "```",
        "",
    ]
    (reports_dir / "build_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=str(DATASET_ROOT / "metadata.jsonl"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--max-clips-per-source", type=int, default=6)
    parser.add_argument("--max-total-clips", type=int, default=360)
    parser.add_argument("--artist-keywords", default="avicii,avici,tim berg,tim_berg")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    metadata_path = Path(args.metadata)
    output_root = Path(args.output_root)
    keywords = [item.strip().lower() for item in args.artist_keywords.split(",") if item.strip()]

    rows = read_jsonl(metadata_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if source_matches(row, keywords):
            grouped[str(row.get("source_id") or "unknown")].append(row)
    for source_rows in grouped.values():
        source_rows.sort(key=lambda item: as_float(item.get("start_time"), 0.0))

    print(f"Loaded rows: {len(rows)}", flush=True)
    print(f"Matched sources: {len(grouped)}", flush=True)

    candidates: list[dict[str, Any]] = []
    for index, (source_id, source_rows) in enumerate(sorted(grouped.items()), start=1):
        source_candidates = build_candidates_for_source(
            source_id=source_id,
            source_rows=source_rows,
            duration=args.duration,
            max_clips_per_source=max(1, args.max_clips_per_source),
        )
        candidates.extend(source_candidates)
        if index == 1 or index % 25 == 0 or index == len(grouped):
            print(
                f"[{index}/{len(grouped)}] candidates={len(candidates)} latest_source={source_id}",
                flush=True,
            )

    selected = global_balance(candidates, args.max_total_clips)
    print(f"Selected clips: {len(selected)}", flush=True)
    print(f"Selected sections: {dict(Counter(item['section'] for item in selected))}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    materialized = materialize_dataset(selected, output_root, force=args.force)
    write_jsonl(output_root / "metadata.jsonl", materialized)
    write_csv(output_root / "metadata.csv", materialized)
    for split in ["train", "val", "test"]:
        write_jsonl(
            output_root / "splits" / f"{split}.jsonl",
            [row for row in materialized if row.get("split") == split],
        )
    write_report(output_root, materialized, args)
    print(json.dumps({
        "dataset_root": str(output_root),
        "clips": len(materialized),
        "sections": dict(Counter(row["section"] for row in materialized)),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
