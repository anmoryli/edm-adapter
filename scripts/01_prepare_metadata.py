"""Prepare metadata from MTG-Jamendo and/or FMA datasets.

Reads raw dataset metadata and produces a unified CSV with columns:
  track_id, file_path, duration, tags (comma-separated)
"""

import argparse
import os
import sys
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, setup_logging


def prepare_mtg_jamendo(config: dict) -> list[dict]:
    """Parse MTG-Jamendo metadata.

    Expected files:
      - raw_30s_cleantags.tsv (or similar) with columns: TRACK_ID, ARTIST, ALBUM, TRACK, PATH, DURATION, TAGS
    """
    logger = setup_logging()
    data_dir = config["sources"]["mtg_jamendo"]["path"]

    # Look for TSV metadata files
    metadata_files = list(Path(data_dir).glob("*.tsv"))
    if not metadata_files:
        # Also try CSV
        metadata_files = list(Path(data_dir).glob("*.csv"))

    if not metadata_files:
        print(f"No metadata files found in {data_dir}")
        print("Please download MTG-Jamendo dataset metadata to this directory.")
        print("Expected: raw_30s_cleantags.tsv or similar")
        return []

    records = []
    for meta_file in metadata_files:
        print(f"Reading: {meta_file}")
        delimiter = "\t" if meta_file.suffix == ".tsv" else ","

        with open(meta_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                track_id = row.get("TRACK_ID") or row.get("track_id") or row.get("track id", "")
                duration = row.get("DURATION") or row.get("duration", "0")
                tags = row.get("TAGS") or row.get("tags", "")
                path = row.get("PATH") or row.get("path", "")

                if not track_id:
                    continue

                records.append({
                    "track_id": str(track_id),
                    "source": "mtg_jamendo",
                    "file_path": path,
                    "duration": float(duration) if duration else 0,
                    "tags": tags,
                    "artist": row.get("ARTIST", ""),
                    "album": row.get("ALBUM", ""),
                    "title": row.get("TRACK", ""),
                })

    print(f"Loaded {len(records)} tracks from MTG-Jamendo")
    return records


def prepare_fma(config: dict) -> list[dict]:
    """Parse FMA metadata.

    Expected: raw_tracks.csv with columns: track_id, title, genre, ...
    """
    data_dir = config["sources"]["fma"]["path"]

    metadata_files = list(Path(data_dir).glob("*.csv"))
    if not metadata_files:
        print(f"No metadata files found in {data_dir}")
        print("Please download FMA dataset metadata to this directory.")
        return []

    records = []
    for meta_file in metadata_files:
        print(f"Reading: {meta_file}")
        with open(meta_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                track_id = row.get("track_id", "")
                if not track_id:
                    continue

                # FMA genres are in a specific column
                genre = row.get("genre") or row.get("genres") or row.get("track_genre_top", "")
                title = row.get("title", "")

                records.append({
                    "track_id": str(track_id),
                    "source": "fma",
                    "file_path": row.get("file_path", ""),
                    "duration": float(row.get("duration", 0)),
                    "tags": genre,
                    "artist": row.get("artist", ""),
                    "album": row.get("album", ""),
                    "title": title,
                })

    print(f"Loaded {len(records)} tracks from FMA")
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare metadata from datasets")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--output", default="data/interim/raw_metadata.csv")
    args = parser.parse_args()

    config = load_yaml(args.config)
    all_records = []

    if config["sources"]["mtg_jamendo"]["enabled"]:
        records = prepare_mtg_jamendo(config)
        all_records.extend(records)

    if config["sources"]["fma"]["enabled"]:
        records = prepare_fma(config)
        all_records.extend(records)

    if not all_records:
        print("\nNo records loaded. Creating sample metadata for testing...")
        all_records = _create_sample_metadata()

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        if all_records:
            writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
            writer.writeheader()
            writer.writerows(all_records)

    print(f"\nSaved {len(all_records)} records to {args.output}")


def _create_sample_metadata() -> list[dict]:
    """Create sample metadata entries for testing without real datasets."""
    genres = ["techno", "house", "ambient", "trap", "drum and bass", "trance", "dubstep"]
    records = []
    for i in range(20):
        genre = genres[i % len(genres)]
        records.append({
            "track_id": f"sample_{i:04d}",
            "source": "sample",
            "file_path": "",
            "duration": 180.0,
            "tags": f"electronic,{genre}",
            "artist": f"artist_{i}",
            "album": f"album_{i}",
            "title": f"track_{i}",
        })
    return records


if __name__ == "__main__":
    main()
