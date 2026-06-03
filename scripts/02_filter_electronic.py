"""Filter electronic music tracks from raw metadata."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, setup_logging
from src.dataset_builder import load_metadata_csv, save_metadata_csv, filter_electronic_tracks


def main():
    parser = argparse.ArgumentParser(description="Filter electronic music tracks")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--input", default="data/interim/raw_metadata.csv")
    parser.add_argument("--output", default="data/interim/electronic_metadata.csv")
    args = parser.parse_args()

    setup_logging()
    config = load_yaml(args.config)

    # Load raw metadata
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        print("Run 01_prepare_metadata.py first.")
        return

    df = load_metadata_csv(args.input)
    print(f"Total tracks: {len(df)}")

    # Get tags from config
    electronic_tags = []
    for category in config["electronic_tags"].values():
        electronic_tags.extend(category)

    exclude_tags = config.get("exclude_tags", [])

    # Filter
    filtered = filter_electronic_tracks(df, electronic_tags, exclude_tags, tag_columns=["tags"])
    print(f"Electronic tracks after filtering: {len(filtered)}")

    # Genre distribution
    genre_counts = {}
    for _, row in filtered.iterrows():
        tags = str(row.get("tags", "")).lower()
        for genre in ["techno", "house", "ambient", "trap", "drum and bass", "dnb", "trance", "dubstep", "chillout"]:
            if genre in tags:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                break
        else:
            genre_counts["electronic_other"] = genre_counts.get("electronic_other", 0) + 1

    print("\nGenre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"  {genre:20s} {count}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_metadata_csv(filtered.to_dict("records"), args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
