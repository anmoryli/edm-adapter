"""Build structured captions for each analyzed audio clip."""

import argparse
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logging
from src.caption_builder import build_caption


def main():
    parser = argparse.ArgumentParser(description="Build captions for audio clips")
    parser.add_argument("--input", default="data/processed/analyzed_metadata.csv")
    parser.add_argument("--output-csv", default="data/processed/metadata.csv")
    parser.add_argument("--output-jsonl", default="data/processed/metadata.jsonl")
    args = parser.parse_args()

    setup_logging()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        print("Run 04_analyze_audio.py first.")
        return

    # Load analyzed metadata
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    print(f"Building captions for {len(records)} clips...")

    captioned = []
    for rec in records:
        bpm = float(rec.get("bpm", 120))
        rms_mean = float(rec.get("rms_mean", 0.05))
        low_freq_ratio = float(rec.get("low_freq_ratio", 0.3))
        genre = rec.get("genre", "electronic")

        caption = build_caption(
            bpm=bpm,
            rms_mean=rms_mean,
            low_freq_ratio=low_freq_ratio,
            genre=genre,
        )

        record = {
            "audio_path": rec["audio_path"],
            "caption": caption,
            "genre": genre,
            "bpm": round(bpm),
            "rms_mean": round(rms_mean, 4),
            "low_freq_ratio": round(low_freq_ratio, 4),
            "spectral_centroid_mean": round(float(rec.get("spectral_centroid_mean", 0)), 2),
            "onset_density": round(float(rec.get("onset_density", 0)), 4),
            "duration": float(rec.get("duration", 10.0)),
            "track_id": rec.get("track_id", ""),
            "segment_id": rec.get("segment_id", 0),
        }
        captioned.append(record)

    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=captioned[0].keys())
        writer.writeheader()
        writer.writerows(captioned)

    # Save JSONL
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for rec in captioned:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(captioned)} captioned records:")
    print(f"  CSV:   {args.output_csv}")
    print(f"  JSONL: {args.output_jsonl}")

    # Print some examples
    print("\nExample captions:")
    for rec in captioned[:5]:
        print(f"  [{rec['genre']}] {rec['caption']}")


if __name__ == "__main__":
    main()
