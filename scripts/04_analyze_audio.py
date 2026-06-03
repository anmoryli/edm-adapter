"""Analyze audio clips: extract BPM, energy, spectral features."""

import argparse
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, setup_logging
from src.audio_io import load_audio, check_clipping, check_silence
from src.audio_features import extract_all_features


def main():
    parser = argparse.ArgumentParser(description="Analyze audio features")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--input", default="data/processed/clips_metadata.csv")
    parser.add_argument("--clips-dir", default="data/processed/clips")
    parser.add_argument("--output", default="data/processed/analyzed_metadata.csv")
    args = parser.parse_args()

    setup_logging()
    config = load_yaml(args.config)
    quality = config["quality_filters"]

    # Load clip metadata
    if not os.path.exists(args.input):
        # Try to build from clips directory
        if os.path.exists(args.clips_dir):
            clip_files = [f for f in os.listdir(args.clips_dir) if f.endswith(".wav")]
            clips = [{"audio_path": os.path.join(args.clips_dir, f), "track_id": f.split("_seg_")[0] if "_seg_" in f else f}
                     for f in clip_files]
        else:
            print(f"No input file or clips directory found.")
            return
    else:
        clips = []
        with open(args.input, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            clips = list(reader)

    print(f"Analyzing {len(clips)} clips...")
    analyzed = []
    skipped = 0

    for idx, clip in enumerate(clips):
        audio_path = clip.get("audio_path", "")
        if not os.path.exists(audio_path):
            skipped += 1
            continue

        try:
            y, sr = load_audio(audio_path, sr=44100, mono=True)

            # Quality checks
            rms = float(__import__("librosa").feature.rms(y=y)[0].mean())
            silence_ratio = check_silence(y)
            clipping_ratio = check_clipping(y)

            if rms < quality["min_rms"]:
                skipped += 1
                continue
            if silence_ratio > quality["max_silence_ratio"]:
                skipped += 1
                continue
            if clipping_ratio > quality["max_clipping_ratio"]:
                skipped += 1
                continue

            # Extract features
            features = extract_all_features(y, sr)

            record = {
                "audio_path": audio_path,
                "track_id": clip.get("track_id", ""),
                "segment_id": clip.get("segment_id", 0),
                "genre": clip.get("genre", "electronic"),
                "duration": float(clip.get("duration", len(y) / sr)),
                "bpm": features["bpm"],
                "rms_mean": features["rms_mean"],
                "rms_std": features["rms_std"],
                "low_freq_ratio": features["low_freq_ratio"],
                "spectral_centroid_mean": features["spectral_centroid_mean"],
                "spectral_bandwidth_mean": features["spectral_bandwidth_mean"],
                "zcr_mean": features["zcr_mean"],
                "onset_density": features["onset_density"],
                "spectral_rolloff_mean": features["spectral_rolloff_mean"],
                "silence_ratio": silence_ratio,
                "clipping_ratio": clipping_ratio,
                "source": clip.get("source", "unknown"),
            }
            analyzed.append(record)

        except Exception as e:
            print(f"  Error processing {audio_path}: {e}")
            skipped += 1

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(clips)}, {len(analyzed)} passed quality filter")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if analyzed:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=analyzed[0].keys())
            writer.writeheader()
            writer.writerows(analyzed)

    print(f"\nAnalysis complete:")
    print(f"  Total clips: {len(clips)}")
    print(f"  Passed quality filter: {len(analyzed)}")
    print(f"  Skipped: {skipped}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
