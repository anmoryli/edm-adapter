"""Segment audio files into fixed-length clips for training.

If no real audio files exist, generates synthetic test audio for development.
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, setup_logging
from src.audio_io import load_audio, save_audio, normalize_audio, convert_to_standard


def segment_audio(y: np.ndarray, sr: int, clip_duration: float, hop_duration: float,
                  skip_start: float, skip_end: float, max_clips: int) -> list[np.ndarray]:
    """Segment audio into fixed-length clips."""
    total_duration = y.shape[-1] / sr
    start_sample = int(skip_start * sr)
    end_sample = int((total_duration - skip_end) * sr)

    if end_sample <= start_sample:
        return []

    clip_samples = int(clip_duration * sr)
    hop_samples = int(hop_duration * sr)
    clips = []

    pos = start_sample
    while pos + clip_samples <= end_sample and len(clips) < max_clips:
        clip = y[..., pos:pos + clip_samples]
        clips.append(clip)
        pos += hop_samples

    return clips


def generate_synthetic_audio(duration: float, sr: int, genre: str, bpm: float, seed: int) -> np.ndarray:
    """Generate synthetic audio for testing when no real audio is available."""
    np.random.seed(seed)
    n_samples = int(duration * sr)

    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Base frequency from BPM
    beat_freq = bpm / 60.0

    # Kick drum (low sine burst)
    kick = np.zeros(n_samples)
    beat_interval = int(sr / beat_freq)
    for i in range(0, n_samples, beat_interval):
        env_len = min(int(0.1 * sr), n_samples - i)
        env = np.exp(-np.linspace(0, 5, env_len))
        kick[i:i + env_len] += env * np.sin(2 * np.pi * 55 * np.arange(env_len) / sr)

    # Bass
    bass_freq = {"techno": 55, "house": 55, "ambient": 40, "trap": 45, "drum and bass": 50}.get(genre, 50)
    bass = 0.3 * np.sin(2 * np.pi * bass_freq * t)

    # Hi-hat (noise bursts)
    hihat = np.zeros(n_samples)
    hh_interval = beat_interval // 2
    for i in range(0, n_samples, hh_interval):
        env_len = min(int(0.02 * sr), n_samples - i)
        env = np.exp(-np.linspace(0, 10, env_len))
        hihat[i:i + env_len] += env * np.random.randn(env_len) * 0.2

    # Synth pad
    pad_freqs = [220, 277, 330, 440]
    pad = sum(0.1 * np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi) for f in pad_freqs)
    pad *= 0.3

    # Mix
    y = kick * 0.5 + bass + hihat + pad * 0.2

    # Add stereo
    y_stereo = np.stack([y, y * 0.95 + np.random.randn(n_samples) * 0.01], axis=0)

    # Normalize
    y_stereo = y_stereo / (np.max(np.abs(y_stereo)) + 1e-8) * 0.8

    return y_stereo.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Segment audio into clips")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--input", default="data/interim/electronic_metadata.csv")
    parser.add_argument("--output-dir", default="data/processed/clips")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic audio for testing")
    parser.add_argument("--num-synthetic", type=int, default=100,
                        help="Number of synthetic clips to generate")
    args = parser.parse_args()

    setup_logging()
    config = load_yaml(args.config)
    seg_config = config["segmentation"]
    sr = 44100

    os.makedirs(args.output_dir, exist_ok=True)
    clip_records = []

    if args.synthetic or not os.path.exists(args.input):
        print("Generating synthetic audio clips for development/testing...")
        genres = ["techno", "house", "ambient", "trap", "drum and bass", "trance", "dubstep"]
        bpms = {
            "techno": 128, "house": 124, "ambient": 85, "trap": 140,
            "drum and bass": 170, "trance": 138, "dubstep": 140,
        }

        for i in range(args.num_synthetic):
            genre = genres[i % len(genres)]
            bpm = bpms[genre]
            duration = seg_config["clip_duration"]

            y = generate_synthetic_audio(duration, sr, genre, bpm, seed=i)
            y = normalize_audio(y, peak_db=-1.0)

            fname = f"track_{i:06d}_seg_000_{genre}_{int(bpm)}.wav"
            out_path = os.path.join(args.output_dir, fname)
            save_audio(out_path, y, sr=sr)

            clip_records.append({
                "audio_path": out_path,
                "track_id": f"track_{i:06d}",
                "segment_id": 0,
                "genre": genre,
                "bpm": bpm,
                "duration": duration,
                "source": "synthetic",
            })

            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{args.num_synthetic} clips")

    else:
        # Process real audio files
        import csv
        with open(args.input, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            tracks = list(reader)

        print(f"Processing {len(tracks)} tracks...")

        for idx, track in enumerate(tracks):
            file_path = track.get("file_path", "")
            if not file_path or not os.path.exists(file_path):
                continue

            try:
                y, _ = load_audio(file_path, sr=sr)
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                continue

            tags = track.get("tags", "")
            genre = "electronic"
            for g in ["techno", "house", "ambient", "trap", "drum and bass", "trance", "dubstep"]:
                if g in tags.lower():
                    genre = g
                    break

            clips = segment_audio(
                y, sr,
                clip_duration=seg_config["clip_duration"],
                hop_duration=seg_config.get("hop_duration", seg_config["clip_duration"]),
                skip_start=seg_config["skip_start"],
                skip_end=seg_config["skip_end"],
                max_clips=seg_config["max_clips_per_track"],
            )

            for seg_id, clip in enumerate(clips):
                fname = f"{track['track_id']}_seg_{seg_id:03d}_{genre}.wav"
                out_path = os.path.join(args.output_dir, fname)
                clip = normalize_audio(clip, peak_db=-1.0)
                save_audio(out_path, clip, sr=sr)

                clip_records.append({
                    "audio_path": out_path,
                    "track_id": track["track_id"],
                    "segment_id": seg_id,
                    "genre": genre,
                    "duration": seg_config["clip_duration"],
                    "source": track.get("source", "unknown"),
                })

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(tracks)} tracks, {len(clip_records)} clips so far")

    # Save clip metadata
    metadata_path = os.path.join("data/processed", "clips_metadata.csv")
    import csv
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        if clip_records:
            writer = csv.DictWriter(f, fieldnames=clip_records[0].keys())
            writer.writeheader()
            writer.writerows(clip_records)

    print(f"\nTotal clips generated: {len(clip_records)}")
    print(f"Clip metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
