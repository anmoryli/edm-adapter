"""Quality-based audio sampler for training data optimization.

Analyzes audio quality and creates weighted copies for training.
Higher-quality samples get more copies -> sampled more often during training.
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def analyze_audio_quality(audio_path: str) -> float:
    """Analyze audio quality and return a score from 0.0 to 1.0.

    Higher score = better quality for training.
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=48000, mono=True, duration=30.0)
    except Exception:
        return 0.5  # Default score if analysis fails

    if len(y) < sr:  # Less than 1 second
        return 0.1

    score = 0.0

    # 1. RMS energy — not too quiet, not clipped
    rms = np.sqrt(np.mean(y**2))
    if 0.02 < rms < 0.3:
        score += 0.25
    elif rms > 0.01:
        score += 0.1

    # 2. Silence ratio — less silence is better
    silence_ratio = np.mean(np.abs(y) < 0.01)
    if silence_ratio < 0.1:
        score += 0.25
    elif silence_ratio < 0.3:
        score += 0.15
    elif silence_ratio < 0.5:
        score += 0.05

    # 3. Spectral richness — higher centroid = more harmonic content
    try:
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        if 1000 < centroid < 5000:
            score += 0.25
        elif centroid > 500:
            score += 0.1
    except Exception:
        score += 0.1

    # 4. Onset density — rhythmic activity
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_density = np.mean(onset_env > np.mean(onset_env))
        if onset_density > 0.3:
            score += 0.25
        elif onset_density > 0.15:
            score += 0.15
        elif onset_density > 0.05:
            score += 0.05
    except Exception:
        score += 0.1

    return min(1.0, score)


def get_repeat_count(quality_score: float, base_repeat: int = 1) -> int:
    """Convert quality score to repeat count.

    Score 0.0-0.3: 1x (base)
    Score 0.3-0.5: 2x
    Score 0.5-0.7: 3x
    Score 0.7-0.9: 4x
    Score 0.9-1.0: 5x
    """
    if quality_score < 0.3:
        return base_repeat
    elif quality_score < 0.5:
        return base_repeat * 2
    elif quality_score < 0.7:
        return base_repeat * 3
    elif quality_score < 0.9:
        return base_repeat * 4
    else:
        return base_repeat * 5


def analyze_training_data(training_dir: str) -> Dict[str, float]:
    """Analyze quality of all audio files in training directory.

    Returns dict mapping filename -> quality_score.
    """
    audio_files = list(Path(training_dir).glob("*.mp3")) + \
                  list(Path(training_dir).glob("*.wav")) + \
                  list(Path(training_dir).glob("*.flac"))

    print(f"Analyzing quality of {len(audio_files)} audio files...")

    scores = {}
    for i, audio_path in enumerate(audio_files):
        score = analyze_audio_quality(str(audio_path))
        scores[audio_path.name] = score

        if (i + 1) % 10 == 0 or i == len(audio_files) - 1:
            print(f"  [{i+1}/{len(audio_files)}] {audio_path.name}: {score:.2f}")

    return scores


def create_quality_weighted_copies(
    training_dir: str,
    scores: Dict[str, float],
    base_repeat: int = 1,
) -> int:
    """Create additional copies of high-quality samples.

    Returns total number of new files created.
    """
    total_new = 0

    for filename, score in scores.items():
        repeat = get_repeat_count(score, base_repeat)
        if repeat <= 1:
            continue

        audio_path = os.path.join(training_dir, filename)
        stem = Path(filename).stem
        ext = Path(filename).suffix

        # Check for associated prompt/lyrics files
        prompt_path = os.path.join(training_dir, f"{stem}_prompt.txt")
        lyrics_path = os.path.join(training_dir, f"{stem}_lyrics.txt")

        for copy_idx in range(1, repeat):
            copy_name = f"{stem}_q{copy_idx}{ext}"
            copy_audio = os.path.join(training_dir, copy_name)
            copy_prompt = os.path.join(training_dir, f"{stem}_q{copy_idx}_prompt.txt")
            copy_lyrics = os.path.join(training_dir, f"{stem}_q{copy_idx}_lyrics.txt")

            if os.path.exists(copy_audio):
                continue  # Already exists

            shutil.copy2(audio_path, copy_audio)
            if os.path.exists(prompt_path):
                shutil.copy2(prompt_path, copy_prompt)
            if os.path.exists(lyrics_path):
                shutil.copy2(lyrics_path, copy_lyrics)

            total_new += 1

    return total_new


def run_quality_sampling(
    training_dir: str,
    output_report: str = None,
    base_repeat: int = 1,
) -> Dict:
    """Full quality sampling pipeline.

    1. Analyze all audio quality
    2. Create weighted copies
    3. Generate report
    """
    print("=" * 60)
    print("Quality-Weighted Sampling")
    print("=" * 60)

    # Step 1: Analyze quality
    scores = analyze_audio_quality(training_dir)

    if not scores:
        print("No audio files found!")
        return {}

    # Statistics
    score_values = list(scores.values())
    print(f"\nQuality Statistics:")
    print(f"  Mean:   {np.mean(score_values):.2f}")
    print(f"  Median: {np.median(score_values):.2f}")
    print(f"  Min:    {np.min(score_values):.2f}")
    print(f"  Max:    {np.max(score_values):.2f}")

    # Distribution
    brackets = [(0, 0.3, "Low"), (0.3, 0.5, "Medium-Low"),
                (0.5, 0.7, "Medium"), (0.7, 0.9, "High"), (0.9, 1.0, "Excellent")]
    print(f"\n  Distribution:")
    for lo, hi, label in brackets:
        count = sum(1 for s in score_values if lo <= s < hi)
        pct = count / len(score_values) * 100
        print(f"    {label:15s} ({lo:.1f}-{hi:.1f}): {count:3d} ({pct:.0f}%)")

    # Step 2: Create weighted copies
    print(f"\nCreating quality-weighted copies (base_repeat={base_repeat})...")
    new_files = create_quality_weighted_copies(training_dir, scores, base_repeat)
    print(f"Created {new_files} new sample copies")

    # Step 3: Report
    total_samples = len(scores) + new_files
    print(f"\nTotal training samples: {total_samples} (was {len(scores)})")

    report = {
        "original_count": len(scores),
        "copies_created": new_files,
        "total_samples": total_samples,
        "quality_stats": {
            "mean": float(np.mean(score_values)),
            "median": float(np.median(score_values)),
            "min": float(np.min(score_values)),
            "max": float(np.max(score_values)),
        },
        "scores": scores,
    }

    if output_report:
        os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
        with open(output_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {output_report}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quality-based audio sampler")
    parser.add_argument("training_dir", help="Training data directory")
    parser.add_argument("--base_repeat", type=int, default=1, help="Base repeat count")
    parser.add_argument("--report", type=str, default=None, help="Save report to file")

    args = parser.parse_args()
    run_quality_sampling(args.training_dir, args.report, args.base_repeat)
