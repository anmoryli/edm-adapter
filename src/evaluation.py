"""Evaluation metrics for generated audio."""

import os
import csv
import numpy as np
import pandas as pd

from src.audio_io import load_audio, check_clipping, check_silence
from src.audio_features import extract_all_features, compute_loop_similarity


def evaluate_single(audio_path: str, target_bpm: float, sr: int = 44100) -> dict:
    """Evaluate a single generated audio file.

    Returns dict with all metrics.
    """
    y, sr = load_audio(audio_path, sr=sr, mono=True)
    features = extract_all_features(y, sr)

    metrics = {
        "audio_path": audio_path,
        "target_bpm": target_bpm,
        "estimated_bpm": features["bpm"],
        "bpm_error": abs(features["bpm"] - target_bpm),
        "rms": features["rms_mean"],
        "low_freq_ratio": features["low_freq_ratio"],
        "onset_density": features["onset_density"],
        "spectral_centroid": features["spectral_centroid_mean"],
        "zcr": features["zcr_mean"],
        "silence_ratio": check_silence(y),
        "clipping_ratio": check_clipping(y),
        "loop_similarity": compute_loop_similarity(y, sr),
    }
    return metrics


def evaluate_batch(
    audio_files: list[str],
    prompts: list[dict],
    model_label: str,
    sr: int = 44100,
) -> list[dict]:
    """Evaluate a batch of generated audio files.

    Args:
        audio_files: List of audio file paths.
        prompts: List of prompt dicts with 'id', 'target_bpm', 'genre'.
        model_label: Label for this model (e.g., 'baseline' or 'finetuned').
        sr: Sample rate.

    Returns:
        List of metric dicts.
    """
    # Build mapping from prompt_id to target_bpm
    prompt_map = {p["id"]: p for p in prompts}

    results = []
    for fpath in audio_files:
        # Extract prompt_id from filename
        basename = os.path.splitext(os.path.basename(fpath))[0]
        # e.g., "techno_dark_seed_042" -> "techno_dark"
        parts = basename.rsplit("_seed_", 1)
        prompt_id = parts[0] if parts else basename
        seed = int(parts[1]) if len(parts) > 1 else 0

        prompt_info = prompt_map.get(prompt_id, {})
        target_bpm = prompt_info.get("target_bpm", 120)
        genre = prompt_info.get("genre", "unknown")

        metrics = evaluate_single(fpath, target_bpm, sr=sr)
        metrics["model"] = model_label
        metrics["prompt_id"] = prompt_id
        metrics["genre"] = genre
        metrics["seed"] = seed

        results.append(metrics)

    return results


def save_metrics_csv(results: list[dict], output_path: str):
    """Save evaluation results to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def compute_summary_stats(results: list[dict]) -> dict:
    """Compute summary statistics from evaluation results."""
    df = pd.DataFrame(results)

    stats = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        stats[model] = {
            "count": len(model_df),
            "mean_bpm_error": float(model_df["bpm_error"].mean()),
            "median_bpm_error": float(model_df["bpm_error"].median()),
            "mean_rms": float(model_df["rms"].mean()),
            "mean_low_freq_ratio": float(model_df["low_freq_ratio"].mean()),
            "mean_onset_density": float(model_df["onset_density"].mean()),
            "mean_silence_ratio": float(model_df["silence_ratio"].mean()),
            "mean_clipping_ratio": float(model_df["clipping_ratio"].mean()),
            "mean_loop_similarity": float(model_df["loop_similarity"].mean()),
        }

    return stats
