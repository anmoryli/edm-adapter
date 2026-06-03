"""Generate report assets: charts, statistics, examples."""

import argparse
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_genre_distribution(metadata_path: str, output_path: str):
    """Plot genre distribution pie chart."""
    df = pd.read_csv(metadata_path)
    genre_counts = df["genre"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    genre_counts.plot(kind="bar", ax=ax, color=plt.cm.Set3(np.linspace(0, 1, len(genre_counts))))
    ax.set_title("Genre Distribution", fontsize=14)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_bpm_distribution(metadata_path: str, output_path: str):
    """Plot BPM distribution histogram."""
    df = pd.read_csv(metadata_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    df["bpm"].hist(bins=30, ax=ax, edgecolor="black", alpha=0.7)
    ax.set_title("BPM Distribution", fontsize=14)
    ax.set_xlabel("BPM")
    ax.set_ylabel("Count")
    ax.axvline(df["bpm"].mean(), color="red", linestyle="--", label=f"Mean: {df['bpm'].mean():.0f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_comparison_metrics(metrics_path: str, output_path: str):
    """Plot baseline vs fine-tuned comparison."""
    df = pd.read_csv(metrics_path)

    metrics_to_compare = ["bpm_error", "rms", "low_freq_ratio", "onset_density", "loop_similarity"]
    available = [m for m in metrics_to_compare if m in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            ax.bar(model, model_df[metric].mean(), yerr=model_df[metric].std(),
                   capsize=5, alpha=0.8, label=model)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Value")

    plt.suptitle("Baseline vs Fine-tuned Comparison", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_caption_examples(metadata_path: str, output_path: str):
    """Generate caption examples markdown."""
    df = pd.read_csv(metadata_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Caption Examples\n\n")
        f.write("Examples of auto-generated captions for training data.\n\n")

        for genre in df["genre"].unique():
            genre_df = df[df["genre"] == genre]
            f.write(f"## {genre.title()}\n\n")
            for _, row in genre_df.head(3).iterrows():
                f.write(f"- **{row['caption']}**\n")
                f.write(f"  - BPM: {row['bpm']}, RMS: {row['rms_mean']:.4f}, Low Freq: {row['low_freq_ratio']:.4f}\n")
                f.write(f"  - File: `{row['audio_path']}`\n\n")

    print(f"Saved: {output_path}")


def generate_dataset_stats(metadata_path: str, output_dir: str):
    """Generate dataset statistics CSV."""
    df = pd.read_csv(metadata_path)

    stats = {
        "total_clips": len(df),
        "unique_genres": df["genre"].nunique(),
        "mean_bpm": df["bpm"].mean(),
        "std_bpm": df["bpm"].std(),
        "mean_rms": df["rms_mean"].mean(),
        "mean_duration": df["duration"].mean(),
    }

    stats_path = os.path.join(output_dir, "dataset_stats.csv")
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")

    # Genre breakdown
    genre_stats = df.groupby("genre").agg({
        "bpm": ["count", "mean", "std"],
        "rms_mean": "mean",
        "low_freq_ratio": "mean",
    }).round(2)

    genre_path = os.path.join(output_dir, "genre_stats.csv")
    genre_stats.to_csv(genre_path)
    print(f"Saved: {genre_path}")


def plot_baseline_bpm_by_prompt(metrics_path: str, output_path: str):
    """Plot BPM accuracy per prompt for baseline."""
    df = pd.read_csv(metrics_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    prompts = df["prompt_id"].unique()
    x = np.arange(len(prompts))
    width = 0.35

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        means = [model_df[model_df["prompt_id"] == p]["bpm_error"].mean() for p in prompts]
        ax.bar(x + (0 if model == "baseline" else width), means, width, label=model, alpha=0.8)

    ax.set_xlabel("Prompt")
    ax.set_ylabel("BPM Error")
    ax.set_title("BPM Error by Prompt")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(prompts, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_baseline_summary(metrics_path: str, output_path: str):
    """Plot a summary dashboard for baseline metrics."""
    df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    metrics = [
        ("bpm_error", "BPM Error"),
        ("rms", "RMS Energy"),
        ("low_freq_ratio", "Low Freq Ratio"),
        ("onset_density", "Onset Density"),
        ("loop_similarity", "Loop Similarity"),
        ("silence_ratio", "Silence Ratio"),
    ]

    for ax, (col, title) in zip(axes.flat, metrics):
        if col in df.columns:
            for model in df["model"].unique():
                model_df = df[df["model"] == model]
                ax.bar(model, model_df[col].mean(), yerr=model_df[col].std(),
                       capsize=3, alpha=0.8)
            ax.set_title(title)
            ax.set_ylabel(col)

    plt.suptitle("Baseline Metrics Summary", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate report assets")
    parser.add_argument("--metadata", default="data/processed/metadata.csv")
    parser.add_argument("--metrics", default="outputs/comparisons/metrics.csv")
    parser.add_argument("--output-dir", default="data/reports")
    args = parser.parse_args()

    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.metadata):
        print("\nGenerating genre distribution...")
        plot_genre_distribution(args.metadata, os.path.join(args.output_dir, "genre_distribution.png"))

        print("\nGenerating BPM distribution...")
        plot_bpm_distribution(args.metadata, os.path.join(args.output_dir, "bpm_distribution.png"))

        print("\nGenerating caption examples...")
        generate_caption_examples(args.metadata, os.path.join(args.output_dir, "caption_examples.md"))

        print("\nGenerating dataset stats...")
        generate_dataset_stats(args.metadata, args.output_dir)
    else:
        print(f"Metadata file not found: {args.metadata}")

    if os.path.exists(args.metrics):
        print("\nGenerating comparison charts...")
        plot_comparison_metrics(args.metrics, os.path.join(args.output_dir, "baseline_vs_finetuned_metrics.png"))

        print("\nGenerating baseline BPM by prompt...")
        plot_baseline_bpm_by_prompt(args.metrics, os.path.join(args.output_dir, "baseline_bpm_by_prompt.png"))

        print("\nGenerating baseline summary dashboard...")
        plot_baseline_summary(args.metrics, os.path.join(args.output_dir, "baseline_summary.png"))
    else:
        print(f"Metrics file not found: {args.metrics}")

    print("\nReport assets generation complete!")


if __name__ == "__main__":
    main()
