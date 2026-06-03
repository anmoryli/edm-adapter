"""Evaluate baseline vs fine-tuned audio and generate comparison metrics."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, setup_logging
from src.evaluation import evaluate_batch, save_metrics_csv, compute_summary_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare baseline vs fine-tuned")
    parser.add_argument("--prompts-config", default="configs/prompts.yaml")
    parser.add_argument("--baseline-dir", default="outputs/baseline")
    parser.add_argument("--finetuned-dir", default="outputs/finetuned")
    parser.add_argument("--output", default="outputs/comparisons/metrics.csv")
    args = parser.parse_args()

    setup_logging()
    config = load_yaml(args.prompts_config)
    prompts = config["prompts"]

    all_results = []

    # Evaluate baseline
    if os.path.exists(args.baseline_dir):
        baseline_files = [
            os.path.join(args.baseline_dir, f)
            for f in os.listdir(args.baseline_dir) if f.endswith(".wav")
        ]
        if baseline_files:
            print(f"Evaluating {len(baseline_files)} baseline files...")
            baseline_results = evaluate_batch(baseline_files, prompts, "baseline")
            all_results.extend(baseline_results)
    else:
        print(f"Baseline directory not found: {args.baseline_dir}")

    # Evaluate fine-tuned
    if os.path.exists(args.finetuned_dir):
        finetuned_files = [
            os.path.join(args.finetuned_dir, f)
            for f in os.listdir(args.finetuned_dir) if f.endswith(".wav")
        ]
        if finetuned_files:
            print(f"Evaluating {len(finetuned_files)} fine-tuned files...")
            finetuned_results = evaluate_batch(finetuned_files, prompts, "finetuned")
            all_results.extend(finetuned_results)
    else:
        print(f"Fine-tuned directory not found: {args.finetuned_dir}")

    if not all_results:
        print("No audio files found to evaluate.")
        return

    # Save detailed metrics
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_metrics_csv(all_results, args.output)
    print(f"\nSaved detailed metrics to: {args.output}")

    # Compute and display summary
    stats = compute_summary_stats(all_results)
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for model_name, model_stats in stats.items():
        print(f"\n--- {model_name.upper()} ---")
        print(f"  Samples evaluated:      {model_stats['count']}")
        print(f"  Mean BPM error:         {model_stats['mean_bpm_error']:.1f}")
        print(f"  Mean RMS energy:        {model_stats['mean_rms']:.4f}")
        print(f"  Mean low freq ratio:    {model_stats['mean_low_freq_ratio']:.4f}")
        print(f"  Mean onset density:     {model_stats['mean_onset_density']:.4f}")
        print(f"  Mean silence ratio:     {model_stats['mean_silence_ratio']:.4f}")
        print(f"  Mean clipping ratio:    {model_stats['mean_clipping_ratio']:.4f}")
        print(f"  Mean loop similarity:   {model_stats['mean_loop_similarity']:.4f}")

    # Per-genre comparison
    if len(stats) > 1:
        import pandas as pd
        df = pd.DataFrame(all_results)
        print("\n--- PER-GENRE BPM ERROR ---")
        for genre in df["genre"].unique():
            genre_df = df[df["genre"] == genre]
            for model in df["model"].unique():
                m_df = genre_df[genre_df["model"] == model]
                if len(m_df) > 0:
                    print(f"  {genre:20s} {model:10s} BPM error: {m_df['bpm_error'].mean():.1f}")


if __name__ == "__main__":
    main()
