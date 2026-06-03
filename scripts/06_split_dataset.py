"""Split dataset into train/val/test sets."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, setup_logging
from src.dataset_builder import load_metadata_jsonl, save_metadata_jsonl, split_dataset


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--input", default="data/processed/metadata.jsonl")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    config = load_yaml(args.config)
    split_config = config["split"]

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        print("Run 05_build_captions.py first.")
        return

    records = load_metadata_jsonl(args.input)
    print(f"Total records: {len(records)}")

    train, val, test = split_dataset(
        records,
        train_ratio=split_config["train"],
        val_ratio=split_config["val"],
        test_ratio=split_config["test"],
        group_key="track_id",
        seed=args.seed,
    )

    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_metadata_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    save_metadata_jsonl(val, os.path.join(args.output_dir, "val.jsonl"))
    save_metadata_jsonl(test, os.path.join(args.output_dir, "test.jsonl"))

    print(f"\nSaved splits to {args.output_dir}/")


if __name__ == "__main__":
    main()
