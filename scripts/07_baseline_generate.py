"""Generate baseline audio using ACE-Step (before fine-tuning)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, setup_logging
from src.generation import load_acestep_model, generate_batch


def main():
    parser = argparse.ArgumentParser(description="生成基线音频（ACE-Step）")
    parser.add_argument("--config", default="configs/prompts.yaml")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    setup_logging()
    config = load_yaml(args.config)
    seeds = config.get("seeds", args.seeds)
    prompts = config["prompts"]

    print(f"为 {len(prompts)} 个提示词生成基线音频，每个 {len(seeds)} 个种子")
    print(f"模型: ACE-Step v1-3.5B")

    os.makedirs(args.output_dir, exist_ok=True)

    print("加载 ACE-Step 模型...")
    pipeline = load_acestep_model(
        checkpoint_dir="",
        device="auto",
        cpu_offload=True,
        dtype="float32",
    )

    output_files = generate_batch(
        pipeline, None, prompts, args.output_dir,
        model_type="acestep", seeds=seeds,
    )
    print(f"\n生成完成，共 {len(output_files)} 个文件，保存在 {args.output_dir}")


if __name__ == "__main__":
    main()
