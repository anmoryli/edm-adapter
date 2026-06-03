"""Generate audio using fine-tuned ACE-Step model (with LoRA)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_yaml, setup_logging
from src.generation import load_acestep_model, generate_batch


def find_latest_lora_checkpoint(log_root: str = "outputs/finetune/logs") -> str | None:
    """Find the newest ACE-Step LoRA adapter checkpoint saved by trainer.py."""
    candidates = []
    if not os.path.isdir(log_root):
        return None

    for root, _, files in os.walk(log_root):
        has_adapter = "adapter_config.json" in files
        has_weights = any(f.endswith(".safetensors") for f in files)
        if has_adapter or has_weights:
            candidates.append(root)

    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="使用微调模型生成音频（ACE-Step + LoRA）")
    parser.add_argument("--lora-path", default=None, help="LoRA 权重路径；留空时自动查找最新 checkpoint")
    parser.add_argument("--lora-weight", type=float, default=1.0, help="LoRA 权重强度")
    parser.add_argument("--config", default="configs/prompts.yaml")
    parser.add_argument("--output-dir", default="outputs/finetuned")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    setup_logging()
    config = load_yaml(args.config)
    seeds = config.get("seeds", args.seeds)
    prompts = config["prompts"]

    if args.lora_path is None:
        args.lora_path = find_latest_lora_checkpoint()

    print(f"使用微调模型生成 {len(prompts)} 个提示词的音频")
    print(f"LoRA 路径: {args.lora_path or '未找到'}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 ACE-Step
    print("加载 ACE-Step 模型...")
    pipeline = load_acestep_model(
        checkpoint_dir="",
        device="auto",
        cpu_offload=True,
        dtype="float32",
    )

    # 加载 LoRA 权重
    lora_path = args.lora_path
    if lora_path and os.path.exists(lora_path):
        try:
            pipeline.load_lora(lora_path, args.lora_weight)
            print(f"已加载 LoRA 权重: {lora_path}")
        except Exception as e:
            print(f"LoRA 加载失败: {e}")
            print("使用基础模型")
    else:
        print(f"LoRA 路径不存在: {lora_path}")
        print("使用基础模型")

    output_files = generate_batch(
        pipeline, None, prompts, args.output_dir,
        model_type="acestep", seeds=seeds,
    )

    print(f"\n生成完成，共 {len(output_files)} 个文件，保存在 {args.output_dir}")


if __name__ == "__main__":
    main()
