"""EDM-Adapter 高级训练配置生成器

根据数据集大小和 GPU 显存自动推荐最佳训练参数。
"""

import os
import json
from pathlib import Path


def detect_gpu_memory():
    """检测 GPU 显存大小（MB）"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
    except:
        pass
    return 0


def count_dataset_samples(dataset_path: str) -> int:
    """统计数据集样本数量"""
    try:
        from datasets import load_from_disk
        ds = load_from_disk(dataset_path)
        return len(ds)
    except:
        return 0


def get_recommended_config(
    dataset_size: int = 0,
    gpu_memory_mb: int = 0,
    quality_level: str = "balanced",  # "fast", "balanced", "quality"
) -> dict:
    """根据条件推荐训练配置

    Args:
        dataset_size: 数据集样本数量
        gpu_memory_mb: GPU 显存大小（MB）
        quality_level: 质量级别

    Returns:
        dict: 推荐的训练配置
    """

    # 基础配置
    config = {
        "lora_config": "config/edm_lora_config.json",
        "learning_rate": 1e-4,
        "max_steps": 10000,
        "every_n_train_steps": 1000,
        "num_workers": 4,
        "precision": "32",
        "accumulate_grad_batches": 1,
        "gradient_clip_val": 0.5,
        "every_plot_step": 1000,
        "shift": 3.0,
        "warmup_steps": 100,
        "weight_decay": 0.01,
    }

    # 根据数据集大小调整
    if dataset_size > 0:
        if dataset_size < 50:
            # 小数据集：更多重复，更少步数，更高学习率
            config["max_steps"] = 5000
            config["learning_rate"] = 1e-4
            config["every_n_train_steps"] = 500
            config["lora_config"] = "config/edm_lora_config_small.json"
        elif dataset_size < 200:
            # 中等数据集
            config["max_steps"] = 10000
            config["learning_rate"] = 5e-5
            config["every_n_train_steps"] = 1000
            config["lora_config"] = "config/edm_lora_config_medium.json"
        else:
            # 大数据集
            config["max_steps"] = 20000
            config["learning_rate"] = 2e-5
            config["every_n_train_steps"] = 2000
            config["lora_config"] = "config/edm_lora_config_large.json"
    # warmup_steps = 5% of max_steps (will be auto-calculated if 0)
    config["warmup_steps"] = max(50, int(config["max_steps"] * 0.05))

    # 根据 GPU 显存调整
    if gpu_memory_mb > 0:
        if gpu_memory_mb < 12000:
            # 12GB 以下
            config["precision"] = "16"
            config["accumulate_grad_batches"] = 4
            config["num_workers"] = 2
            config["lora_config"] = "config/edm_lora_config_small.json"
        elif gpu_memory_mb < 16000:
            # 12-16GB
            config["precision"] = "16"
            config["accumulate_grad_batches"] = 2
            config["num_workers"] = 4
        elif gpu_memory_mb < 24000:
            # 16-24GB
            config["precision"] = "bf16"
            config["accumulate_grad_batches"] = 1
            config["num_workers"] = 8
        else:
            # 24GB+
            config["precision"] = "bf16"
            config["accumulate_grad_batches"] = 1
            config["num_workers"] = 8
            config["lora_config"] = "config/edm_lora_config_large.json"

    # 根据质量级别调整
    if quality_level == "fast":
        config["max_steps"] = min(config["max_steps"], 3000)
        config["learning_rate"] *= 2
        config["every_n_train_steps"] = 500
        config["every_plot_step"] = 500
    elif quality_level == "quality":
        config["max_steps"] = int(config["max_steps"] * 1.5)
        config["learning_rate"] *= 0.5
        config["every_n_train_steps"] = max(500, config["every_n_train_steps"] // 2)
        config["warmup_steps"] = int(config["warmup_steps"] * 1.5)

    return config


def print_config_summary(config: dict, dataset_size: int = 0, gpu_memory_mb: int = 0):
    """打印配置摘要"""
    print("=" * 60)
    print("推荐训练配置")
    print("=" * 60)

    if dataset_size > 0:
        print(f"\n数据集大小: {dataset_size} 样本")
    if gpu_memory_mb > 0:
        print(f"GPU 显存: {gpu_memory_mb} MB ({gpu_memory_mb/1024:.1f} GB)")

    print(f"\n[LoRA 配置]")
    print(f"  配置文件: {config['lora_config']}")
    try:
        with open(config['lora_config'], 'r') as f:
            lora = json.load(f)
        print(f"  Rank (r): {lora.get('r', 'N/A')}")
        print(f"  Alpha: {lora.get('lora_alpha', 'N/A')}")
    except:
        pass

    print(f"\n[训练参数]")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  最大步数: {config['max_steps']}")
    print(f"  Warmup 步数: {config['warmup_steps']}")
    print(f"  保存间隔: {config['every_n_train_steps']}")
    print(f"  精度: {config['precision']}")
    print(f"  梯度累积: {config['accumulate_grad_batches']}")
    print(f"  数据加载线程: {config['num_workers']}")

    print(f"\n[预估训练时间]")
    # 假设每步约 1-2 秒
    steps = config['max_steps']
    accum = config['accumulate_grad_batches']
    effective_steps = steps * accum
    min_time = effective_steps * 1 / 60
    max_time = effective_steps * 2 / 60
    print(f"  约 {min_time:.0f} - {max_time:.0f} 分钟")
    print(f"  (取决于 GPU 性能)")


def generate_training_command(config: dict, base_args: dict = None) -> str:
    """生成训练命令"""
    cmd_parts = ["python scripts/train_edm_lora.py --train"]

    if base_args:
        for k, v in base_args.items():
            if v is not None:
                cmd_parts.append(f"--{k} {v}")

    cmd_parts.extend([
        f"--learning_rate {config['learning_rate']}",
        f"--max_steps {config['max_steps']}",
        f"--warmup_steps {config['warmup_steps']}",
        f"--every_n_train_steps {config['every_n_train_steps']}",
        f"--num_workers {config['num_workers']}",
        f"--precision {config['precision']}",
        f"--accumulate_grad_batches {config['accumulate_grad_batches']}",
        f"--gradient_clip_val {config['gradient_clip_val']}",
        f"--every_plot_step {config['every_plot_step']}",
        f"--shift {config['shift']}",
        f"--lora_config_path {config['lora_config']}",
    ])

    return " \\\n    ".join(cmd_parts)


def save_config_template(config: dict, output_path: str):
    """保存配置模板"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    template = {
        "description": "EDM-Adapter 训练配置模板",
        "created_by": "train_config_optimizer.py",
        "config": config,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=4, ensure_ascii=False)

    print(f"\n配置已保存到: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练配置优化器")
    parser.add_argument("--dataset_path", type=str, default="edm_lora_dataset",
                       help="数据集路径")
    parser.add_argument("--quality", type=str, default="balanced",
                       choices=["fast", "balanced", "quality"],
                       help="质量级别")
    parser.add_argument("--output", type=str, default=None,
                       help="保存配置到文件")
    parser.add_argument("--gpu_memory", type=int, default=0,
                       help="GPU 显存大小（MB），0=自动检测")

    args = parser.parse_args()

    # 检测 GPU
    gpu_memory = args.gpu_memory
    if gpu_memory == 0:
        gpu_memory = detect_gpu_memory()
        if gpu_memory == 0:
            print("无法检测 GPU 显存，请使用 --gpu_memory 参数指定")
            gpu_memory = 16000  # 默认假设 16GB

    # 统计数据集
    dataset_size = 0
    if os.path.exists(args.dataset_path):
        dataset_size = count_dataset_samples(args.dataset_path)

    # 获取推荐配置
    config = get_recommended_config(
        dataset_size=dataset_size,
        gpu_memory_mb=gpu_memory,
        quality_level=args.quality,
    )

    # 打印摘要
    print_config_summary(config, dataset_size, gpu_memory)

    # 生成训练命令
    print(f"\n[训练命令]")
    cmd = generate_training_command(config)
    print(cmd)

    # 保存配置
    if args.output:
        save_config_template(config, args.output)
