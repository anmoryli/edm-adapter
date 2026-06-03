"""EDM-Adapter 微调脚本：一键启动 ACE-Step LoRA 训练

使用方法：
    python scripts/train_edm_lora.py --help

示例：
    # 准备数据
    python scripts/train_edm_lora.py --prepare_data

    # 开始训练（单卡 GPU）
    python scripts/train_edm_lora.py --train

    # 自定义参数训练
    python scripts/train_edm_lora.py --train --max_steps 5000 --learning_rate 5e-5
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ============================================================
# 配置
# ============================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ACE-Step 路径
ACESTEP_DIR = os.path.join(PROJECT_ROOT, "ACE-Step")
ACESTEP_CONFIG = os.path.join(ACESTEP_DIR, "config", "edm_lora_config.json")

# 数据路径
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "finetune", "metadata.jsonl")
AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "finetune", "audio")
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "finetune", "training")
DATASET_PATH = os.path.join(PROJECT_ROOT, "edm_lora_dataset")

# 模型路径
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "ace-step", "ACE-Step-v1-3.5B")

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "finetune")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
EXPS_DIR = os.path.join(OUTPUT_DIR, "exps")


def prepare_data(args):
    """准备训练数据"""
    print("=" * 60)
    print("步骤 1: 准备训练数据")
    print("=" * 60)

    # 调用 prepare_training_data.py
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "prepare_training_data.py"),
        args.metadata_path,
        args.audio_dir,
        args.training_data_dir,
    ]
    if args.copy_audio:
        cmd.append("--copy")

    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("数据准备失败！")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("步骤 2: 转换为 HuggingFace Dataset 格式")
    print("=" * 60)

    # 复制 EDM LoRA 配置到 ACE-Step 目录
    config_src = os.path.join(PROJECT_ROOT, "config", "edm_lora_config.json")
    config_dst = os.path.join(ACESTEP_DIR, "config", "edm_lora_config.json")
    os.makedirs(os.path.dirname(config_dst), exist_ok=True)
    shutil.copy2(config_src, config_dst)
    print(f"LoRA 配置已复制到: {config_dst}")

    # 运行 convert2hf_dataset.py
    repeat_count = args.repeat_count
    if repeat_count <= 0:
        # 根据数据量自动计算重复次数
        metadata_count = sum(1 for _ in open(args.metadata_path, 'r', encoding='utf-8'))
        if metadata_count < 50:
            repeat_count = 2000
        elif metadata_count < 100:
            repeat_count = 1000
        elif metadata_count < 500:
            repeat_count = 500
        else:
            repeat_count = 200
        print(f"数据量: {metadata_count} 条, 自动设置重复次数: {repeat_count}")

    cmd = [
        sys.executable,
        os.path.join(ACESTEP_DIR, "convert2hf_dataset.py"),
        "--data_dir", args.training_data_dir,
        "--repeat_count", str(repeat_count),
        "--output_name", args.dataset_path,
    ]

    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ACESTEP_DIR)
    if result.returncode != 0:
        print("数据集转换失败！")
        sys.exit(1)

    # 步骤 3: 质量加权采样（可选）
    if args.quality_weighting:
        print("\n" + "=" * 60)
        print("步骤 3: 基于音频质量的加权采样")
        print("=" * 60)

        from quality_sampler import run_quality_sampling
        report_path = os.path.join(args.training_data_dir, "quality_report.json")
        report = run_quality_sampling(
            args.training_data_dir,
            output_report=report_path,
            base_repeat=1,
        )
        if report:
            print(f"质量加权后的训练样本数: {report['total_samples']}")

    print("\n数据准备完成！")
    print(f"训练数据目录: {args.training_data_dir}")
    print(f"数据集路径: {args.dataset_path}")


def find_latest_checkpoint(log_dir: str) -> str:
    """查找最新的 checkpoint 目录用于断点续训"""
    if not os.path.exists(log_dir):
        return None

    # 查找所有包含 lora 权重的目录
    checkpoints = []
    for root, dirs, files in os.walk(log_dir):
        if "adapter_config.json" in files or "adapter_model.safetensors" in files:
            checkpoints.append(root)

    if not checkpoints:
        return None

    # 按修改时间排序，返回最新的
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoints[0]


def get_training_progress(log_dir: str) -> dict:
    """获取训练进度信息"""
    progress = {
        "total_steps": 0,
        "checkpoints": [],
        "latest_checkpoint": None,
        "log_files": [],
    }

    if not os.path.exists(log_dir):
        return progress

    # 查找 checkpoint
    for root, dirs, files in os.walk(log_dir):
        if "adapter_config.json" in files:
            ckpt_name = os.path.basename(root)
            # 从目录名提取 step 信息: epoch=0-step=1000_lora
            import re
            match = re.search(r'step=(\d+)', ckpt_name)
            if match:
                step = int(match.group(1))
                progress["checkpoints"].append({
                    "path": root,
                    "name": ckpt_name,
                    "step": step,
                    "time": os.path.getmtime(root),
                })
                progress["total_steps"] = max(progress["total_steps"], step)

    # 排序 checkpoints
    progress["checkpoints"].sort(key=lambda x: x["step"])
    if progress["checkpoints"]:
        progress["latest_checkpoint"] = progress["checkpoints"][-1]

    # 查找日志文件
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.endswith(".log") or f == "events.out.tfevents.*":
                progress["log_files"].append(os.path.join(root, f))

    return progress


def stream_training_output(process):
    """实时流式输出训练日志"""
    import threading
    import queue

    def reader(pipe, queue, prefix):
        try:
            for line in iter(pipe.readline, ''):
                queue.put((prefix, line))
        finally:
            pipe.close()

    # 创建线程读取 stdout 和 stderr
    q = queue.Queue()
    stdout_thread = threading.Thread(target=reader, args=(process.stdout, q, ""))
    stderr_thread = threading.Thread(target=reader, args=(process.stderr, q, "[WARN]"))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    # 实时输出
    while True:
        try:
            prefix, line = q.get(timeout=0.1)
            if prefix:
                print(f"{prefix} {line}", end="", flush=True)
            else:
                print(line, end="", flush=True)
        except queue.Empty:
            if process.poll() is not None:
                break

    # 等待线程结束
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)

    return process.returncode


def train(args):
    """启动训练（支持实时日志和断点续训）"""
    print("=" * 60)
    print("启动 ACE-Step LoRA 微调训练")
    print("=" * 60)

    # 检查数据集是否存在
    if not os.path.exists(args.dataset_path):
        print(f"数据集不存在: {args.dataset_path}")
        print("请先运行: python scripts/train_edm_lora.py --prepare_data")
        sys.exit(1)

    # 检查模型是否存在
    if not os.path.exists(args.checkpoint_dir):
        print(f"模型不存在: {args.checkpoint_dir}")
        print("请确保 ACE-Step 模型已下载到此路径")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.exps_dir, exist_ok=True)

    # 处理断点续训
    ckpt_path = args.ckpt_path
    if args.resume:
        # 自动查找最新 checkpoint
        latest_ckpt = find_latest_checkpoint(args.log_dir)
        if latest_ckpt:
            ckpt_path = latest_ckpt
            progress = get_training_progress(args.log_dir)
            print(f"\n🔄 断点续训模式")
            print(f"   从 checkpoint 恢复: {ckpt_path}")
            if progress["latest_checkpoint"]:
                print(f"   已完成步数: {progress['latest_checkpoint']['step']}")
        else:
            print(f"\n⚠️  未找到可用的 checkpoint，将从头开始训练")
            ckpt_path = None

    # Resolve all paths to absolute before passing to subprocess (cwd changes to ACE-Step)
    args.dataset_path = os.path.abspath(args.dataset_path)
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    args.lora_config_path = os.path.abspath(args.lora_config_path)
    args.log_dir = os.path.abspath(args.log_dir)

    if args.warmup_steps <= 0:
        args.warmup_steps = max(1, int(args.max_steps * 0.05))
    if args.warmup_steps >= args.max_steps:
        capped_warmup = max(1, args.max_steps // 5)
        print(
            f"⚠️  warmup_steps({args.warmup_steps}) >= max_steps({args.max_steps})，"
            f"自动改为 {capped_warmup}，避免整段训练都处于极低学习率。"
        )
        args.warmup_steps = capped_warmup
    if args.every_n_train_steps > args.max_steps:
        print(
            f"⚠️  every_n_train_steps({args.every_n_train_steps}) > max_steps({args.max_steps})，"
            f"自动改为 {args.max_steps}，确保至少保存一次 LoRA checkpoint。"
        )
        args.every_n_train_steps = args.max_steps

    # 构建训练命令 — 通过 wrapper 脚本注入 torchaudio 兼容性补丁
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_path = os.path.join(scripts_dir, "run_trainer.py")

    cmd = [
        sys.executable,
        "-u",  # 不缓冲输出，确保实时日志
        wrapper_path,
        "--dataset_path", args.dataset_path,
        "--checkpoint_dir", args.checkpoint_dir,
        "--lora_config_path", args.lora_config_path,
        "--exp_name", args.exp_name,
        "--learning_rate", str(args.learning_rate),
        "--max_steps", str(args.max_steps),
        "--every_n_train_steps", str(args.every_n_train_steps),
        "--num_workers", str(args.num_workers),
        "--devices", str(args.devices),
        "--precision", args.precision,
        "--accumulate_grad_batches", str(args.accumulate_grad_batches),
        "--gradient_clip_val", str(args.gradient_clip_val),
        "--logger_dir", args.log_dir,
        "--every_plot_step", str(args.every_plot_step),
        "--shift", str(args.shift),
        "--warmup_steps", str(args.warmup_steps),
    ]

    if args.use_cpu:
        cmd.append("--use_cpu")

    if ckpt_path:
        cmd.extend(["--ckpt_path", ckpt_path])

    print(f"\n训练配置:")
    print(f"  数据集: {args.dataset_path}")
    print(f"  模型: {args.checkpoint_dir}")
    print(f"  LoRA 配置: {args.lora_config_path}")
    print(f"  实验名称: {args.exp_name}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  最大步数: {args.max_steps}")
    print(f"  保存间隔: {args.every_n_train_steps}")
    print(f"  设备数: {args.devices}")
    print(f"  精度: {args.precision}")
    print(f"  Warmup 步数: {args.warmup_steps} ({'自动' if args.warmup_steps == 0 else '手动'})")
    print(f"  设备: {'CPU' if args.use_cpu else 'GPU'}")
    print(f"  日志目录: {args.log_dir}")
    if ckpt_path:
        print(f"  恢复训练: {ckpt_path}")
    print(f"\n运行命令:")
    print(f"  {' '.join(cmd)}")
    print("\n" + "=" * 60)
    print("训练开始... (实时日志输出)")
    print("=" * 60 + "\n")

    # 启动训练（实时输出日志）
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    process = subprocess.Popen(
        cmd,
        cwd=ACESTEP_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace',
        env=env,
    )

    # 实时流式输出
    return_code = stream_training_output(process)

    if return_code == 0:
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)

        # 显示最终进度
        progress = get_training_progress(args.log_dir)
        if progress["checkpoints"]:
            print(f"\n保存的 checkpoint:")
            for ckpt in progress["checkpoints"][-5:]:
                print(f"  - {ckpt['name']} (step {ckpt['step']})")

        print(f"\n使用方法:")
        print(f"  1. 在 Gradio 界面的模型选择器中选择微调模型")
        print(f"  2. 或使用代码加载:")
        print(f"     pipeline.load_lora('{progress['latest_checkpoint']['path']}', 1.0)")
    else:
        print(f"\n训练失败，退出码: {return_code}")
        sys.exit(1)


def show_status(args):
    """显示训练状态（详细版）"""
    print("=" * 60)
    print("训练状态")
    print("=" * 60)

    # 检查数据准备状态
    print("\n[1] 数据准备:")
    if os.path.exists(args.training_data_dir):
        audio_count = len(list(Path(args.training_data_dir).glob("*.mp3")))
        print(f"   [OK] 训练数据目录: {args.training_data_dir}")
        print(f"   [OK] 音频文件数: {audio_count}")
    else:
        print(f"   [!!] 训练数据目录不存在")

    if os.path.exists(args.dataset_path):
        print(f"   [OK] HuggingFace 数据集: {args.dataset_path}")
    else:
        print(f"   [!!] HuggingFace 数据集不存在")

    # 检查模型状态
    print("\n[2] 模型:")
    if os.path.exists(args.checkpoint_dir):
        print(f"   [OK] ACE-Step 模型: {args.checkpoint_dir}")
    else:
        print(f"   [!!] ACE-Step 模型不存在")

    if os.path.exists(args.lora_config_path):
        print(f"   [OK] LoRA 配置: {args.lora_config_path}")
    else:
        print(f"   [!!] LoRA 配置不存在")

    # 检查训练输出
    print("\n[3] 训练进度:")
    progress = get_training_progress(args.log_dir)

    if not progress["checkpoints"]:
        print(f"   [--] 尚未开始训练或无 checkpoint")
        print(f"\n   启动训练: python scripts/train_edm_lora.py --train")
    else:
        print(f"   [OK] 日志目录: {args.log_dir}")
        print(f"   [OK] 已完成步数: {progress['total_steps']}")
        print(f"   [OK] 已保存 checkpoint: {len(progress['checkpoints'])} 个")

        # 显示最新 checkpoint
        if progress["latest_checkpoint"]:
            latest = progress["latest_checkpoint"]
            from datetime import datetime
            save_time = datetime.fromtimestamp(latest["time"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n   [最新 checkpoint]")
            print(f"      名称: {latest['name']}")
            print(f"      步数: {latest['step']}")
            print(f"      时间: {save_time}")
            print(f"      路径: {latest['path']}")

        # 显示所有 checkpoint 列表
        print(f"\n   [所有 checkpoint]")
        for ckpt in progress["checkpoints"][-5:]:
            from datetime import datetime
            save_time = datetime.fromtimestamp(ckpt["time"]).strftime("%m-%d %H:%M")
            print(f"      - step={ckpt['step']:>6} | {save_time} | {ckpt['name']}")

        # 续训提示
        print(f"\n   [断点续训命令]")
        print(f"      python scripts/train_edm_lora.py --train --resume")
        print(f"\n   [监控训练]")
        print(f"      tensorboard --logdir {args.log_dir} --port 6006")


def main():
    parser = argparse.ArgumentParser(
        description="EDM-Adapter 微调工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 准备数据
  python scripts/train_edm_lora.py --prepare_data

  # 开始训练
  python scripts/train_edm_lora.py --train

  # 从最新 checkpoint 恢复训练
  python scripts/train_edm_lora.py --train --resume

  # 指定 checkpoint 恢复训练
  python scripts/train_edm_lora.py --train --ckpt_path outputs/finetune/logs/xxx/checkpoints/epoch=0-step=1000_lora

  # 查看状态
  python scripts/train_edm_lora.py --status

  # 自定义参数训练
  python scripts/train_edm_lora.py --train --max_steps 5000 --learning_rate 5e-5 --devices 2
        """
    )

    # 操作模式
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare_data", action="store_true", help="准备训练数据")
    mode.add_argument("--train", action="store_true", help="开始训练")
    mode.add_argument("--status", action="store_true", help="查看训练状态")

    # 数据相关参数
    data_group = parser.add_argument_group("数据参数")
    data_group.add_argument("--metadata_path", type=str, default=METADATA_PATH, help="metadata.jsonl 路径")
    data_group.add_argument("--audio_dir", type=str, default=AUDIO_DIR, help="音频文件目录")
    data_group.add_argument("--training_data_dir", type=str, default=TRAINING_DATA_DIR, help="训练数据输出目录")
    data_group.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="HuggingFace 数据集路径")
    data_group.add_argument("--repeat_count", type=int, default=0, help="数据重复次数（0=自动计算）")
    data_group.add_argument("--copy_audio", action="store_true", help="复制音频文件（默认创建软链接）")
    data_group.add_argument("--quality_weighting", action="store_true", help="启用基于音频质量的加权采样（高质量样本重复更多次）")

    # 模型相关参数
    model_group = parser.add_argument_group("模型参数")
    model_group.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, help="ACE-Step 模型路径")
    model_group.add_argument("--lora_config_path", type=str, default=ACESTEP_CONFIG, help="LoRA 配置文件路径")

    # 训练相关参数
    train_group = parser.add_argument_group("训练参数")
    train_group.add_argument("--exp_name", type=str, default="edm_lora", help="实验名称")
    train_group.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    train_group.add_argument("--max_steps", type=int, default=10000, help="最大训练步数")
    train_group.add_argument("--every_n_train_steps", type=int, default=1000, help="保存 checkpoint 间隔")
    train_group.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    train_group.add_argument("--devices", type=int, default=1, help="GPU 数量")
    train_group.add_argument("--precision", type=str, default="32", choices=["16", "32", "bf16"], help="训练精度")
    train_group.add_argument("--accumulate_grad_batches", type=int, default=2, help="梯度累积批次")
    train_group.add_argument("--gradient_clip_val", type=float, default=0.5, help="梯度裁剪值")
    train_group.add_argument("--every_plot_step", type=int, default=1000, help="生成样本间隔")
    train_group.add_argument("--shift", type=float, default=3.0, help="Scheduler shift 参数")
    train_group.add_argument("--warmup_steps", type=int, default=0, help="学习率预热步数（0=自动：max_steps 的 5%%）")
    train_group.add_argument("--use_cpu", action="store_true", help="使用 CPU 训练（极慢，仅用于测试）")
    train_group.add_argument("--resume", action="store_true", help="从最新 checkpoint 恢复训练")
    train_group.add_argument("--ckpt_path", type=str, default=None, help="指定 checkpoint 路径恢复训练")
    train_group.add_argument("--log_dir", type=str, default=LOG_DIR, help="日志目录")
    train_group.add_argument("--exps_dir", type=str, default=EXPS_DIR, help="实验目录")

    args = parser.parse_args()

    # 确保 LoRA 配置存在
    if not os.path.exists(args.lora_config_path):
        print(f"LoRA 配置不存在: {args.lora_config_path}")
        print("创建默认配置...")
        os.makedirs(os.path.dirname(args.lora_config_path), exist_ok=True)
        default_config = {
            "r": 64,
            "lora_alpha": 16,
            "target_modules": [
                "linear_q",
                "linear_k",
                "linear_v",
                "to_q",
                "to_k",
                "to_v",
                "to_out.0"
            ],
            "use_rslora": True
        }
        with open(args.lora_config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        print(f"已创建: {args.lora_config_path}")

    # 执行操作
    if args.prepare_data:
        prepare_data(args)
    elif args.train:
        train(args)
    elif args.status:
        show_status(args)


if __name__ == "__main__":
    main()
