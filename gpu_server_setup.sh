#!/bin/bash
# EDM-Adapter GPU 服务器快速部署脚本
# 使用方法: bash gpu_server_setup.sh

set -e

echo "=========================================="
echo "EDM-Adapter GPU 服务器部署"
echo "=========================================="

# 1. 检查 CUDA
echo ""
echo "1. 检查 CUDA 环境..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✅ CUDA 可用"
else
    echo "❌ 未检测到 NVIDIA GPU，请确保已安装 CUDA 驱动"
    exit 1
fi

# 2. 检查 Python
echo ""
echo "2. 检查 Python 环境..."
python_version=$(python3 --version 2>&1)
echo "Python 版本: $python_version"

# 3. 创建虚拟环境（可选）
echo ""
read -p "是否创建虚拟环境? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ 虚拟环境已激活"
fi

# 4. 安装依赖
echo ""
echo "3. 安装依赖..."
pip install --upgrade pip

# 安装 PyTorch (CUDA 12.x)
echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 安装 ACE-Step
echo "安装 ACE-Step..."
cd ACE-Step
pip install -e .
cd ..

# 安装训练依赖
echo "安装训练依赖..."
pip install peft datasets pytorch-lightning tensorboard transformers diffusers accelerate

echo ""
echo "✅ 依赖安装完成"

# 5. 检查模型
echo ""
echo "4. 检查模型..."
if [ -d "models/ace-step/ACE-Step-v1-3.5B" ]; then
    echo "✅ ACE-Step 模型已存在"
else
    echo "⚠️  ACE-Step 模型不存在"
    read -p "是否下载模型? (y/n): " download_model
    if [ "$download_model" = "y" ]; then
        echo "下载 ACE-Step 模型 (~8GB)..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'ACE-Step/ACE-Step-v1-3.5B',
    local_dir='models/ace-step/ACE-Step-v1-3.5B',
    resume_download=True
)
"
        echo "✅ 模型下载完成"
    fi
fi

# 6. 准备数据
echo ""
echo "5. 准备训练数据..."
if [ -f "data/finetune/metadata.jsonl" ]; then
    echo "✅ metadata.jsonl 已存在"
    read -p "是否重新准备训练数据? (y/n): " prepare_data
    if [ "$prepare_data" = "y" ]; then
        python3 scripts/train_edm_lora.py --prepare_data
    fi
else
    echo "⚠️  metadata.jsonl 不存在"
    echo "请先准备数据集，参考 docs/微调指南.md"
fi

# 7. 完成
echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "接下来可以运行："
echo ""
echo "  # 查看训练状态"
echo "  python scripts/train_edm_lora.py --status"
echo ""
echo "  # 开始训练"
echo "  python scripts/train_edm_lora.py --train"
echo ""
echo "  # 自定义参数训练"
echo "  python scripts/train_edm_lora.py --train --max_steps 10000 --learning_rate 5e-5"
echo ""
echo "  # 监控训练"
echo "  tensorboard --logdir outputs/finetune/logs --port 6006"
echo ""
