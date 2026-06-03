# 训练脚本说明

## 脚本列表

| 脚本 | 功能 | 使用方法 |
|------|------|---------|
| `auto_tag_audio.py` | 自动分析音频并生成标签 | `python scripts/auto_tag_audio.py <音频目录> <输出路径>` |
| `prepare_training_data.py` | 准备 ACE-Step 训练数据 | `python scripts/prepare_training_data.py` |
| `train_edm_lora.py` | 一键训练脚本 | `python scripts/train_edm_lora.py --train` |

## 快速开始

### 1. 自动标签（如果没有标签）

```bash
python scripts/auto_tag_audio.py "data/finetune/audio" "data/finetune/metadata.jsonl"
```

### 2. 准备数据

```bash
python scripts/train_edm_lora.py --prepare_data
```

### 3. 开始训练

```bash
python scripts/train_edm_lora.py --train
```

### 4. 查看状态

```bash
python scripts/train_edm_lora.py --status
```

## 详细文档

请参考 [微调指南](../docs/微调指南.md)
