# EDM-Adapter 训练最佳实践

本指南提供微调 ACE-Step 模型的最佳实践和高级技巧，帮助你获得最佳的音乐生成效果。

---

## 1. 数据质量优化

### 1.1 音频质量要求

| 指标 | 最低要求 | 推荐值 | 说明 |
|------|---------|--------|------|
| 采样率 | 44.1kHz | 48kHz | ACE-Step 原生支持 48kHz |
| 位深 | 16bit | 24bit | 更高的动态范围 |
| 时长 | 10秒 | 30-180秒 | 太短缺乏上下文，太长增加训练时间 |
| 响度 | -20 LUFS | -14 LUFS | 统一响度有助于训练稳定性 |
| 静音比例 | < 30% | < 10% | 过多静音会降低训练效率 |

### 1.2 使用高级预处理器

```bash
# 基础预处理（推荐）
python scripts/advanced_audio_preprocessor.py \
    --input_dir data/finetune/audio \
    --output_dir data/finetune/audio_processed \
    --segment_duration 30

# 固定时长预处理（适合短音频训练）
python scripts/advanced_audio_preprocessor.py \
    --input_dir data/finetune/audio \
    --output_dir data/finetune/audio_30s \
    --target_duration 30

# 完整预处理（最严格）
python scripts/advanced_audio_preprocessor.py \
    --input_dir data/finetune/audio \
    --output_dir data/finetune/audio_full \
    --segment_duration 60 \
    --target_lufs -14
```

### 1.3 预处理功能说明

| 功能 | 说明 | 建议 |
|------|------|------|
| 片段选择 | 自动选择音频中质量最好的片段 | ✅ 推荐开启 |
| 响度标准化 | 统一所有音频的响度水平 | ✅ 推荐开启 |
| 静音移除 | 移除首尾静音部分 | ✅ 推荐开启 |
| 固定时长 | 将所有音频裁剪/填充到相同长度 | ⚠️ 按需使用 |

---

## 2. 标签质量优化

### 2.1 标签内容建议

**必填标签：**
- 音乐风格（genre）
- 节奏速度（BPM）
- 调性（key）

**推荐标签：**
- 情绪氛围（mood）
- 能量级别（energy）
- 主要乐器（instruments）
- 制作人/艺术家风格（artist style）

**示例：**
```
Avicii style, trance, uplifting, euphoric, soaring, 136 BPM, D key, heavy bass, synthesizer, emotional
```

### 2.2 标签格式规范

```bash
# ✅ 正确格式
"Avicii style, trance, uplifting, 136 BPM, D key, heavy bass"

# ❌ 错误格式
"Avicii风格，trance，uplifting，136 BPM，D key，heavy bass"  # 避免中文标点
"Avicii style trance uplifting 136 BPM D key heavy bass"      # 缺少逗号分隔
```

### 2.3 自动标签优化

```bash
# 使用自动标签脚本
python scripts/auto_tag_audio.py \
    "data/finetune/audio" \
    "data/finetune/metadata.jsonl"

# 然后手动检查和修正关键标签
# 特别关注：
# - BPM 检测是否准确
# - 风格分类是否正确
# - 能量级别是否合理
```

---

## 3. 训练参数优化

### 3.1 数据集大小与参数关系

| 数据集大小 | 学习率 | 最大步数 | LoRA Rank | 批次大小 | 预计时间 |
|-----------|--------|---------|-----------|---------|---------|
| < 50 首 | 1e-4 | 5,000 | 32 | 1-2 | 1-2 小时 |
| 50-200 首 | 5e-5 | 10,000 | 64 | 2-4 | 2-4 小时 |
| 200-500 首 | 2e-5 | 20,000 | 64-128 | 4-8 | 4-8 小时 |
| 500+ 首 | 1e-5 | 30,000 | 128 | 8+ | 8+ 小时 |

### 3.2 GPU 显存优化

| GPU 显存 | 精度 | 梯度累积 | LoRA Rank | 批次大小 |
|---------|------|---------|-----------|---------|
| 12GB | FP16 | 4 | 32 | 1 |
| 16GB | FP16 | 2 | 64 | 1-2 |
| 24GB | BF16 | 1 | 64-128 | 2-4 |
| 40GB+ | BF16 | 1 | 128 | 4-8 |

### 3.3 使用配置优化器

```bash
# 自动推荐配置
python scripts/train_config_optimizer.py \
    --dataset_path edm_lora_dataset \
    --quality balanced

# 快速训练模式
python scripts/train_config_optimizer.py \
    --dataset_path edm_lora_dataset \
    --quality fast

# 高质量模式
python scripts/train_config_optimizer.py \
    --dataset_path edm_lora_dataset \
    --quality quality

# 指定 GPU 显存
python scripts/train_config_optimizer.py \
    --dataset_path edm_lora_dataset \
    --gpu_memory 16000
```

### 3.4 质量加权采样

对于数据质量参差不齐的数据集，启用质量加权采样可以让模型更多地学习高质量样本：

```bash
# 在数据准备阶段启用
python scripts/train_edm_lora.py --prepare_data --quality_weighting
```

质量评分基于：
- RMS 能量（不要太安静也不要削波）
- 静音比例（越少越好）
- 频谱丰富度（频谱质心在 1000-5000 Hz 为佳）
- 节奏活跃度（起音密度越高越好）

评分 0.7+ 的样本会获得 3-5 倍的训练权重。

### 3.5 学习率调度策略

本框架使用 **余弦退火 + 线性预热** 调度器，比原始的线性衰减效果更好：

```
学习率
  ^
  |    /\
  |   /  \
  |  /    \___
  | /         \___
  |/              \___
  +-------------------> 步数
  [warmup] [余弦退火]
```

- **预热阶段**：学习率从 0 线性增长到目标值（默认 5% 的 max_steps）
- **退火阶段**：学习率按余弦曲线平滑衰减

```bash
# 自定义 warmup 步数
python scripts/train_edm_lora.py --train --warmup_steps 500

# 使用自动计算（推荐，默认为 max_steps 的 5%）
python scripts/train_edm_lora.py --train
```

---

## 4. 训练过程监控

### 4.1 关键指标

| 指标 | 正常范围 | 异常处理 |
|------|---------|---------|
| Loss | 0.01 - 0.1 | 持续不下降：降低学习率 |
| Learning Rate | 按计划变化 | 检查 warmup 设置 |
| GPU 内存 | < 90% 使用率 | 降低批次大小或 LoRA Rank |
| 训练速度 | 1-3 秒/步 | 检查数据加载瓶颈 |

### 4.2 TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/finetune/logs --port 6006

# 关注的图表：
# - train/loss: 训练损失
# - train/learning_rate: 学习率变化
# - train/denoising_loss: 去噪损失
```

### 4.3 生成样本检查

训练过程中会定期生成样本音频，检查：

1. **音质**：是否有明显的噪声或失真
2. **风格一致性**：是否符合训练数据的风格
3. **多样性**：不同种子生成的结果是否有变化
4. **可控性**：不同的提示词是否影响生成结果

---

## 5. 常见问题与解决方案

### 5.1 训练不收敛

**症状**：Loss 持续高位震荡或不下降

**解决方案**：
```bash
# 1. 降低学习率
--learning_rate 1e-5

# 2. 增加 warmup 步数
--warmup_steps 500

# 3. 检查数据质量
python scripts/advanced_audio_preprocessor.py \
    --input_dir data/finetune/audio \
    --output_dir data/finetune/audio_check
```

### 5.2 过拟合

**症状**：训练 Loss 很低但生成效果差

**解决方案**：
```bash
# 1. 减少训练步数
--max_steps 3000

# 2. 增加 LoRA Dropout（在配置文件中）
"lora_dropout": 0.1

# 3. 使用更小的 LoRA Rank
--lora_config_path config/edm_lora_config_small.json

# 4. 增加数据多样性
# - 添加更多不同的音频
# - 增加数据重复次数
--repeat_count 500
```

### 5.3 显存不足 (OOM)

**解决方案**：
```bash
# 1. 使用 FP16 精度
--precision 16

# 2. 增加梯度累积
--accumulate_grad_batches 4

# 3. 使用更小的 LoRA 配置
--lora_config_path config/edm_lora_config_small.json

# 4. 减少数据加载线程
--num_workers 2
```

### 5.4 生成质量下降

**解决方案**：
```bash
# 1. 检查提示词质量
# 确保标签准确、格式正确

# 2. 调整推理参数
# - guidance_scale: 10-20（越高越严格遵循提示）
# - infer_step: 50-100（越多质量越好）

# 3. 使用多个种子尝试
# 不同种子可能产生不同质量的结果
```

---

## 6. 高级技巧

### 6.1 分阶段训练

对于大数据集，可以分阶段训练：

```bash
# 阶段 1：基础训练（高学习率）
python scripts/train_edm_lora.py --train \
    --learning_rate 1e-4 \
    --max_steps 5000

# 阶段 2：精细训练（低学习率，从 checkpoint 恢复）
python scripts/train_edm_lora.py --train --resume \
    --learning_rate 1e-5 \
    --max_steps 10000
```

### 6.2 多风格混合训练

如果数据包含多种风格：

```bash
# 1. 为每种风格创建单独的数据集
python scripts/train_config_optimizer.py \
    --dataset_path edm_lora_dataset_techno \
    --quality balanced

# 2. 训练多个 LoRA
python scripts/train_edm_lora.py --train \
    --exp_name techno_lora \
    --dataset_path edm_lora_dataset_techno

# 3. 在推理时选择不同风格的 LoRA
```

### 6.3 数据增强

对于小数据集，可以使用数据增强：

```python
# 在 advanced_audio_preprocessor.py 中添加
def augment_audio(y, sr):
    """音频数据增强"""
    augmented = []
    
    # 1. 时间拉伸
    y_stretch = librosa.effects.time_stretch(y, rate=0.9)
    augmented.append(y_stretch)
    
    # 2. 音高偏移
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    augmented.append(y_shift)
    
    # 3. 添加轻微噪声
    noise = np.random.randn(len(y)) * 0.005
    y_noise = y + noise
    augmented.append(y_noise)
    
    return augmented
```

### 6.4 混合精度训练

```bash
# 使用 BF16（需要 Ampere 或更新的 GPU）
--precision bf16

# 使用 FP16（兼容性更好）
--precision 16

# 使用 FP32（最稳定，但最慢）
--precision 32
```

---

## 7. 评估与选择最佳 Checkpoint

### 7.1 评估指标

| 指标 | 说明 | 如何评估 |
|------|------|---------|
| 主观音质 | 生成音乐的听感 | 人工试听 |
| 风格一致性 | 是否符合目标风格 | 与参考音频对比 |
| 提示词遵循 | 是否按提示词生成 | 变化提示词测试 |
| 多样性 | 不同种子的差异 | 多次生成对比 |

### 7.2 选择最佳 Checkpoint

```bash
# 1. 查看所有 checkpoint
python scripts/train_edm_lora.py --status

# 2. 对不同 checkpoint 进行生成测试
python demo/app.py  # 在界面中选择不同 checkpoint

# 3. 记录每个 checkpoint 的表现
# - 选择 loss 稳定且生成质量好的
# - 避免选择过拟合的 checkpoint
```

---

## 8. 部署与使用

### 8.1 导出 LoRA 权重

```bash
# 找到最佳 checkpoint
ls outputs/finetune/logs/xxx/checkpoints/

# 复制到可用目录
cp -r outputs/finetune/logs/xxx/checkpoints/epoch=0-step=5000_lora \
      outputs/checkpoints/my_edm_lora
```

### 8.2 在 Gradio 界面使用

```bash
python demo/app.py
# 在"模型选择"下拉菜单中选择你的 LoRA
```

### 8.3 在代码中使用

```python
from src.generation import load_acestep_model, generate_acestep

# 加载模型
pipeline = load_acestep_model()

# 加载 LoRA
pipeline.load_lora("outputs/checkpoints/my_edm_lora", 1.0)

# 生成音乐
audio, sr = generate_acestep(
    pipeline=pipeline,
    prompt="your style description",
    lyrics="[instrumental]",
    duration=60.0,
    seed=42,
)
```

---

## 9. 检查清单

### 训练前检查

- [ ] 音频文件质量良好（无明显噪声、失真）
- [ ] 标签准确且格式正确
- [ ] 数据集已正确转换为 HuggingFace 格式
- [ ] 启用质量加权采样（`--quality_weighting`）
- [ ] GPU 显存足够
- [ ] 模型已下载
- [ ] Warmup 步数已合理设置（默认自动为 max_steps 的 5%）

### 训练中检查

- [ ] Loss 正常下降
- [ ] GPU 内存使用正常
- [ ] 定期生成样本检查质量
- [ ] Checkpoint 正常保存

### 训练后检查

- [ ] 选择最佳 checkpoint
- [ ] 测试不同风格的生成效果
- [ ] 确认音质满足要求
- [ ] 导出 LoRA 权重

---

## 10. 参考资源

- [ACE-Step 官方仓库](https://github.com/ace-step/ACE-Step)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 库文档](https://huggingface.co/docs/peft)
- [PyTorch Lightning 文档](https://lightning.ai/docs/pytorch/stable/)

---

**祝你训练顺利，生成出满意的音乐！** 🎵
