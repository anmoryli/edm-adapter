# EDM-StructLoRA 技术方案

## 1. 目标

本项目不再只做普通 ACE-Step LoRA 微调，而是做一个面向 EDM 的可控微调方法：

**EDM-StructLoRA: Structure- and Attribute-Aware LoRA Routing for Controllable EDM Generation**

目标是让模型在文本生成音乐时更稳定地控制：

- section：intro / build-up / drop / breakdown / outro / loop
- BPM：节奏速度
- energy：低能量到高能量
- subgenre：melodic house / progressive house / festival EDM 等
- bass / onset / loop：低频、鼓点密度、首尾连贯性

## 2. 方法概述

普通 LoRA 只有一个固定 adapter：

```text
prompt -> ACE-Step + fixed LoRA -> audio
```

EDM-StructLoRA 使用共享 LoRA + 多个专家 LoRA：

```text
prompt + EDM metadata + control curve
        -> router
        -> shared LoRA + section experts + energy experts + subgenre experts
        -> ACE-Step diffusion transformer
        -> audio
```

核心区别是：adapter 权重不再固定，而是由音乐结构和属性动态决定。

## 3. 模块设计

### 3.1 共享 LoRA

共享 LoRA 学习 EDM 域整体偏移，例如：

- 电子鼓组
- sidechain bass
- supersaw / pluck / pad
- 明亮混音
- festival / melodic house 的整体音色分布

默认配置：

```json
{
  "r": 32,
  "lora_alpha": 16,
  "use_rslora": true
}
```

### 3.2 Section 专家 LoRA

每个 section 一个低秩专家：

```text
section_intro
section_build_up
section_drop
section_breakdown
section_outro
section_loop
```

drop 专家主要学习高能量鼓、低频、主旋律冲击；build-up 专家主要学习 snare roll、riser、tension；intro/breakdown 专家学习低密度铺底和渐进结构。

默认配置：

```json
{
  "r": 8,
  "lora_alpha": 8,
  "use_rslora": true
}
```

### 3.3 Attribute 专家 LoRA

属性专家包括：

- energy_low / energy_medium / energy_high / energy_very_high
- subgenre_melodic_house / subgenre_progressive_house / subgenre_festival_edm 等

这些专家不单独决定生成结果，而是和共享 LoRA、section 专家一起混合。

### 3.4 Router

router 根据 metadata 计算 adapter 权重：

```text
weights = f(section, energy, subgenre, bpm, tag_confidence, sample_weight)
```

例如：

```text
prompt: high energy melodic house drop, 128 BPM

edm_shared: 1.0
section_drop: 0.81
energy_high: 0.30
subgenre_melodic_house: 0.36
```

训练时按 batch metadata 动态激活 adapter mixture；生成时根据用户输入的 BPM / section / energy / subgenre 激活对应 adapter。

## 4. 时间控制曲线

每个 clip 会生成一个 latent-frame aligned control curve：

```text
dataset/controls/{clip_id}.pt
```

tensor 形状：

```text
[latent_frames, feature_dim]
```

当前 feature_dim 为 36，包含：

- section one-hot
- subgenre one-hot
- energy one-hot
- energy scalar
- normalized BPM
- BPM confidence
- beat phase sin / cos
- time position
- low-frequency ratio
- onset density
- loop start / end marker
- tag confidence mean
- quality weight

训练时，control curve 会经过一个小型 MLP projector，变成额外 text-conditioning tokens，然后拼接到 UMT5 文本 embedding 后面：

```text
UMT5 caption tokens + EDM control tokens -> ACE-Step transformer
```

这样模型不是只靠 caption 理解 “drop / high energy / 128 BPM”，而是收到显式结构控制信号。

## 5. 训练流程

### 5.1 构建控制曲线

```bash
python scripts/build_edm_control_assets.py --dataset-root dataset
```

输出：

```text
dataset/controls/*.pt
dataset/controls/schema.json
dataset/reports/control_assets_report.md
```

同时更新：

```text
dataset/metadata.jsonl
dataset/metadata.csv
dataset/splits/train.jsonl
dataset/splits/val.jsonl
dataset/splits/test.jsonl
```

### 5.2 转换 ACE-Step 训练数据

```bash
python scripts/prepare_ace_control_dataset.py \
  --dataset-root dataset \
  --split train \
  --output outputs/datasets/edm_control_lora_train \
  --path-mode absolute \
  --min-quality 4
```

输出 HuggingFace Dataset，字段包括：

```text
filename
tags
recaption
section
energy
subgenre
bpm
sample_weight
latent_path
control_path
text_token_path
```

### 5.3 训练

推荐 GPU 命令：

```bash
python scripts/train_edm_control_lora.py \
  --dataset-path outputs/datasets/edm_control_lora_train \
  --checkpoint-dir models/ace-step/ACE-Step-v1-3.5B \
  --config config/edm_control_lora.json \
  --max-steps 10000 \
  --learning-rate 1e-4 \
  --precision bf16-mixed \
  --accumulate-grad-batches 8 \
  --devices 1
```

默认策略：

- 使用缓存好的 ACE-Step DCAE latent，避免每步重复 encode。
- 默认不启用 MERT / mHuBERT SSL loss，先保证 LoRA 消融干净、显存更低。
- 使用 sample_weight 做 confidence-aware weighted sampling。
- 保存 adapter bundle，而不是单个 LoRA 文件。

输出 checkpoint：

```text
outputs/edm_control_lora/logs/.../checkpoints/epoch=*-step=*_edm_control_lora/
  manifest.json
  control_conditioner.pt
  adapters/
    edm_shared/
    section_drop/
    section_build_up/
    energy_high/
    subgenre_melodic_house/
    ...
```

## 6. 生成流程

```bash
python scripts/generate_edm_control_lora.py \
  --prompt "uplifting piano chords, warm sidechain bass, wide supersaw lead" \
  --lora-bundle outputs/edm_control_lora/logs/.../checkpoints/epoch=0-step=10000_edm_control_lora \
  --checkpoint-dir models/ace-step/ACE-Step-v1-3.5B \
  --duration 8 \
  --section drop \
  --energy high \
  --subgenre "melodic house" \
  --bpm 128 \
  --infer-step 60 \
  --seed 1234
```

生成时会：

1. 加载 ACE-Step base model。
2. 加载 manifest 里的所有 LoRA experts。
3. 根据 section / BPM / energy / subgenre 计算 adapter mixture。
4. 加载 control_conditioner，将控制曲线转为 control tokens。
5. 输出 wav 和同名 JSON sidecar，记录 prompt、adapter 权重、参数。

默认不会强行追加很重的风格提示词，避免默认 prompt 压过用户自己的 prompt。需要把控制标签写进 prompt 时才加：

```bash
--append-control-tags
```

## 7. 对照实验

建议的消融实验：

1. ACE-Step base，无 LoRA。
2. 普通 LoRA，r=64。
3. rsLoRA / DoRA baseline。
4. Metadata-rich caption LoRA。
5. Section-aware LoRA。
6. Section + energy + subgenre router。
7. 完整 EDM-StructLoRA：router + control tokens + weighted sampling。

每组参数量要报告，最好做 parameter-matched 对比。

## 8. 评估指标

必须报告：

- BPM error：目标 BPM 和生成 BPM 的误差。
- onset density error：鼓点密度是否随 energy/section 变化。
- low-frequency ratio：drop 是否有更强低频。
- loop similarity：loop 首尾是否更连贯。
- section controllability：目标 section 与分类器预测是否一致。
- CLAP / text-audio similarity：文本控制一致性。
- FAD 或 embedding distance：整体音质和分布距离。
- 人工盲听：风格匹配、drop 冲击、loop 可用性、音质。

## 9. 论文卖点

不要把论文写成：

```text
We fine-tune ACE-Step with LoRA for EDM.
```

应该写成：

```text
We propose a structure- and attribute-conditioned parameter-efficient adaptation framework for controllable EDM generation.
```

贡献点：

1. EDM structure-aware adapter routing。
2. Attribute-conditioned LoRA mixture for BPM / energy / subgenre control。
3. Latent-aligned EDM control curves as lightweight conditioning tokens。
4. EDM controllability benchmark and ablation on ACE-Step。

## 10. 当前代码文件

```text
config/edm_control_lora.json
src/edm_control/
scripts/build_edm_control_assets.py
scripts/prepare_ace_control_dataset.py
scripts/train_edm_control_lora.py
scripts/generate_edm_control_lora.py
docs/EDM_STRUCT_LORA_TECHNICAL_PLAN.md
```

## 11. 已知边界

- 本地机器当前是 CPU torch，不能验证完整 3.5B 模型训练速度。
- 本地 smoke test 只验证数据、路由、控制曲线和脚本入口能跑通。
- 真正训练需要 GPU 环境安装 `requirements-train.txt`。
- 完整实验版本需要补齐训练结果、消融、人工听评和更强 baseline。
