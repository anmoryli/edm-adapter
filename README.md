# EDM-Adapter

EDM-Adapter 是一个面向智能语音处理与电子音乐生成实验的本地工程项目，包含 ACE-Step 音乐生成、LoRA 风格微调、Demucs 分轨、Seed-VC 授权歌声音色转换、AI 生成 Mel 水印、Gradio 前端任务队列和论文式实验报告。

仓库地址：https://github.com/anmoryli/edm-adapter

## 项目说明

本仓库只保存可复现项目流程所需的源代码、配置文件、脚本、前端页面和报告文件。为了避免仓库过大，以下内容不会提交到 GitHub：

- 数据集与切片音频
- 本地模型权重、LoRA 权重、ckpt、pth、pt、safetensors
- 生成结果、训练输出、任务输出目录
- 外部工具克隆目录，例如 ACE-Step、Seed-VC
- 本地日志、缓存、node_modules 和压缩包

如果需要完整运行音乐生成或音色转换，需要按本地环境重新准备模型、数据和外部工具。

## 主要功能

- 基于 ACE-Step 的文本到音乐生成
- 面向 EDM/Avicii 风格的 LoRA 微调与同 seed baseline 对比
- 上传歌曲后使用 Demucs 分离人声与伴奏
- 使用 Seed-VC 做授权目标音色转换
- 将换音色后人声加轻混响后与伴奏重新合成为新歌
- 生成完整 Mel 图谱，并在末尾加入频谱负形“AI生成”水印
- 前端展示任务队列、分轨音频、换音色人声、重混新歌、波形图和 Mel 图谱
- 自动生成论文式技术报告、图表和实验过程说明

## 快速开始

```bash
conda create -n edm-adapter python=3.10 -y
conda activate edm-adapter
pip install -r requirements.txt

python demo/app.py
```

默认前端地址：

```text
http://127.0.0.1:7860
```

如果需要指定端口：

```bash
set EDM_GRADIO_PORT=7861
python demo/app.py
```

## 基础训练流程

```bash
# 1. 检查环境
python scripts/00_check_env.py

# 2. 准备元数据
python scripts/01_prepare_metadata.py

# 3. 筛选电子音乐
python scripts/02_filter_electronic.py

# 4. 切分音频片段
python scripts/03_segment_audio.py

# 5. 提取音频特征
python scripts/04_analyze_audio.py

# 6. 构建训练 caption
python scripts/05_build_captions.py

# 7. 划分训练集、验证集和测试集
python scripts/06_split_dataset.py

# 8. 生成 baseline
python scripts/07_baseline_generate.py --model musicgen

# 9. 微调模型
python scripts/08_finetune.py --model musicgen --max-samples 100

# 10. 使用微调后模型生成
python scripts/09_generate_after_finetune.py --model musicgen

# 11. 评估并生成图表
python scripts/10_evaluate.py
python scripts/11_make_report_assets.py
```

## 音色转换说明

音色转换功能依赖授权目标音色样本和本地转换器。当前工程通过 Seed-VC 桥接脚本调用外部转换器，典型流程为：

```text
上传歌曲 -> Demucs 分离 vocals/accompaniment -> Seed-VC 转换 vocals
-> 人声去沙哑与力度修复 -> 轻混响 -> 与伴奏重混 -> 输出新歌
```

可用环境变量：

```bash
set EDM_VOICE_CONVERTER_TIMEOUT_SEC=0
set EDM_VOICE_REVERB_MIX=0.14
set EDM_VOICE_REVERB_SECONDS=0.85
set EDM_VOICE_REVERB_PRE_DELAY_MS=24
```

其中 `EDM_VOICE_CONVERTER_TIMEOUT_SEC=0` 表示不限制 Seed-VC 推理超时时间。

## 目录结构

```text
edm-adapter/
  config/                 配置文件
  demo/                   Gradio 前端
  docs/                   技术方案和实验说明
  report/                 论文报告、图表和 DOCX/PDF/TEX/HTML
  scripts/                数据处理、训练、生成和评估脚本
  src/                    核心 Python 模块
  avicii_training_code/   Avicii/EDM 相关训练与对比脚本
```

以下目录通常只存在于本地，不会进入 Git：

```text
dataset/
models/
outputs/
external_tools/
ACE-Step/
1761704195865-bk9wgc-tomori1_e12_s2664/
```

## 论文报告

智能语音处理论文报告位于：

```text
report/intelligent_speech_processing_paper/
```

常用文件：

- `ace_step_isp_lora_voice_conversion_paper.docx`
- `ace_step_isp_lora_voice_conversion_paper.pdf`
- `ace_step_isp_lora_voice_conversion_paper.tex`
- `paper_web.html`

重新生成报告：

```bash
python report/intelligent_speech_processing_paper/build_isp_paper.py
```

## 依赖说明

基础依赖见：

```text
requirements.txt
requirements-train.txt
```

部分功能需要额外准备：

- ACE-Step 基础模型
- Seed-VC 外部工具
- Demucs 分轨环境
- 本地授权目标音色样本或模型
- 训练数据集和生成输出目录

## 使用限制

本项目仅用于课程实验、科研原型和授权数据条件下的音频处理研究。涉及声音克隆、歌声音色转换或风格迁移时，应确保目标音色、训练数据和输入歌曲具有合法授权。
