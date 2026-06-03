# EDM-Adapter 答辩 PPT 提纲

## Slide 1: 封面

- 题目：EDM-Adapter：基于电子音乐数据集微调的文本到音乐生成系统
- 姓名/学号/日期

## Slide 2: 项目背景

- 通用音乐生成模型对电子音乐细分风格控制不足
- 电子音乐制作中 loop/sample/drum beat 短素材需求量大
- 现有模型在低频律动、鼓组质感、风格细分方面有待提升

## Slide 3: 项目目标

- 构建电子音乐领域数据集
- 实现自动 caption 构造
- 基于 Stable Audio Open / MusicGen 进行领域微调
- 对比微调前后效果
- 提供可交互的 Web Demo

## Slide 4: 数据来源

- MTG-Jamendo: 55,000+ 首音频, 195 个标签
- FMA: 106,574 首 CC 授权音频
- 筛选标签: electronic, techno, house, ambient, trap, drum and bass

## Slide 5: 数据处理流程

```
原始音频 → 格式统一 → 标签筛选 → 10秒切片 → 质量过滤
        → 特征提取 (BPM/RMS/低频) → Caption 自动生成
        → 数据集划分
```

## Slide 6: Caption 构造

- 模板: `[BPM] BPM [能量] [情绪] [风格] loop with [乐器], [氛围] atmosphere`
- 示例:
  - "128 BPM high energy dark techno loop with heavy kick drum, deep bass, metallic hi-hats"
  - "124 BPM danceable warm house loop with warm synth chords, groovy bassline"

## Slide 7: 模型方法

- 主模型: Stable Audio Open 1.0 (44.1kHz, 最长 47 秒)
- 备选: MusicGen small
- 微调策略: 小规模试跑 → 正式训练
- 训练配置: lr=1e-5, batch=4, fp16

## Slide 8: 系统演示

- Gradio Web Demo 截图
- 功能: 选择风格/BPM/情绪/时长 → 生成音频 → 波形/频谱/分析
- 支持 Baseline / Finetuned 模型切换

## Slide 9: 实验对比 - 自动指标

| 指标 | Baseline | Finetuned |
|------|----------|-----------|
| BPM Error | | |
| Low Freq Ratio | | |
| Onset Density | | |
| Loop Similarity | | |

(插入对比图表)

## Slide 10: 实验对比 - 听感评估

- 5 维度评分: 风格匹配、鼓点稳定、Bass 表现、电子质感、Loop 可用性
- 评分结果对比

## Slide 11: 创新点

1. 面向电子音乐领域的生成模型适配
2. 结构化音乐 Caption 自动构造
3. 电子音乐短片段 (loop/beat/texture) 生成
4. 多维度微调前后对比评估
5. 可交互 Gradio 生成系统

## Slide 12: 总结与展望

### 总结
- 成功构建电子音乐领域微调系统
- 微调后模型在风格匹配、BPM 准确性方面提升
- 提供完整的数据处理-训练-评估-展示流水线

### 展望
- 扩大训练数据 (3000-10000 片段)
- 增加更多子风格
- 探索 LoRA 参数高效微调
- 支持 BPM/Key 精确控制
- 支持更长时长生成
