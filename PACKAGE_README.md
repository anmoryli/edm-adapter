# EDM-Adapter 最小运行包说明

这是面向演示和复现的最小运行包，不包含训练数据集、历史输出、完整模型缓存和临时日志。

## 启动

```powershell
pip install -r requirements.txt
python demo\app.py
```

打开 `http://127.0.0.1:7860`。

## 已保留

- `demo/`、`src/`、`scripts/`、`config/`：网页、音频处理、Seed-VC 桥接和工具代码。
- `ACE-Step/`：ACE-Step 推理源码。
- `external_tools/seed-vc/`：Seed-VC 源码和配置，不含 Hugging Face 模型缓存。
- `1761704195865-bk9wgc-tomori1_e12_s2664/参考/`：授权音色转换用参考音频。
- `outputs/avicii_local_lora/.../step=960_avicii_local_lora/`：最新 LoRA adapter。
- `true_music.jpg`、`gsd.mp4`：网页展示资产。
- `report/intelligent_speech_processing_paper/paper_web.html`：网页技术报告。

## 未打包

- `dataset/`：原始和处理后的训练数据。
- `models/`：ACE-Step 基础权重，本地体积约 10 GB，首次生成时按原逻辑下载。
- `external_tools/seed-vc/checkpoints/`：Seed-VC、Whisper、BigVGAN 等缓存，本地体积约 2.3 GB，首次音色转换时按原逻辑下载。
- 历史 `outputs/web_generations/`、训练日志、缓存、`node_modules/`、`__pycache__/`。

## 说明

如果要离线运行，需要另外把 `models/ace-step/ACE-Step-v1-3.5B/` 和 `external_tools/seed-vc/checkpoints/` 拷回相同相对路径。
