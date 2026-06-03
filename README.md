# EDM-Adapter

Electronic Music Domain-Adapted Text-to-Music Generation System.

Repository: https://github.com/anmoryli/edm-adapter

## Overview

EDM-Adapter fine-tunes open-source text-to-audio models (Stable Audio Open 1.0 / MusicGen) on electronic music datasets to generate better Techno, House, Trap, Ambient, Drum & Bass, and Future Bass short clips.

This repository contains the source code, configuration, web UI, scripts, and report artifacts needed to reproduce the project workflow. Local datasets, model weights, generated audio, checkpoints, external tool clones, and other large runtime artifacts are intentionally excluded from git.

## Quick Start

```bash
# 1. Setup environment
conda create -n edm-adapter python=3.10 -y
conda activate edm-adapter
pip install -r requirements.txt

# 2. Check environment
python scripts/00_check_env.py

# 3. Generate synthetic data for testing (no real dataset needed)
python scripts/03_segment_audio.py --synthetic --num-synthetic 100

# 4. Analyze audio features
python scripts/04_analyze_audio.py

# 5. Build captions
python scripts/05_build_captions.py

# 6. Split dataset
python scripts/06_split_dataset.py

# 7. Generate baseline audio
python scripts/07_baseline_generate.py --model musicgen

# 8. Fine-tune (small scale)
python scripts/08_finetune.py --model musicgen --max-samples 100

# 9. Generate with fine-tuned model
python scripts/09_generate_after_finetune.py --model musicgen

# 10. Evaluate
python scripts/10_evaluate.py

# 11. Generate report assets
python scripts/11_make_report_assets.py

# 12. Launch demo
python demo/app.py
```

## Full Pipeline with Real Data

```bash
# Download MTG-Jamendo or FMA dataset metadata
# Place in data/raw/mtg_jamendo/ or data/raw/fma/

# 1. Prepare metadata
python scripts/01_prepare_metadata.py

# 2. Filter electronic music
python scripts/02_filter_electronic.py

# 3. Segment audio (requires real audio files)
python scripts/03_segment_audio.py

# Continue from step 4 above...
```

## Project Structure

```
edm-adapter/
├── configs/          # YAML configuration files
├── data/             # Raw, interim, and processed data
├── scripts/          # Pipeline scripts (00-11)
├── src/              # Core modules
├── demo/             # Gradio web demo
├── outputs/          # Generated audio, checkpoints, comparisons
└── report/           # Technical report and presentation
```

## Scripts

| Script | Description |
|--------|-------------|
| 00_check_env.py | Check environment and dependencies |
| 01_prepare_metadata.py | Parse dataset metadata |
| 02_filter_electronic.py | Filter electronic music tracks |
| 03_segment_audio.py | Segment audio into clips |
| 04_analyze_audio.py | Extract audio features |
| 05_build_captions.py | Generate training captions |
| 06_split_dataset.py | Split into train/val/test |
| 07_baseline_generate.py | Generate baseline audio |
| 08_finetune.py | Fine-tune model |
| 09_generate_after_finetune.py | Generate with fine-tuned model |
| 10_evaluate.py | Evaluate and compare |
| 11_make_report_assets.py | Generate charts and stats |

## Models

- **Primary**: Stable Audio Open 1.0 (Stability AI)
- **Fallback**: MusicGen small (Meta)

## License

This project is for educational and research purposes.
