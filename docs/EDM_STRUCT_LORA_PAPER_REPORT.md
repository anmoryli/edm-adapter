# EDM-StructLoRA: Structure- and Attribute-Aware LoRA Routing for Controllable EDM Generation

**Research Prototype Report**  
**Project:** EDM-Adapter / ACE-Step EDM Adaptation

## Abstract

This report presents EDM-StructLoRA, a parameter-efficient adaptation framework for ACE-Step-based electronic dance music generation. Instead of training a single fixed LoRA, the method combines a shared EDM adapter, section-specific experts, attribute experts, and a metadata-conditioned router. A latent-aligned control curve injects BPM, beat phase, energy, low-frequency ratio, onset density, loop boundary, and confidence signals as additional text-conditioning tokens. The current implementation prepares 6,016 cleaned EDM clips, cached ACE-Step latents, control curves, and portable HuggingFace datasets for GPU training.

**Keywords:** text-to-music generation; LoRA; controllable music generation; EDM; diffusion transformer; ACE-Step.

## 1. Introduction

Plain LoRA fine-tuning can adapt a base text-to-music model to an EDM dataset, but it does not explicitly model EDM structure. This report focuses on whether section, BPM, energy, subgenre, bass behavior, and loop structure can be controlled more reliably than plain LoRA baselines.

EDM-StructLoRA treats EDM metadata as part of the adaptation mechanism. Section and attribute labels select LoRA expert mixtures, while frame-level control curves inject time-varying rhythmic and spectral attributes.

## 2. Method

**Fig. 1. Overview of EDM-StructLoRA.**  
The web report renders the method diagram automatically from `src/reports/paper_report.py`.

**Table 1. Method components.**

| Component | Implementation | Purpose |
|---|---|---|
| Shared LoRA | r=32, rsLoRA, attention target modules | Capture EDM domain shift shared by all clips. |
| Section experts | r=8 experts for intro/build-up/drop/breakdown/outro/loop | Model section-specific generation offsets. |
| Attribute experts | energy and subgenre experts | Improve control over energy and EDM substyle. |
| Adapter router | section, energy, subgenre, BPM, confidence | Activate parameter-efficient experts dynamically. |
| Control conditioner | T x 36 control curve to 8 text-conditioning tokens | Inject beat phase, low-frequency, onset, loop, and confidence signals. |

## 3. Dataset

**Table 2. Dataset summary.**

| Item | Value |
|---|---:|
| Raw source files | 231 |
| Processed source files | 203 |
| Training-quality clips | 6016 |
| Train / validation / test | 4833 / 557 / 626 |
| Clip duration | 8 s |
| ACE-Step latents | 6016 |
| Control curves | 6016 |

**Fig. 2. Dataset label distribution.**  
The web report renders subgenre, section, and energy distributions from `dataset/reports/label_statistics.json`.

**Fig. 3. Latent-aligned EDM control curve.**  
The web report renders a sampled control curve from `dataset/controls/*.pt`.

## 4. Experimental Protocol

**Table 3. Ablation design.**

| ID | System | Purpose |
|---|---|---|
| A0 | ACE-Step base | Unadapted text-to-music baseline. |
| A1 | Plain LoRA r=64 | Parameter-efficient domain adaptation baseline. |
| A2 | Metadata-rich caption LoRA | Tests whether caption engineering alone is sufficient. |
| A3 | Section-aware LoRA | Isolates the benefit of structure-specific adapters. |
| A4 | Section + attribute routed LoRA | Tests dynamic adapter routing. |
| A5 | Full EDM-StructLoRA | Router + control tokens + confidence-aware sampling. |

**Table 4. Evaluation metrics.**

| Metric | Definition |
|---|---|
| BPM error | Absolute difference between target BPM and estimated generated BPM. |
| Onset density error | Difference between target and generated onset density. |
| Low-frequency ratio | Energy share in the bass band. |
| Loop similarity | Head-tail spectral or embedding similarity. |
| FAD / embedding distance | Distribution distance to reference EDM data. |
| Human blind test | Style match, drop impact, loop usability, and audio quality. |

## 5. Reproducibility

```bash
python scripts/build_edm_control_assets.py --dataset-root dataset
python scripts/prepare_ace_control_dataset.py --dataset-root dataset --split train --output outputs/datasets/edm_control_lora_train --path-mode project-relative
python scripts/train_edm_control_lora.py --dataset-path outputs/datasets/edm_control_lora_train --checkpoint-dir models/ace-step/ACE-Step-v1-3.5B --config config/edm_control_lora.json --max-steps 10000 --learning-rate 1e-4 --precision bf16-mixed --accumulate-grad-batches 8 --devices 1
```

## 6. References

1. ACE-Step: A Step Towards Music Generation Foundation Model.
2. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models.
3. Music ControlNet: Multiple Time-varying Controls for Music Generation.
4. MusiConGen: Rhythm and Chord Control for Transformer-Based Text-to-Music Generation.
5. MuseControlLite: Lightweight Control for Music Generation.
6. DoRA, rsLoRA, AdaLoRA, and X-LoRA as parameter-efficient adaptation baselines.
