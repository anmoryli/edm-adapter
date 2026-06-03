# Avicii Local Fine-Tune Architecture

## Goal

Train an Avicii-focused ACE-Step adapter on this Windows machine while keeping CPU usage low. The local environment currently has CPU-only PyTorch, so the default path is a constrained real LoRA fine-tune rather than full-model weight updates.

## Architecture

```text
Avicii clips + cached ACE-Step latents
        -> weighted sampler
        -> style trigger + rich EDM prompt prefix
        -> ACE-Step base transformer + avicii_style LoRA
        -> denoising loss on cached MusicDCAE latents
        -> LoRA bundle for generation
```

The adapter is trained inside the ACE-Step transformer. On CPU, only a small subset of LoRA tensors is trainable by default:

- last 1 transformer block LoRA
- `genre_embedder` / `speaker_embedder` / `t_block.1` LoRA
- `final_layer.linear` LoRA

This keeps backpropagation from traversing the whole 3.5B model while still changing the base model's generation path through a loadable LoRA adapter.

## Data

The existing dataset is already prepared:

- 6016 total 8-second clips
- 4833 train clips
- cached latents in `dataset/latents`
- HuggingFace train split in `outputs/datasets/edm_control_lora_train`

The sampler upweights high-energy drops and melodic/progressive/festival house material, because those parts usually carry the strongest Avicii-like traits: piano/pluck hooks, sidechain bass, bright supersaws, four-on-the-floor drums, and polished festival drops.

## Local CPU Training

Default low-CPU command:

```powershell
python scripts\train_avicii_local_lora.py `
  --cpu-threads 2 `
  --sample-size 512 `
  --train-last-n-blocks 1 `
  --max-steps 120 `
  --every-n-train-steps 40
```

Stronger but slower:

```powershell
python scripts\train_avicii_local_lora.py `
  --cpu-threads 4 `
  --sample-size 1024 `
  --train-last-n-blocks 2 `
  --max-steps 300 `
  --learning-rate 5e-4 `
  --every-n-train-steps 100
```

The trigger phrase written into training prompts is:

```text
avicii_adapter_style
```

## Generation

Use the latest saved adapter:

```powershell
python scripts\generate_avicii_local_lora.py `
  --duration 8 `
  --infer-step 30 `
  --lora-weight 1.6 `
  --prompt "uplifting progressive house, 128 BPM, bright piano chords, emotional melodic lead, sidechain bass, polished festival EDM drop"
```

The generation script automatically finds the latest bundle under:

```text
outputs/avicii_local_lora/logs
```

Only lightweight LoRA bundles are saved. Full Lightning `.ckpt` files are disabled because they are roughly 15 GB on this model and are not needed for generation.

## GPU Upgrade Path

If CUDA PyTorch is installed later, use the existing full EDM-StructLoRA trainer for broader controllability:

```powershell
python scripts\train_edm_control_lora.py `
  --dataset-path outputs\datasets\edm_control_lora_train `
  --checkpoint-dir models\ace-step\ACE-Step-v1-3.5B `
  --config config\edm_control_lora.json `
  --max-steps 10000 `
  --learning-rate 1e-4 `
  --precision bf16-mixed `
  --accumulate-grad-batches 8 `
  --devices 1
```

CPU can run the local LoRA path, but full router/control-token LoRA should be treated as a GPU job.
