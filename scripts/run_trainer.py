"""Wrapper script that patches torchaudio before running ACE-Step trainer.

Patches applied:
1. torchaudio.save/load -> soundfile (Windows compatibility)

All other fixes (CPU mode, cosine LR, warmup) are now in trainer.py directly.

Usage: python scripts/run_trainer.py <trainer_args...>
"""
import sys
import os

# Add scripts dir to path and import patch
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)
import torchaudio_patch  # noqa: F401 — patches torchaudio.save/load

# Resolve trainer path
project_root = os.path.dirname(scripts_dir)
trainer_path = os.path.join(project_root, "ACE-Step", "trainer.py")

if not os.path.exists(trainer_path):
    print(f"ERROR: ACE-Step trainer not found at {trainer_path}")
    sys.exit(1)

# Change working directory to ACE-Step (trainer expects this)
os.chdir(os.path.dirname(trainer_path))

# Add ACE-Step to path
sys.path.insert(0, os.path.dirname(trainer_path))

print(f"[run_trainer] Running ACE-Step trainer from {trainer_path}")

# Execute trainer.py directly — sys.argv is already set by train_edm_lora.py
with open(trainer_path, 'r', encoding='utf-8') as f:
    code = f.read()
exec(compile(code, trainer_path, 'exec'), {'__name__': '__main__', '__file__': trainer_path})
