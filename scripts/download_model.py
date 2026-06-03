"""Download or migrate a Hugging Face model into the project-local models directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_store import (  # noqa: E402
    configure_hf_environment,
    get_local_model_dir,
    local_model_exists,
    repo_id_to_dirname,
)


DEFAULT_MODELS = [
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
    "facebook/musicgen-large",
]


def get_global_hf_snapshot(model_id: str) -> Path | None:
    """Return a cached Hugging Face snapshot directory when available."""
    repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = []
    for snapshot in snapshots_dir.iterdir():
        if not snapshot.is_dir():
            continue
        if (snapshot / "config.json").exists():
            weight_name = "model.safetensors" if (snapshot / "model.safetensors").exists() else "pytorch_model.bin"
            if (snapshot / weight_name).exists():
                candidates.append(snapshot)

    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def copy_snapshot(snapshot_dir: Path, target_dir: Path):
    """Copy a cached snapshot into the project-local model directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in snapshot_dir.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def download_with_hf_hub(model_id: str, target_dir: Path):
    """Download a model directly into the project-local directory."""
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )


def ensure_model(model_id: str):
    """Ensure a model exists inside ./models."""
    configure_hf_environment()
    target_dir = get_local_model_dir(model_id)

    if local_model_exists(model_id):
        print(f"{model_id} already available at {target_dir}")
        return

    snapshot_dir = get_global_hf_snapshot(model_id)
    if snapshot_dir is not None:
        print(f"Copying cached snapshot for {model_id} -> {target_dir}")
        copy_snapshot(snapshot_dir, target_dir)
        return

    print(f"Downloading {model_id} -> {target_dir}")
    download_with_hf_hub(model_id, target_dir)


def main():
    parser = argparse.ArgumentParser(description="Download models into ./models")
    parser.add_argument("models", nargs="*", help="Hugging Face model ids")
    parser.add_argument("--all", action="store_true", help="Download default MusicGen models")
    args = parser.parse_args()

    model_ids = args.models
    if args.all or not model_ids:
        model_ids = DEFAULT_MODELS if args.all else ["facebook/musicgen-medium"]

    print("Project-local model storage:")
    for model_id in model_ids:
        print(f"  - {model_id} -> models/{repo_id_to_dirname(model_id)}")

    for model_id in model_ids:
        ensure_model(model_id)

    print("Done.")


if __name__ == "__main__":
    main()
