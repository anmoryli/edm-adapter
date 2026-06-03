"""Project-local model storage helpers."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"
HF_ROOT = MODELS_ROOT / "huggingface"

MODEL_DIR_NAMES = {
    "facebook/musicgen-small": "musicgen-small",
    "facebook/musicgen-medium": "musicgen-medium",
    "facebook/musicgen-large": "musicgen-large",
    "stabilityai/stable-audio-open-1.0": "stable-audio-open-1.0",
}


def ensure_models_root() -> Path:
    """Create the project-local model directories."""
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    HF_ROOT.mkdir(parents=True, exist_ok=True)
    return MODELS_ROOT


def get_project_root() -> Path:
    """Return the repository root."""
    return PROJECT_ROOT


def get_models_root() -> Path:
    """Return the project-local models directory."""
    ensure_models_root()
    return MODELS_ROOT


def get_hf_cache_root() -> Path:
    """Return the project-local Hugging Face cache directory."""
    ensure_models_root()
    return HF_ROOT


def configure_hf_environment() -> dict[str, str]:
    """Point Hugging Face caches at the project-local models directory."""
    cache_root = get_hf_cache_root()
    hub_root = cache_root / "hub"
    transformers_root = cache_root / "transformers"
    hub_root.mkdir(parents=True, exist_ok=True)
    transformers_root.mkdir(parents=True, exist_ok=True)

    env_vars = {
        "HF_HOME": str(cache_root),
        "HF_HUB_CACHE": str(hub_root),
        "HUGGINGFACE_HUB_CACHE": str(hub_root),
        "TRANSFORMERS_CACHE": str(transformers_root),
    }
    for key, value in env_vars.items():
        os.environ[key] = value
    return env_vars


def repo_id_to_dirname(repo_id: str) -> str:
    """Convert a Hugging Face repo id into a local directory name."""
    if repo_id in MODEL_DIR_NAMES:
        return MODEL_DIR_NAMES[repo_id]
    return repo_id.replace("/", "--")


def get_local_model_dir(model_name_or_path: str) -> Path:
    """Return the preferred local storage directory for a model."""
    return get_models_root() / repo_id_to_dirname(model_name_or_path)


def local_model_exists(model_name_or_path: str) -> bool:
    """Return True when the model has already been exported locally."""
    model_dir = get_local_model_dir(model_name_or_path)
    return (model_dir / "config.json").exists() and any(
        (model_dir / candidate).exists()
        for candidate in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json")
    )


def resolve_model_source(model_name_or_path: str) -> str:
    """Resolve a model id to a project-local directory when available."""
    explicit_path = Path(model_name_or_path)
    if explicit_path.exists():
        return str(explicit_path)

    local_dir = get_local_model_dir(model_name_or_path)
    if local_model_exists(model_name_or_path):
        return str(local_dir)

    return model_name_or_path
