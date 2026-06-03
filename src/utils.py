"""General utilities."""

import os
import logging
import yaml
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Configure logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def load_yaml(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str):
    """Save dict to YAML file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def get_device(preference: str = "auto") -> str:
    """Get compute device."""
    import torch
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def count_files(directory: str, ext: str = ".wav") -> int:
    """Count files with given extension in directory."""
    return sum(1 for f in os.listdir(directory) if f.endswith(ext))
