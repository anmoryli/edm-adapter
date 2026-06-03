"""Build latent-aligned EDM control curves from metadata."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .taxonomy import (
    DEFAULT_ENERGIES,
    DEFAULT_SECTIONS,
    DEFAULT_SUBGENRES,
    ENERGY_TO_VALUE,
)


@dataclass(frozen=True)
class ControlCurveConfig:
    frame_count: int = 87
    bpm_min: float = 60.0
    bpm_max: float = 180.0
    onset_density_max: float = 30.0
    low_freq_ratio_max: float = 0.35

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        names += [f"section:{name}" for name in DEFAULT_SECTIONS]
        names += [f"subgenre:{name}" for name in DEFAULT_SUBGENRES]
        names += [f"energy:{name}" for name in DEFAULT_ENERGIES]
        names += [
            "energy_value",
            "bpm_norm",
            "bpm_confidence",
            "beat_phase_sin",
            "beat_phase_cos",
            "time_position",
            "low_freq_ratio_norm",
            "onset_density_norm",
            "loop_start_marker",
            "loop_end_marker",
            "tag_confidence_mean",
            "quality_weight",
        ]
        return names

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)


def infer_frame_count(row: dict, default: int = 87) -> int:
    latent_config = row.get("latent_config") or {}
    latent_shape = latent_config.get("latent_shape") or []
    if len(latent_shape) >= 3:
        return int(latent_shape[-1])
    latent_length = latent_config.get("latent_length")
    if latent_length:
        return int(latent_length) + 1
    return default


def _one_hot(value: str, vocab: list[str]) -> list[float]:
    value = value or "unknown"
    return [1.0 if value == item else 0.0 for item in vocab]


def _tag_confidence_mean(row: dict) -> float:
    tag_conf = row.get("tag_confidence") or {}
    if not isinstance(tag_conf, dict) or not tag_conf:
        return 0.0
    return float(sum(float(v) for v in tag_conf.values()) / len(tag_conf))


def build_control_curve(row: dict, config: ControlCurveConfig | None = None) -> tuple[torch.Tensor, dict]:
    """Create a [frames, features] control tensor for one metadata row."""

    if config is None:
        config = ControlCurveConfig(frame_count=infer_frame_count(row))

    frames = config.frame_count
    duration = max(float(row.get("duration") or 8.0), 1e-3)
    section = str(row.get("section") or "unknown")
    subgenre = str(row.get("subgenre") or "unknown")
    energy = str(row.get("energy") or "medium")
    bpm = float(row.get("bpm") or 128.0)
    bpm_norm = (bpm - config.bpm_min) / (config.bpm_max - config.bpm_min)
    bpm_norm = max(0.0, min(1.0, bpm_norm))
    energy_value = ENERGY_TO_VALUE.get(energy, ENERGY_TO_VALUE["medium"])
    audio_features = row.get("audio_features") or {}
    low_ratio = float(audio_features.get("low_freq_ratio") or 0.0)
    onset_density = float(audio_features.get("onset_density") or 0.0)
    low_norm = max(0.0, min(1.0, low_ratio / config.low_freq_ratio_max))
    onset_norm = max(0.0, min(1.0, onset_density / config.onset_density_max))
    bpm_conf = max(0.0, min(1.0, float(row.get("bpm_confidence") or 0.0)))
    tag_conf = _tag_confidence_mean(row)
    quality_weight = max(0.0, min(1.0, float(row.get("quality_score") or 4.0) / 5.0))

    section_vec = _one_hot(section, DEFAULT_SECTIONS)
    subgenre_vec = _one_hot(subgenre, DEFAULT_SUBGENRES)
    energy_vec = _one_hot(energy, DEFAULT_ENERGIES)

    rows: list[list[float]] = []
    beat_hz = bpm / 60.0
    for i in range(frames):
        pos = 0.0 if frames <= 1 else i / (frames - 1)
        t = pos * duration
        phase = 2.0 * math.pi * beat_hz * t
        loop_start = 1.0 if i == 0 else 0.0
        loop_end = 1.0 if i == frames - 1 else 0.0
        feature = []
        feature += section_vec
        feature += subgenre_vec
        feature += energy_vec
        feature += [
            energy_value,
            bpm_norm,
            bpm_conf,
            math.sin(phase),
            math.cos(phase),
            pos,
            low_norm,
            onset_norm,
            loop_start,
            loop_end,
            tag_conf,
            quality_weight,
        ]
        rows.append(feature)

    tensor = torch.tensor(rows, dtype=torch.float32)
    meta = {
        "feature_names": config.feature_names,
        "feature_dim": config.feature_dim,
        "frame_count": frames,
        "duration": duration,
        "section": section,
        "subgenre": subgenre,
        "energy": energy,
        "bpm": bpm,
    }
    return tensor, meta


def save_schema(path: str | Path, config: ControlCurveConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "name": "edm_latent_aligned_control_curve_v1",
        "frame_count_default": config.frame_count,
        "feature_dim": config.feature_dim,
        "feature_names": config.feature_names,
        "description": (
            "Latent-frame controls for section, subgenre, energy, BPM beat phase, "
            "low-frequency ratio, onset density, loop boundary markers, and confidence."
        ),
    }
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
