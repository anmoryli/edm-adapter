"""Adapter routing for structure- and attribute-aware EDM LoRA."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .taxonomy import DEFAULT_ENERGIES, DEFAULT_SECTIONS, DEFAULT_SUBGENRES


def safe_name(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("+", "plus")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


@dataclass
class RouterConfig:
    shared_adapter: str = "edm_shared"
    section_scale: float = 0.90
    energy_scale: float = 0.35
    subgenre_scale: float = 0.45
    bpm_scale: float = 0.20
    normalize: bool = False
    sections: list[str] = field(default_factory=lambda: DEFAULT_SECTIONS.copy())
    energies: list[str] = field(default_factory=lambda: DEFAULT_ENERGIES.copy())
    subgenres: list[str] = field(default_factory=lambda: DEFAULT_SUBGENRES.copy())

    @classmethod
    def from_dict(cls, data: dict | None) -> "RouterConfig":
        if not data:
            return cls()
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})


class EDMAdapterRouter:
    """Compute and apply adapter mixture weights from EDM metadata."""

    def __init__(self, config: RouterConfig | dict | None = None) -> None:
        if isinstance(config, RouterConfig):
            self.config = config
        else:
            self.config = RouterConfig.from_dict(config)

    def adapter_specs(self) -> list[dict]:
        specs = [{"name": self.config.shared_adapter, "kind": "shared", "value": "all"}]
        for section in self.config.sections:
            specs.append({"name": self.section_adapter(section), "kind": "section", "value": section})
        for energy in self.config.energies:
            specs.append({"name": self.energy_adapter(energy), "kind": "energy", "value": energy})
        for subgenre in self.config.subgenres:
            specs.append({"name": self.subgenre_adapter(subgenre), "kind": "subgenre", "value": subgenre})
        return specs

    def adapter_names(self) -> list[str]:
        return [spec["name"] for spec in self.adapter_specs()]

    def section_adapter(self, section: str) -> str:
        return f"section_{safe_name(section)}"

    def energy_adapter(self, energy: str) -> str:
        return f"energy_{safe_name(energy)}"

    def subgenre_adapter(self, subgenre: str) -> str:
        return f"subgenre_{safe_name(subgenre)}"

    def weights_for_batch(self, rows: Iterable[dict]) -> dict[str, float]:
        weights: dict[str, float] = {self.config.shared_adapter: 1.0}
        rows = list(rows)
        if not rows:
            return weights

        for row in rows:
            row_weight = float(row.get("sample_weight") or 1.0)
            tag_conf = row.get("tag_confidence") or {}
            section_conf = float(tag_conf.get("section", 0.65)) if isinstance(tag_conf, dict) else 0.65
            subgenre_conf = float(tag_conf.get("subgenre", 0.60)) if isinstance(tag_conf, dict) else 0.60
            energy_conf = float(tag_conf.get("energy", 0.65)) if isinstance(tag_conf, dict) else 0.65

            section = str(row.get("section") or "unknown")
            energy = str(row.get("energy") or "medium")
            subgenre = str(row.get("subgenre") or "")
            bpm = float(row.get("bpm") or 128.0)
            bpm_delta = min(1.0, abs(bpm - 128.0) / 60.0)
            bpm_boost = 1.0 + self.config.bpm_scale * bpm_delta

            self._add(weights, self.section_adapter(section), self.config.section_scale * section_conf * row_weight * bpm_boost)
            self._add(weights, self.energy_adapter(energy), self.config.energy_scale * energy_conf * row_weight * bpm_boost)
            if subgenre:
                self._add(weights, self.subgenre_adapter(subgenre), self.config.subgenre_scale * subgenre_conf * row_weight)

        denom = max(1, len(rows))
        weights = {name: value / denom for name, value in weights.items() if value > 0.0}
        if self.config.normalize:
            total = math.sqrt(sum(value * value for value in weights.values())) or 1.0
            weights = {name: value / total for name, value in weights.items()}
        return weights

    @staticmethod
    def _add(weights: dict[str, float], name: str, value: float) -> None:
        weights[name] = weights.get(name, 0.0) + float(value)

    def apply_to_model(self, model, weights: dict[str, float]) -> None:
        names = list(weights.keys())
        values = [float(weights[name]) for name in names]
        if hasattr(model, "set_adapters"):
            try:
                model.set_adapters(names, adapter_weights=values)
                return
            except TypeError:
                model.set_adapters(names, values)
                return
        try:
            from diffusers.utils.peft_utils import set_weights_and_activate_adapters
        except ImportError as exc:
            raise RuntimeError("diffusers is required to activate adapter routing") from exc
        set_weights_and_activate_adapters(model, names, values)


def save_router_manifest(path: str | Path, router: EDMAdapterRouter, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "edm_control_lora_bundle_v1",
        "router": router.config.__dict__,
        "adapters": router.adapter_specs(),
    }
    if extra:
        manifest.update(extra)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def load_router_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
