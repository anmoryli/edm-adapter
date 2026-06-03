"""ACE-Step dataset extension with cached latents, controls, and routing metadata."""

from __future__ import annotations

import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torchaudio
from loguru import logger

from src.edm_control.metadata_utils import resolve_dataset_path


class ControlText2MusicDatasetMixin:
    """Mixin for ACE-Step Text2MusicDataset.

    The concrete class is built at runtime after ACE-Step is on sys.path.
    """

    project_root = Path(__file__).resolve().parents[2]

    def get_audio(self, item):
        """Load audio through soundfile to avoid torchcodec issues in newer torchaudio."""

        import soundfile as sf

        filename_value = item["filename"]
        filename = Path(filename_value)
        if not filename.is_absolute():
            candidates = [
                self.project_root / filename_value,
                self.project_root / "dataset" / filename_value,
                Path(self.train_dataset_path).parent / filename_value,
            ]
            filename = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        try:
            audio_np, sr = sf.read(filename, always_2d=True, dtype="float32")
        except Exception as exc:
            logger.error(f"Failed to load audio {filename}: {exc}")
            return None

        audio_np = np.asarray(audio_np).T
        audio = torch.from_numpy(audio_np)
        if audio.shape[0] == 1:
            audio = torch.cat([audio, audio], dim=0)
        audio = audio[:2]
        if sr != 48000:
            audio = torchaudio.transforms.Resample(sr, 48000)(audio)
        audio = torch.clamp(audio, -1.0, 1.0)
        if audio.shape[-1] < 48000 * 3:
            audio = torch.nn.functional.pad(audio, (0, 48000 * 3 - audio.shape[-1]))
        silent_ratio = torch.mean(torch.all(audio == 0, dim=0).float()).item()
        if silent_ratio > 0.95:
            logger.error(f"Silent audio {filename}")
            return None
        return audio

    def tokenize_lyrics_map(self, item, debug=False):
        item["norm_lyrics"] = item.get("norm_lyrics") or "[instrumental]"
        item["lyric_token_idx"] = [0]
        return item

    def _resolve_asset_path(self, item: dict, field: str) -> Path | None:
        value = item.get(field)
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        candidates = [
            self.project_root / value,
            self.project_root / "dataset" / value,
            Path(self.train_dataset_path).parent / value,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_latent(self, item: dict) -> torch.Tensor | None:
        path = self._resolve_asset_path(item, "latent_path")
        if path is None or not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        latent = payload["latent"] if isinstance(payload, dict) else payload
        return latent.float()

    def _load_control(self, item: dict) -> torch.Tensor | None:
        path = self._resolve_asset_path(item, "control_path")
        if path is None or not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        control = payload["control"] if isinstance(payload, dict) else payload
        return control.float()

    def _route_metadata(self, item: dict) -> dict:
        return {
            "clip_id": item.get("clip_id") or item.get("keys"),
            "section": item.get("section", "unknown"),
            "energy": item.get("energy", "medium"),
            "subgenre": item.get("subgenre", "unknown"),
            "bpm": float(item.get("bpm") or 128.0),
            "bpm_confidence": float(item.get("bpm_confidence") or 0.0),
            "tag_confidence": item.get("tag_confidence") or {},
            "quality_score": int(item.get("quality_score") or 4),
            "sample_weight": float(item.get("sample_weight") or 1.0),
        }

    def process(self, item):
        examples = super().process(item)
        if not examples:
            return examples
        latent = self._load_latent(item)
        control = self._load_control(item)
        route_metadata = self._route_metadata(item)
        sample_weight = torch.tensor(route_metadata["sample_weight"], dtype=torch.float32)

        for example in examples:
            if latent is not None:
                example["target_latent"] = latent
                example["latent_length"] = latent.shape[-1]
            if control is not None:
                example["control_curve"] = control
                example["control_length"] = control.shape[0]
            example["sample_weight"] = sample_weight
            example["route_metadata"] = route_metadata
        return examples

    def get_full_features(self, idx):
        examples = {
            "keys": [],
            "target_wavs": [],
            "vocal_wavs": [],
            "wav_lengths": [],
            "structured_tags": [],
            "prompts": [],
            "speaker_embs": [],
            "lyric_token_ids": [],
            "lyric_masks": [],
            "candidate_lyric_chunks": [],
            "target_latents": [],
            "latent_lengths": [],
            "control_curves": [],
            "control_lengths": [],
            "sample_weights": [],
            "route_metadatas": [],
        }

        item = self.pretrain_ds[idx]
        item["idx"] = idx
        item = self.tokenize_lyrics_map(item)
        features = self.process(item)

        if features:
            for feature in features:
                for key, value in feature.items():
                    target_key = key + "s"
                    if key == "key":
                        target_key = "keys"
                    elif key == "wav_length":
                        target_key = "wav_lengths"
                    elif key == "candidate_lyric_chunk":
                        target_key = "candidate_lyric_chunks"
                    if value is not None and target_key in examples:
                        examples[target_key].append(value)
        return examples

    def collate_fn(self, batch):
        batch = self.pack_batch(batch)
        output = {}

        for key, value in batch.items():
            if key in [
                "keys",
                "structured_tags",
                "prompts",
                "candidate_lyric_chunks",
                "route_metadatas",
            ]:
                padded = value
            elif key in ["wav_lengths", "latent_lengths", "control_lengths"]:
                padded = torch.LongTensor(value)
            elif key in ["src_wavs", "target_wavs", "vocal_wavs"]:
                max_length = max(seq.shape[1] for seq in value)
                padded = torch.stack(
                    [torch.nn.functional.pad(seq, (0, max_length - seq.shape[1])) for seq in value]
                )
            elif key == "target_latents":
                max_length = max(seq.shape[-1] for seq in value)
                padded = torch.stack(
                    [torch.nn.functional.pad(seq, (0, max_length - seq.shape[-1])) for seq in value]
                )
            elif key == "control_curves":
                max_length = max(seq.shape[0] for seq in value)
                padded = torch.stack(
                    [torch.nn.functional.pad(seq, (0, 0, 0, max_length - seq.shape[0])) for seq in value]
                )
                output["control_masks"] = torch.stack(
                    [
                        torch.nn.functional.pad(torch.ones(seq.shape[0]), (0, max_length - seq.shape[0]))
                        for seq in value
                    ]
                )
            elif key == "speaker_embs":
                padded = torch.stack(value)
            elif key == "sample_weights":
                padded = torch.stack(value).float()
            elif key in [
                "chunk_masks",
                "clap_attention_masks",
                "lyric_token_ids",
                "lyric_masks",
            ]:
                max_length = max(len(seq) for seq in value)
                padded = torch.stack(
                    [torch.nn.functional.pad(seq, (0, max_length - len(seq))) for seq in value]
                )
            else:
                padded = value
            output[key] = padded

        return output

    def __getitem__(self, idx):
        try:
            example = self.get_full_features(idx)
            if len(example["keys"]) == 0:
                raise RuntimeError(f"empty example idx={idx}")
            return example
        except Exception as exc:
            logger.error(f"Error in getting item {idx}: {exc}")
            traceback.print_exc()
            new_idx = random.choice(range(len(self)))
            return self.__getitem__(new_idx)


def build_control_dataset_class():
    from acestep.text2music_dataset import Text2MusicDataset

    class EDMControlText2MusicDataset(ControlText2MusicDatasetMixin, Text2MusicDataset):
        def __init__(
            self,
            train=True,
            train_dataset_path=None,
            max_duration=240.0,
            sample_size=None,
            shuffle=True,
            minibatch_size=1,
        ):
            self.train_dataset_path = train_dataset_path
            self.max_duration = max_duration
            self.minibatch_size = minibatch_size
            self.train = train
            self.setup_full(train, shuffle, sample_size)
            logger.info(f"Dataset size: {len(self)} total {self.total_samples} samples")

    return EDMControlText2MusicDataset
