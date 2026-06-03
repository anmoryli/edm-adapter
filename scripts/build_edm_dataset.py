"""Build a clean EDM text-to-music dataset.

This script keeps dataset/raw_audio intact, then rebuilds processed audio, clips,
mel spectrograms, metadata, song-level splits, and reports.
All metadata paths are POSIX-style relative paths so the dataset can run on Linux.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_ROOT / "raw_audio"
PROCESSED_DIR = DATASET_ROOT / "processed_audio"
CLIPS_DIR = DATASET_ROOT / "clips"
MELS_DIR = DATASET_ROOT / "mels"
LATENTS_DIR = DATASET_ROOT / "latents"
TOKENS_DIR = DATASET_ROOT / "tokens"
SPLITS_DIR = DATASET_ROOT / "splits"
REPORTS_DIR = DATASET_ROOT / "reports"

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aiff", ".aif"}

TARGET_SR = 44100
TARGET_CHANNELS = 2
CLIP_SECONDS = 8.0
HOP_SECONDS = 8.0
MIN_SOURCE_SECONDS = 8.0
MIN_RMS = 0.012
MAX_CLIPPING_RATIO = 0.04
TARGET_RMS = 0.10
PEAK_LIMIT = 0.98

MEL_SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160

RANDOM_SEED = 20260518

SUBGENRES = [
    "progressive house",
    "melodic house",
    "festival EDM",
    "future bass",
    "electro house",
    "tropical house",
    "piano house",
    "folk EDM",
    "big room house",
    "deep house",
    "dance pop",
]

MOODS = [
    "uplifting",
    "emotional",
    "melancholic",
    "dreamy",
    "energetic",
    "dark",
    "aggressive",
    "nostalgic",
    "euphoric",
    "relaxing",
    "tense",
    "hopeful",
]

SECTIONS = ["intro", "build-up", "drop", "breakdown", "bridge", "chorus", "outro", "loop", "unknown"]

INSTRUMENT_VOCAB = [
    "kick",
    "clap",
    "snare",
    "hi-hat",
    "cymbal",
    "bass",
    "sub bass",
    "sidechain bass",
    "piano",
    "acoustic guitar",
    "electric guitar",
    "pluck synth",
    "supersaw synth",
    "pad",
    "lead synth",
    "vocal chop",
    "strings",
    "brass",
    "fx riser",
    "white noise sweep",
]

PRODUCTION_VOCAB = [
    "sidechain compression",
    "wide stereo",
    "bright mix",
    "warm mix",
    "heavy reverb",
    "dry mix",
    "punchy drums",
    "clean master",
    "lo-fi texture",
    "distorted bass",
]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class RemovedItem:
    item_type: str
    path: str
    reason: str
    detail: str = ""


def posix_rel(path: Path, base: Path = DATASET_ROOT) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def slugify(value: str, fallback: str = "audio") -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def reset_generated_dirs() -> None:
    for path in [PROCESSED_DIR, CLIPS_DIR, MELS_DIR, LATENTS_DIR, TOKENS_DIR, SPLITS_DIR, REPORTS_DIR]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for path in [DATASET_ROOT / "metadata.jsonl", DATASET_ROOT / "metadata.csv"]:
        if path.exists():
            path.unlink()


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, CLIPS_DIR, MELS_DIR, LATENTS_DIR, TOKENS_DIR, SPLITS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=None, mono=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    if y.shape[0] > y.shape[1]:
        # librosa should return channels x samples; this guard handles odd decoders.
        y = y.T
    return y, int(sr)


def to_stereo(y: np.ndarray) -> np.ndarray:
    if y.shape[0] == 1:
        return np.repeat(y, 2, axis=0)
    if y.shape[0] >= 2:
        return y[:2]
    return np.repeat(y.reshape(1, -1), 2, axis=0)


def normalize_audio(y: np.ndarray) -> tuple[np.ndarray, bool, float]:
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    peak = float(np.max(np.abs(y)) + 1e-12)
    gain = min(TARGET_RMS / rms, PEAK_LIMIT / peak)
    if not np.isfinite(gain) or gain <= 0:
        gain = 1.0
    return (y * gain).astype(np.float32), True, float(gain)


def save_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.T, sr, subtype="PCM_16")


def estimate_key(y_mono: np.ndarray, sr: int) -> tuple[str, float]:
    # Key detection is intentionally conservative. A weak heuristic key label is
    # worse than unknown for text-to-music training metadata.
    return "unknown", 0.0


def analyze_clip(y: np.ndarray, sr: int) -> dict[str, Any]:
    y_mono = np.mean(y, axis=0).astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(y_mono))) + 1e-12)
    peak = float(np.max(np.abs(y_mono)) + 1e-12)
    clipping_ratio = float(np.mean(np.abs(y_mono) >= 0.985))
    duration = float(len(y_mono) / sr)

    try:
        tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
        bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0
    except Exception:
        bpm = 0.0
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 0.0

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y_mono, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y_mono)))
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
    onset_density = float(len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / max(duration, 1e-6))

    stft = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = float(np.sum(stft[freqs <= 180]))
    total = float(np.sum(stft) + 1e-12)
    low_freq_ratio = low / total

    key, key_conf = estimate_key(y_mono, sr)

    return {
        "bpm": round(bpm) if bpm else 0,
        "bpm_confidence": 0.72 if bpm else 0.0,
        "key": key,
        "key_confidence": key_conf,
        "rms": rms,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "zero_crossing_rate": zcr,
        "onset_density": onset_density,
        "low_freq_ratio": low_freq_ratio,
    }


def infer_energy(features: dict[str, Any]) -> tuple[str, float]:
    rms = features["rms"]
    onset = features["onset_density"]
    centroid = features["spectral_centroid"]
    score = 0
    score += 2 if rms > 0.09 else 1 if rms > 0.045 else 0
    score += 2 if onset > 3.4 else 1 if onset > 1.8 else 0
    score += 1 if centroid > 2400 else 0
    if score >= 5:
        return "very_high", 0.78
    if score >= 3:
        return "high", 0.75
    if score >= 1:
        return "medium", 0.68
    return "low", 0.72


def infer_subgenre(features: dict[str, Any], energy: str) -> tuple[str, float]:
    bpm = features["bpm"] or 0
    centroid = features["spectral_centroid"]
    low = features["low_freq_ratio"]

    if 122 <= bpm <= 132:
        if energy in {"high", "very_high"} and centroid > 2300:
            return "festival EDM", 0.70
        if low > 0.28:
            return "progressive house", 0.74
        return "melodic house", 0.70
    if 116 <= bpm < 122:
        return "dance pop", 0.62
    if 132 < bpm <= 138:
        return "big room house", 0.66
    if 138 < bpm <= 155:
        return "future bass", 0.62
    if bpm > 155:
        return "electro house", 0.58
    if 90 <= bpm < 116:
        return "tropical house", 0.56
    if bpm and bpm < 90:
        return "deep house", 0.52
    return "progressive house", 0.45


def infer_section(source_duration: float, start_time: float, features: dict[str, Any], energy: str) -> tuple[str, float]:
    rel = start_time / max(source_duration, 1e-6)
    onset = features["onset_density"]
    if rel < 0.10 and energy in {"low", "medium"}:
        return "intro", 0.70
    if rel > 0.88:
        return "outro", 0.68
    if energy in {"high", "very_high"} and onset > 2.6:
        return "drop", 0.70
    if onset > 3.2 and energy == "medium":
        return "build-up", 0.58
    if energy == "low":
        return "breakdown", 0.58
    return "loop", 0.55


def infer_mood(subgenre: str, energy: str, features: dict[str, Any]) -> tuple[list[str], float]:
    centroid = features["spectral_centroid"]
    low = features["low_freq_ratio"]
    moods: list[str] = []
    if subgenre in {"progressive house", "melodic house", "festival EDM", "piano house"}:
        moods.extend(["uplifting", "euphoric"])
    if subgenre in {"deep house", "tropical house", "dance pop"}:
        moods.extend(["relaxing", "hopeful"])
    if energy in {"high", "very_high"}:
        moods.append("energetic")
    if centroid < 1800:
        moods.append("dreamy")
    if low > 0.34 and centroid < 2200:
        moods.append("emotional")
    if not moods:
        moods.append("uplifting")
    deduped = []
    for mood in moods:
        if mood in MOODS and mood not in deduped:
            deduped.append(mood)
    return deduped[:3], 0.62


def infer_instruments_and_production(subgenre: str, energy: str, features: dict[str, Any]) -> tuple[list[str], list[str], str, list[str], dict[str, float]]:
    instruments = ["kick", "bass"]
    drums = ["four-on-the-floor kick"] if subgenre in {"progressive house", "melodic house", "festival EDM", "big room house", "deep house"} else ["punchy kick"]
    production = ["clean master", "wide stereo"]

    if features["onset_density"] > 2.0:
        instruments.extend(["clap", "hi-hat"])
        drums.extend(["clap on backbeat", "open hi-hat"])
        production.append("punchy drums")
    if features["low_freq_ratio"] > 0.24:
        instruments.append("sidechain bass")
        bass = "sidechain bass"
        production.append("sidechain compression")
    else:
        bass = "bass"
    if features["spectral_centroid"] > 2300:
        instruments.extend(["supersaw synth", "lead synth", "white noise sweep"])
        production.append("bright mix")
    else:
        instruments.extend(["pad", "pluck synth"])
        production.append("warm mix")
    if subgenre in {"progressive house", "piano house", "melodic house"}:
        instruments.append("piano")
    if subgenre == "folk EDM":
        instruments.append("acoustic guitar")
    if energy in {"high", "very_high"}:
        instruments.append("fx riser")

    def dedupe(values: list[str], vocab: list[str] | None = None) -> list[str]:
        out = []
        for value in values:
            if vocab is not None and value not in vocab:
                continue
            if value not in out:
                out.append(value)
        return out

    instruments = dedupe(instruments, INSTRUMENT_VOCAB)[:8]
    drums = dedupe(drums)[:4]
    production = dedupe(production, PRODUCTION_VOCAB)[:5]
    confidence = {
        "instruments": 0.55,
        "drums": 0.58,
        "bass": 0.62,
        "production": 0.55,
    }
    return instruments, drums, bass, production, confidence


def quality_score(features: dict[str, Any]) -> tuple[int, str, list[str], str | None]:
    score = 5
    defects: list[str] = []
    remove_reason: str | None = None

    if features["rms"] < MIN_RMS:
        score -= 3
        defects.append("too_silent")
        remove_reason = "too_silent"
    if features["clipping_ratio"] > MAX_CLIPPING_RATIO:
        score -= 2
        defects.append("clipping")
        remove_reason = remove_reason or "clipping"
    if features["peak"] < 0.02:
        score -= 1
        defects.append("very_low_peak")
    if features["spectral_centroid"] < 350:
        score -= 1
        defects.append("low_frequency_only")

    score = max(1, min(5, score))
    if score >= 4:
        quality = "good"
    elif score == 3:
        quality = "backup"
    else:
        quality = "bad"
    return score, quality, defects, remove_reason


def make_caption(metadata: dict[str, Any]) -> str:
    bpm = metadata.get("bpm") or "unknown"
    bpm_text = f"{bpm} BPM" if isinstance(bpm, int) and bpm > 0 else "unknown BPM"
    mood = " and ".join(metadata["mood"][:2]) if metadata.get("mood") else "energetic"
    drums = ", ".join(metadata.get("drums", [])[:2]) or "electronic drums"
    instruments = [i for i in metadata.get("instruments", []) if i not in {"kick", "clap", "hi-hat", "bass"}]
    instrument_text = ", ".join(instruments[:3]) or "synth textures"
    production = ", ".join(metadata.get("production", [])[:2]) or "clean master"
    return (
        f"A {bpm_text} {mood} {metadata['subgenre']} {metadata['section']} "
        f"with {drums}, {metadata['bass']}, {instrument_text}, and {production}."
    )


def save_mel(clip_audio: np.ndarray, clip_sr: int, mel_path: Path) -> None:
    mono = np.mean(clip_audio, axis=0)
    mono16 = librosa.resample(mono, orig_sr=clip_sr, target_sr=MEL_SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=mono16,
        sr=MEL_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(mel_path), log_mel)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def assign_song_level_splits(rows: list[dict[str, Any]]) -> None:
    source_ids = sorted({row["source_id"] for row in rows})
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(source_ids)
    n = len(source_ids)
    if n == 0:
        return
    train_n = max(1, int(n * 0.80))
    val_n = max(1, int(n * 0.10)) if n >= 10 else max(0, int(n * 0.10))
    if train_n + val_n >= n and n > 1:
        train_n = max(1, n - 2)
        val_n = 1
    train_ids = set(source_ids[:train_n])
    val_ids = set(source_ids[train_n:train_n + val_n])
    test_ids = set(source_ids[train_n + val_n:])
    if not test_ids and n > 1:
        moved = next(iter(val_ids or train_ids))
        val_ids.discard(moved)
        train_ids.discard(moved)
        test_ids.add(moved)

    for row in rows:
        sid = row["source_id"]
        if sid in train_ids:
            row["split"] = "train"
        elif sid in val_ids:
            row["split"] = "val"
        elif sid in test_ids:
            row["split"] = "test"
        else:
            row["split"] = "train"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def build_dataset(max_files: int | None = None) -> None:
    ensure_dirs()
    reset_generated_dirs()

    raw_files = sorted(path for path in RAW_DIR.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTS)
    if max_files:
        raw_files = raw_files[:max_files]

    removed: list[RemovedItem] = []
    duplicate_report: dict[str, Any] = {"duplicates": []}
    seen_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    source_success = 0

    for file_index, raw_path in enumerate(raw_files, start=1):
        print(f"[{file_index}/{len(raw_files)}] {raw_path.name}", flush=True)
        source_rel = posix_rel(raw_path)
        try:
            raw_hash = sha1_file(raw_path)
        except Exception as exc:
            removed.append(RemovedItem("source", source_rel, "hash_failed", str(exc)))
            continue
        if raw_hash in seen_hashes:
            duplicate_report["duplicates"].append({"path": source_rel, "duplicate_of": seen_hashes[raw_hash], "sha1": raw_hash})
            removed.append(RemovedItem("source", source_rel, "duplicate_file", f"duplicate_of={seen_hashes[raw_hash]}"))
            continue
        seen_hashes[raw_hash] = source_rel

        try:
            y, original_sr = load_audio(raw_path)
        except Exception as exc:
            removed.append(RemovedItem("source", source_rel, "corrupted_audio", str(exc)))
            continue

        if y.size == 0:
            removed.append(RemovedItem("source", source_rel, "empty_audio"))
            continue

        original_channels = int(y.shape[0])
        original_duration = float(y.shape[1] / original_sr)
        if original_duration < MIN_SOURCE_SECONDS:
            removed.append(RemovedItem("source", source_rel, "too_short", f"duration={original_duration:.2f}"))
            continue

        if original_sr != TARGET_SR:
            y_resampled = np.stack([librosa.resample(ch, orig_sr=original_sr, target_sr=TARGET_SR) for ch in y], axis=0)
        else:
            y_resampled = y
        y_stereo = to_stereo(y_resampled)
        y_norm, loudness_normalized, gain = normalize_audio(y_stereo)

        source_id = f"src_{file_index:04d}_{slugify(raw_path.stem, 'song')[:52]}"
        processed_path = PROCESSED_DIR / f"{source_id}.wav"
        save_wav(processed_path, y_norm, TARGET_SR)
        source_success += 1

        total_samples = y_norm.shape[1]
        clip_samples = int(CLIP_SECONDS * TARGET_SR)
        hop_samples = int(HOP_SECONDS * TARGET_SR)
        source_duration = float(total_samples / TARGET_SR)
        clip_count_for_source = 0

        for clip_start in range(0, max(0, total_samples - clip_samples + 1), hop_samples):
            start_time = clip_start / TARGET_SR
            end_time = start_time + CLIP_SECONDS
            clip_audio = y_norm[:, clip_start:clip_start + clip_samples]
            clip_index = int(round(start_time))
            clip_id = f"{source_id}_{clip_index:04d}s_{int(round(end_time)):04d}s"

            try:
                features = analyze_clip(clip_audio, TARGET_SR)
            except Exception as exc:
                removed.append(RemovedItem("clip", f"clips/{clip_id}.wav", "feature_extraction_failed", str(exc)))
                continue

            score, quality, defects, remove_reason = quality_score(features)
            if score < 4:
                removed.append(RemovedItem("clip", f"clips/{clip_id}.wav", remove_reason or "low_quality", f"quality_score={score}; defects={defects}"))
                continue

            energy, energy_conf = infer_energy(features)
            subgenre, subgenre_conf = infer_subgenre(features, energy)
            section, section_conf = infer_section(source_duration, start_time, features, energy)
            mood, mood_conf = infer_mood(subgenre, energy, features)
            instruments, drums, bass, production, tag_conf = infer_instruments_and_production(subgenre, energy, features)

            clip_path = CLIPS_DIR / f"{clip_id}.wav"
            mel_path = MELS_DIR / f"{clip_id}.npy"
            save_wav(clip_path, clip_audio, TARGET_SR)
            save_mel(clip_audio, TARGET_SR, mel_path)

            row: dict[str, Any] = {
                "clip_id": clip_id,
                "source_id": source_id,
                "source_file": source_rel,
                "processed_audio_path": posix_rel(processed_path),
                "audio_path": posix_rel(clip_path),
                "clip_path": posix_rel(clip_path),
                "mel_path": posix_rel(mel_path),
                "latent_path": "",
                "audio_token_path": "",
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration": CLIP_SECONDS,
                "original_duration": round(original_duration, 3),
                "sample_rate": TARGET_SR,
                "channels": TARGET_CHANNELS,
                "original_sample_rate": original_sr,
                "target_sample_rate": TARGET_SR,
                "original_channels": original_channels,
                "target_channels": TARGET_CHANNELS,
                "loudness_normalized": loudness_normalized,
                "normalization_gain": round(gain, 4),
                "genre": "EDM",
                "subgenre": subgenre,
                "bpm": int(features["bpm"]) if features["bpm"] else 0,
                "bpm_confidence": features["bpm_confidence"],
                "key": features["key"],
                "key_confidence": features["key_confidence"],
                "mood": mood,
                "energy": energy,
                "section": section,
                "instruments": instruments,
                "drums": drums,
                "bass": bass,
                "vocal": "unknown",
                "production": production,
                "quality_score": score,
                "quality": quality,
                "audio_defects": defects,
                "split": "",
                "caption": "",
                "tags_source": {
                    "bpm": "auto",
                    "key": "auto" if features["key"] != "unknown" else "unknown",
                    "caption": "heuristic_generated_checked_required",
                    "subgenre": "heuristic_audio_features",
                    "section": "heuristic_position_energy",
                },
                "tag_confidence": {
                    "subgenre": subgenre_conf,
                    "mood": mood_conf,
                    "energy": energy_conf,
                    "section": section_conf,
                    **tag_conf,
                },
                "audio_features": {
                    "rms": round(features["rms"], 6),
                    "peak": round(features["peak"], 6),
                    "clipping_ratio": round(features["clipping_ratio"], 6),
                    "spectral_centroid": round(features["spectral_centroid"], 2),
                    "spectral_rolloff": round(features["spectral_rolloff"], 2),
                    "zero_crossing_rate": round(features["zero_crossing_rate"], 6),
                    "onset_density": round(features["onset_density"], 3),
                    "low_freq_ratio": round(features["low_freq_ratio"], 6),
                },
                "mel_config": {
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "sample_rate": MEL_SAMPLE_RATE,
                },
                "codec": "",
                "codec_sample_rate": 0,
                "codebook_count": 0,
                "processing_notes": [
                    "source audio preserved in raw_audio",
                    "processed to stereo 44.1kHz PCM16 wav",
                    "mel spectrogram generated at 16kHz for diffusion-style experiments",
                    "latents and audio tokens intentionally not generated until target VAE/codec is selected",
                ],
            }
            row["caption"] = make_caption(row)
            rows.append(row)
            clip_count_for_source += 1

        if clip_count_for_source == 0:
            removed.append(RemovedItem("source", source_rel, "no_good_clips", "all clips removed or source too short after processing"))

    assign_song_level_splits(rows)

    metadata_path = DATASET_ROOT / "metadata.jsonl"
    metadata_csv_path = DATASET_ROOT / "metadata.csv"
    write_jsonl(metadata_path, rows)
    write_csv(metadata_csv_path, rows)

    for split in ["train", "val", "test"]:
        write_jsonl(SPLITS_DIR / f"{split}.jsonl", [row for row in rows if row.get("split") == split])

    removed_rows = [item.__dict__ for item in removed]
    write_jsonl(REPORTS_DIR / "removed_files.jsonl", removed_rows)
    (REPORTS_DIR / "duplicate_report.json").write_text(json.dumps(duplicate_report, ensure_ascii=False, indent=2), encoding="utf-8")

    label_stats = {
        "sample_count": len(rows),
        "source_file_count": len(raw_files),
        "processed_source_count": source_success,
        "split": Counter(row["split"] for row in rows),
        "subgenre": Counter(row["subgenre"] for row in rows),
        "mood": Counter(m for row in rows for m in row["mood"]),
        "section": Counter(row["section"] for row in rows),
        "energy": Counter(row["energy"] for row in rows),
        "quality": Counter(row["quality"] for row in rows),
        "bpm": Counter(str(row["bpm"]) for row in rows),
    }
    serializable_stats = {key: dict(value) if isinstance(value, Counter) else value for key, value in label_stats.items()}
    (REPORTS_DIR / "label_statistics.json").write_text(json.dumps(serializable_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    missing_caption = [row["clip_id"] for row in rows if not row.get("caption")]
    low_quality = [row["clip_id"] for row in rows if row.get("quality_score", 0) < 4]
    avg_duration = float(np.mean([row["duration"] for row in rows])) if rows else 0.0
    report = f"""# EDM Dataset Preprocess Report

## Summary

- Raw source files discovered: {len(raw_files)}
- Successfully processed source files: {source_success}
- Removed / isolated items: {len(removed_rows)}
- Duplicate source files: {len(duplicate_report['duplicates'])}
- Final training-quality clips: {len(rows)}
- Average clip duration: {avg_duration:.2f}s
- Clip sample rate: {TARGET_SR} Hz
- Clip channels: {TARGET_CHANNELS}
- Mel sample rate: {MEL_SAMPLE_RATE} Hz
- Mel parameters: n_mels={N_MELS}, n_fft={N_FFT}, hop_length={HOP_LENGTH}

## Split Counts

{json.dumps(serializable_stats['split'], ensure_ascii=False, indent=2)}

## Subgenre Distribution

{json.dumps(serializable_stats['subgenre'], ensure_ascii=False, indent=2)}

## BPM Distribution

{json.dumps(serializable_stats['bpm'], ensure_ascii=False, indent=2)}

## Mood Distribution

{json.dumps(serializable_stats['mood'], ensure_ascii=False, indent=2)}

## Section Distribution

{json.dumps(serializable_stats['section'], ensure_ascii=False, indent=2)}

## Energy Distribution

{json.dumps(serializable_stats['energy'], ensure_ascii=False, indent=2)}

## Removed Reason Statistics

{json.dumps(dict(Counter(item['reason'] for item in removed_rows)), ensure_ascii=False, indent=2)}

## Low Quality Sample List

{json.dumps(low_quality[:200], ensure_ascii=False, indent=2)}

## Duplicate Sample List

See `duplicate_report.json`.

## Missing Caption Samples

{json.dumps(missing_caption, ensure_ascii=False, indent=2)}

## Notes

- Original raw audio is preserved in `raw_audio/`.
- All metadata paths use Linux/POSIX `/` separators.
- Captions avoid direct artist-name style labels and describe audible musical properties.
- Splits are assigned by source song/file, not by random clip, to avoid leakage.
- `tokens/` and `latents/` are intentionally empty until the target EnCodec/SoundStream codec or target VAE is selected.
"""
    (REPORTS_DIR / "preprocess_report.md").write_text(report, encoding="utf-8")

    print("\nDataset build complete.", flush=True)
    print(f"metadata: {metadata_path}", flush=True)
    print(f"clips: {len(rows)}", flush=True)
    print(f"removed: {len(removed_rows)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean EDM text-to-music dataset.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional debug limit for raw files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(max_files=args.max_files)

