"""Audio feature extraction: BPM, energy, spectral features, etc."""

import numpy as np
import librosa


def extract_all_features(y: np.ndarray, sr: int = 44100) -> dict:
    """Extract all audio features for caption generation and evaluation.

    Args:
        y: Audio array, shape (samples,) mono.
        sr: Sample rate.

    Returns:
        Dictionary of extracted features.
    """
    features = {}

    # BPM
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features["bpm"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    features["num_beats"] = len(beats)

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["rms_max"] = float(np.max(rms))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = float(np.mean(zcr))

    # Onset density
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    duration = len(y) / sr
    features["onset_density"] = float(len(onsets) / duration) if duration > 0 else 0.0

    # Low frequency energy ratio (20-250 Hz)
    features["low_freq_ratio"] = _compute_low_freq_ratio(y, sr)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))

    # Chroma features (for key detection)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = np.mean(chroma, axis=1).tolist()

    return features


def _compute_low_freq_ratio(y: np.ndarray, sr: int) -> float:
    """Compute ratio of energy in 20-250 Hz band to total energy."""
    S = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)

    low_mask = (freqs >= 20) & (freqs <= 250)
    total_energy = np.sum(S ** 2)
    if total_energy < 1e-10:
        return 0.0
    low_energy = np.sum(S[low_mask] ** 2)
    return float(low_energy / total_energy)


def estimate_bpm(y: np.ndarray, sr: int = 44100) -> float:
    """Quick BPM estimation."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo) if np.isscalar(tempo) else float(tempo[0])


def compute_rms(y: np.ndarray) -> float:
    """Compute mean RMS energy."""
    rms = librosa.feature.rms(y=y)[0]
    return float(np.mean(rms))


def compute_loop_similarity(y: np.ndarray, sr: int = 44100, tail_seconds: float = 0.5) -> float:
    """Compute cosine similarity between start and end of audio for loop evaluation.

    Returns value in [-1, 1], higher means better loop continuity.
    """
    tail_samples = int(tail_seconds * sr)
    if len(y) < tail_samples * 4:
        return 0.0

    start = y[:tail_samples]
    end = y[-tail_samples:]

    # Compute mel spectrograms
    S_start = librosa.feature.melspectrogram(y=start, sr=sr, n_mels=64)
    S_end = librosa.feature.melspectrogram(y=end, sr=sr, n_mels=64)

    # Mean over time
    v_start = np.mean(S_start, axis=1)
    v_end = np.mean(S_end, axis=1)

    # Cosine similarity
    dot = np.dot(v_start, v_end)
    norm = np.linalg.norm(v_start) * np.linalg.norm(v_end)
    if norm < 1e-10:
        return 0.0
    return float(dot / norm)
