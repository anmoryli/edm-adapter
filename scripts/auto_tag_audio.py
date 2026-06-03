"""自动音频标签生成器：分析音频文件并生成 ACE-Step 微调所需的 metadata.jsonl"""

import os
import json
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm


def analyze_audio_features(audio_path: str, sr: int = 22050) -> dict:
    """分析音频特征，提取 BPM、能量、频谱特征等"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=60)  # 只分析前60秒

        # BPM 检测
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if isinstance(tempo, (int, float, np.floating)) else float(tempo[0])

        # RMS 能量
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        # 频谱质心（音色明亮度）
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_mean = float(np.mean(spectral_centroid))

        # 频谱带宽（音色丰富度）
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        bandwidth_mean = float(np.mean(spectral_bandwidth))

        # 频谱对比度（频段分离度）
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = float(np.mean(spectral_contrast))

        # 色度特征（调性）
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        estimated_key = key_names[int(np.argmax(chroma_mean))]

        # 起音密度（节奏活跃度）
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_density = float(np.mean(onset_env > np.mean(onset_env)))

        # 低频能量比（bass 强度）
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        low_mask = freqs < 250
        high_mask = freqs >= 250
        low_energy = np.mean(stft[low_mask])
        high_energy = np.mean(stft[high_mask])
        low_freq_ratio = float(low_energy / (high_energy + 1e-8))

        # MFCC（音色特征）
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()

        return {
            "bpm": round(bpm),
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "spectral_centroid": centroid_mean,
            "spectral_bandwidth": bandwidth_mean,
            "spectral_contrast": contrast_mean,
            "estimated_key": estimated_key,
            "onset_density": onset_density,
            "low_freq_ratio": low_freq_ratio,
            "mfcc_mean": mfcc_mean,
        }
    except Exception as e:
        print(f"分析失败 {audio_path}: {e}")
        return None


def classify_genre(features: dict, filename: str) -> str:
    """基于特征和文件名推断音乐风格"""
    filename_lower = filename.lower()

    # 基于文件名关键词
    if any(kw in filename_lower for kw in ['techno', 'industrial']):
        return "techno"
    if any(kw in filename_lower for kw in ['house', 'groove']):
        return "house"
    if any(kw in filename_lower for kw in ['trap', '808']):
        return "trap"
    if any(kw in filename_lower for kw in ['ambient', 'atmospheric', 'chill']):
        return "ambient"
    if any(kw in filename_lower for kw in ['drum and bass', 'dnb', 'jungle']):
        return "drum_and_bass"
    if any(kw in filename_lower for kw in ['dubstep', 'wobble']):
        return "dubstep"
    if any(kw in filename_lower for kw in ['trance', 'euphoric']):
        return "trance"
    if any(kw in filename_lower for kw in ['future bass']):
        return "future_bass"

    # 基于音频特征推断
    bpm = features.get("bpm", 128)
    low_ratio = features.get("low_freq_ratio", 0.5)
    centroid = features.get("spectral_centroid", 2000)
    onset = features.get("onset_density", 0.5)

    if bpm >= 160:
        return "drum_and_bass"
    elif bpm >= 145:
        if low_ratio > 0.8:
            return "dubstep"
        return "trance"
    elif bpm >= 135:
        return "trance"
    elif bpm >= 125:
        if centroid > 3000:
            return "techno"
        return "house"
    elif bpm >= 110:
        if low_ratio > 0.7:
            return "trap"
        return "future_bass"
    else:
        return "ambient"


def classify_energy(features: dict) -> str:
    """基于能量特征推断情绪"""
    rms = features.get("rms_mean", 0.05)
    onset = features.get("onset_density", 0.5)
    centroid = features.get("spectral_centroid", 2000)

    energy_score = rms * 10 + onset + centroid / 5000

    if energy_score > 2.5:
        return "high_energy"
    elif energy_score > 1.8:
        return "energetic"
    elif energy_score > 1.2:
        return "moderate"
    else:
        return "calm"


def extract_artist_from_filename(filename: str) -> str:
    """从文件名提取艺术家信息"""
    # 常见格式: "053 - Avicii - Without You..."
    parts = filename.split(" - ")
    if len(parts) >= 2:
        artist = parts[1].strip()
        # 去除括号内容
        if "(" in artist:
            artist = artist[:artist.index("(")].strip()
        return artist
    return ""


def extract_title_from_filename(filename: str) -> str:
    """从文件名提取歌曲标题"""
    parts = filename.split(" - ")
    if len(parts) >= 3:
        title = parts[2].strip()
        # 去除括号内容和YouTube ID
        if "[" in title:
            title = title[:title.index("[")].strip()
        if "(" in title:
            # 保留主标题，去除括号内的remix信息
            main_title = title[:title.index("(")].strip()
            if main_title:
                return main_title
        return title
    elif len(parts) >= 2:
        return parts[1].strip()
    return filename


def generate_tags(features: dict, genre: str, energy: str, artist: str, title: str) -> str:
    """生成 ACE-Step 格式的标签描述"""
    bpm = features.get("bpm", 128)
    key = features.get("estimated_key", "C")

    # 风格描述映射
    genre_desc = {
        "techno": "dark techno, industrial, hypnotic",
        "house": "house music, groovy, warm, soulful",
        "trap": "trap, aggressive, dark, heavy bass",
        "ambient": "ambient, atmospheric, dreamy, ethereal",
        "drum_and_bass": "drum and bass, energetic, fast breakbeat",
        "dubstep": "dubstep, heavy bass, aggressive, wobble",
        "trance": "trance, uplifting, euphoric, soaring",
        "future_bass": "future bass, emotional, euphoric, bright",
    }

    # 能量描述
    energy_desc = {
        "high_energy": "high energy, powerful, intense, peak time",
        "energetic": "energetic, driving, uplifting",
        "moderate": "moderate energy, balanced, groove",
        "calm": "calm, relaxed, chill, atmospheric",
    }

    # 低频特征描述
    low_ratio = features.get("low_freq_ratio", 0.5)
    bass_desc = "heavy bass" if low_ratio > 0.7 else "balanced bass" if low_ratio > 0.4 else "light bass"

    # 构建标签
    tags = f"{genre_desc.get(genre, 'electronic music')}, {energy_desc.get(energy, 'moderate energy')}, {bpm} BPM, {key} key, {bass_desc}"

    # 添加艺术家信息（如果有）
    if artist:
        tags = f"{artist} style, {tags}"

    return tags


def auto_tag_directory(audio_dir: str, output_path: str, max_duration: float = 180.0):
    """自动为目录中的所有音频文件生成标签

    Args:
        audio_dir: 音频文件目录
        output_path: 输出 metadata.jsonl 路径
        max_duration: 最大处理时长（秒），超过的音频会被截断
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    audio_files = []

    for f in sorted(os.listdir(audio_dir)):
        if Path(f).suffix.lower() in audio_extensions:
            audio_files.append(os.path.join(audio_dir, f))

    print(f"找到 {len(audio_files)} 个音频文件")
    print(f"输出路径: {output_path}")

    metadata_list = []

    for audio_path in tqdm(audio_files, desc="分析音频"):
        filename = os.path.basename(audio_path)
        file_id = Path(filename).stem

        # 提取艺术家和标题
        artist = extract_artist_from_filename(filename)
        title = extract_title_from_filename(filename)

        # 分析音频特征
        features = analyze_audio_features(audio_path)
        if features is None:
            print(f"跳过: {filename}")
            continue

        # 获取音频时长
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True, duration=5)
            duration = librosa.get_duration(filename=audio_path)
        except:
            duration = 60.0

        # 如果音频太长，截断到 max_duration
        if duration > max_duration:
            duration = max_duration

        # 分类
        genre = classify_genre(features, filename)
        energy = classify_energy(features)

        # 生成标签
        tags = generate_tags(features, genre, energy, artist, title)

        # 构建元数据
        metadata = {
            "id": file_id,
            "audio_path": audio_path,
            "duration": round(duration, 1),
            "prompt": tags,
            "lyrics": "[instrumental]",  # 默认纯器乐，如有歌词可手动添加
            "artist": artist,
            "title": title,
            "genre": genre,
            "bpm": features["bpm"],
            "key": features["estimated_key"],
            "energy": energy,
            "low_freq_ratio": round(features["low_freq_ratio"], 3),
            "spectral_centroid": round(features["spectral_centroid"], 1),
        }

        metadata_list.append(metadata)

    # 保存 metadata.jsonl
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in metadata_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n完成！已生成 {len(metadata_list)} 条元数据")
    print(f"保存至: {output_path}")

    # 打印统计
    genres = {}
    for item in metadata_list:
        g = item["genre"]
        genres[g] = genres.get(g, 0) + 1

    print("\n风格分布:")
    for genre, count in sorted(genres.items(), key=lambda x: -x[1]):
        print(f"  {genre}: {count}")

    return metadata_list


if __name__ == "__main__":
    import sys

    audio_dir = sys.argv[1] if len(sys.argv) > 1 else "data/finetune/audio"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/finetune/metadata.jsonl"

    auto_tag_directory(audio_dir, output_path)
