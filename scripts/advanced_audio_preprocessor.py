"""高级音频预处理器：音频分段选择、响度标准化、质量筛选

用于在训练前优化音频数据质量。
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class AudioSegment:
    """音频片段信息"""
    start_time: float
    end_time: float
    energy: float
    onset_density: float
    spectral_centroid: float
    score: float  # 综合评分


class AudioPreprocessor:
    """音频预处理器"""

    def __init__(
        self,
        target_sr: int = 48000,
        target_lufs: float = -14.0,  # 响度标准化目标
        min_duration: float = 10.0,
        max_duration: float = 180.0,
        segment_duration: float = 30.0,  # 分析片段时长
    ):
        self.target_sr = target_sr
        self.target_lufs = target_lufs
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.segment_duration = segment_duration

    def analyze_segment_quality(self, y: np.ndarray, sr: int) -> dict:
        """分析单个音频片段的质量指标"""
        # RMS 能量
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        # 频谱质心（音色明亮度）
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_mean = float(np.mean(spectral_centroid))

        # 起音密度（节奏活跃度）
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_density = float(np.mean(onset_env > np.mean(onset_env)))

        # 频谱对比度（频段分离度）
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = float(np.mean(spectral_contrast))

        # 零交叉率（噪声程度）
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))

        # 静音比例
        silence_threshold = 0.01
        silent_ratio = float(np.mean(np.abs(y) < silence_threshold))

        return {
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "spectral_centroid": centroid_mean,
            "onset_density": onset_density,
            "spectral_contrast": contrast_mean,
            "zcr_mean": zcr_mean,
            "silent_ratio": silent_ratio,
        }

    def calculate_segment_score(self, metrics: dict) -> float:
        """计算片段综合评分（越高越好）"""
        score = 0.0

        # 能量得分（不要太高也不要太低）
        rms = metrics["rms_mean"]
        if 0.02 < rms < 0.3:
            score += 2.0
        elif rms > 0.01:
            score += 1.0

        # 节奏活跃度得分
        onset = metrics["onset_density"]
        if onset > 0.3:
            score += 2.0
        elif onset > 0.15:
            score += 1.0

        # 频谱丰富度得分
        centroid = metrics["spectral_centroid"]
        if 1000 < centroid < 5000:
            score += 1.5
        elif centroid > 500:
            score += 0.5

        # 对比度得分
        contrast = metrics["spectral_contrast"]
        if contrast > 20:
            score += 1.0

        # 惩罚：静音太多
        if metrics["silent_ratio"] > 0.3:
            score -= 2.0

        # 惩罚：噪声太多
        if metrics["zcr_mean"] > 0.2:
            score -= 1.0

        return score

    def find_best_segment(
        self,
        y: np.ndarray,
        sr: int,
        segment_duration: float = None,
        hop_duration: float = 5.0,
    ) -> Tuple[float, float, float]:
        """找到音频中质量最好的片段

        Args:
            y: 音频数据
            sr: 采样率
            segment_duration: 片段时长（秒）
            hop_duration: 滑动窗口步长（秒）

        Returns:
            (start_time, end_time, score) 最佳片段
        """
        if segment_duration is None:
            segment_duration = self.segment_duration

        segment_samples = int(segment_duration * sr)
        hop_samples = int(hop_duration * sr)

        # 如果音频长度小于片段长度，直接返回
        if len(y) <= segment_samples:
            metrics = self.analyze_segment_quality(y, sr)
            score = self.calculate_segment_score(metrics)
            return 0.0, len(y) / sr, score

        best_score = -float('inf')
        best_start = 0
        best_end = segment_duration

        # 滑动窗口搜索
        for start_sample in range(0, len(y) - segment_samples, hop_samples):
            end_sample = start_sample + segment_samples
            segment = y[start_sample:end_sample]

            metrics = self.analyze_segment_quality(segment, sr)
            score = self.calculate_segment_score(metrics)

            if score > best_score:
                best_score = score
                best_start = start_sample / sr
                best_end = end_sample / sr

        return best_start, best_end, best_score

    def normalize_loudness(self, y: np.ndarray, sr: int) -> np.ndarray:
        """响度标准化（简化版，基于 RMS）"""
        # 计算当前 RMS
        rms = np.sqrt(np.mean(y**2))

        if rms < 1e-6:
            return y

        # 目标 RMS（基于 target_lufs 的近似值）
        target_rms = 0.1  # 约 -20 dBFS

        # 计算增益
        gain = target_rms / rms

        # 限制增益范围
        gain = min(gain, 10.0)  # 最多放大 10 倍
        gain = max(gain, 0.1)   # 最多缩小 10 倍

        # 应用增益
        y_normalized = y * gain

        # 防止削波
        max_val = np.max(np.abs(y_normalized))
        if max_val > 0.95:
            y_normalized = y_normalized * 0.95 / max_val

        return y_normalized

    def remove_silence(self, y: np.ndarray, sr: int, threshold: float = 0.01) -> np.ndarray:
        """移除首尾静音"""
        # 找到非静音区域
        non_silent = np.where(np.abs(y) > threshold)[0]

        if len(non_silent) == 0:
            return y

        start = max(0, non_silent[0] - sr // 10)  # 保留 100ms 的开头
        end = min(len(y), non_silent[-1] + sr // 10)  # 保留 100ms 的结尾

        return y[start:end]

    def preprocess_audio(
        self,
        input_path: str,
        output_path: str,
        select_best_segment: bool = True,
        normalize: bool = True,
        remove_silence: bool = True,
        target_duration: float = None,
    ) -> dict:
        """预处理单个音频文件

        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            select_best_segment: 是否选择最佳片段
            normalize: 是否响度标准化
            remove_silence: 是否移除首尾静音
            target_duration: 目标时长（秒）

        Returns:
            dict: 处理结果信息
        """
        # 加载音频
        y, sr = librosa.load(input_path, sr=self.target_sr, mono=False)

        # 转换为单声道进行分析
        if y.ndim > 1:
            y_mono = np.mean(y, axis=0)
        else:
            y_mono = y

        original_duration = len(y_mono) / sr

        # 移除首尾静音
        if remove_silence:
            y_mono = self.remove_silence(y_mono, sr)
            if y.ndim > 1:
                # 对立体声也进行相同裁剪
                non_silent = np.where(np.abs(y_mono) > 0.01)[0]
                if len(non_silent) > 0:
                    start = max(0, non_silent[0] - sr // 10)
                    end = min(len(y_mono), non_silent[-1] + sr // 10)
                    y = y[:, start:end] if y.ndim > 1 else y[start:end]
                    y_mono = y_mono[start:end]

        # 选择最佳片段
        segment_info = {}
        if select_best_segment and len(y_mono) > self.segment_duration * sr:
            start_time, end_time, score = self.find_best_segment(y_mono, sr)

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            if y.ndim > 1:
                y = y[:, start_sample:end_sample]
            else:
                y = y[start_sample:end_sample]
            y_mono = y_mono[start_sample:end_sample]

            segment_info = {
                "segment_start": start_time,
                "segment_end": end_time,
                "segment_score": score,
            }

        # 如果指定了目标时长，裁剪或填充
        if target_duration:
            target_samples = int(target_duration * sr)
            if len(y_mono) > target_samples:
                if y.ndim > 1:
                    y = y[:, :target_samples]
                else:
                    y = y[:target_samples]
            elif len(y_mono) < target_samples:
                pad_length = target_samples - len(y_mono)
                if y.ndim > 1:
                    y = np.pad(y, ((0, 0), (0, pad_length)), mode='constant')
                else:
                    y = np.pad(y, (0, pad_length), mode='constant')

        # 响度标准化
        if normalize:
            if y.ndim > 1:
                # 对每个通道分别标准化
                for i in range(y.shape[0]):
                    y[i] = self.normalize_loudness(y[i], sr)
            else:
                y = self.normalize_loudness(y, sr)

        # 分析最终质量
        y_final = np.mean(y, axis=0) if y.ndim > 1 else y
        final_metrics = self.analyze_segment_quality(y_final, sr)
        final_score = self.calculate_segment_score(final_metrics)

        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, y.T if y.ndim > 1 else y, sr)

        final_duration = len(y_final) / sr

        return {
            "input_path": input_path,
            "output_path": output_path,
            "original_duration": original_duration,
            "final_duration": final_duration,
            "quality_score": final_score,
            "metrics": final_metrics,
            **segment_info,
        }

    def batch_preprocess(
        self,
        input_dir: str,
        output_dir: str,
        metadata_path: str = None,
        **kwargs,
    ) -> List[dict]:
        """批量预处理音频文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            metadata_path: 元数据文件路径（可选）
            **kwargs: 传递给 preprocess_audio 的参数

        Returns:
            list: 处理结果列表
        """
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        results = []

        # 读取元数据（如果有）
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        metadata[item['id']] = item

        # 扫描音频文件
        audio_files = []
        for f in sorted(os.listdir(input_dir)):
            if Path(f).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(input_dir, f))

        print(f"找到 {len(audio_files)} 个音频文件")

        for i, input_path in enumerate(audio_files):
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, filename)

            print(f"[{i+1}/{len(audio_files)}] 处理: {filename}")

            try:
                result = self.preprocess_audio(
                    input_path, output_path, **kwargs
                )
                results.append(result)

                # 显示质量评分
                score = result['quality_score']
                duration = result['final_duration']
                print(f"  评分: {score:.1f}, 时长: {duration:.1f}s")

            except Exception as e:
                print(f"  失败: {e}")
                results.append({
                    "input_path": input_path,
                    "error": str(e),
                })

        # 统计
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]

        print(f"\n处理完成:")
        print(f"  成功: {len(successful)}")
        print(f"  失败: {len(failed)}")

        if successful:
            scores = [r['quality_score'] for r in successful]
            print(f"  质量评分: {min(scores):.1f} - {max(scores):.1f} (平均: {np.mean(scores):.1f})")

            durations = [r['final_duration'] for r in successful]
            print(f"  时长范围: {min(durations):.1f}s - {max(durations):.1f}s (平均: {np.mean(durations):.1f}s)")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="高级音频预处理器")
    parser.add_argument("--input_dir", type=str, required=True, help="输入音频目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--metadata", type=str, default=None, help="元数据文件路径")
    parser.add_argument("--segment_duration", type=float, default=30.0,
                       help="目标片段时长（秒）")
    parser.add_argument("--target_duration", type=float, default=None,
                       help="固定输出时长（秒）")
    parser.add_argument("--no_segment", action="store_true",
                       help="不进行片段选择")
    parser.add_argument("--no_normalize", action="store_true",
                       help="不进行响度标准化")
    parser.add_argument("--target_lufs", type=float, default=-14.0,
                       help="目标响度（LUFS）")

    args = parser.parse_args()

    preprocessor = AudioPreprocessor(
        segment_duration=args.segment_duration,
        target_lufs=args.target_lufs,
    )

    results = preprocessor.batch_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        select_best_segment=not args.no_segment,
        normalize=not args.no_normalize,
        target_duration=args.target_duration,
    )

    # 保存处理报告
    report_path = os.path.join(args.output_dir, "preprocess_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n处理报告已保存到: {report_path}")
