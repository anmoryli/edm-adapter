"""准备 ACE-Step 微调数据：从 metadata.jsonl 生成训练所需格式

ACE-Step 要求的格式：
- audio.mp3      - 音频文件
- audio_prompt.txt - 风格标签（逗号分隔）
- audio_lyrics.txt - 歌词
"""

import os
import json
import shutil
from pathlib import Path


def prepare_training_data(
    metadata_path: str = "data/finetune/metadata.jsonl",
    audio_dir: str = "data/finetune/audio",
    output_dir: str = "data/finetune/training",
    copy_audio: bool = False,
):
    """从 metadata.jsonl 准备 ACE-Step 训练数据

    Args:
        metadata_path: metadata.jsonl 路径
        audio_dir: 音频文件目录
        output_dir: 输出目录
        copy_audio: 是否复制音频文件（False 则创建软链接）
    """
    # 读取元数据
    metadata_list = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                metadata_list.append(json.loads(line))

    print(f"读取到 {len(metadata_list)} 条元数据")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    success = 0
    fail = 0

    for item in metadata_list:
        try:
            # 获取音频文件路径
            audio_filename = os.path.basename(item['audio_path'])
            audio_path = os.path.join(audio_dir, audio_filename)

            if not os.path.exists(audio_path):
                # 尝试用原始路径
                audio_path = item['audio_path']
                if not os.path.exists(audio_path):
                    print(f"跳过（文件不存在）: {audio_filename}")
                    fail += 1
                    continue

            # 获取文件名（不含扩展名）
            stem = Path(audio_filename).stem

            # 生成 prompt 文本（逗号分隔的标签）
            prompt_text = item.get('prompt', '')
            # 简化 prompt，去除 "xxx style," 前缀
            if 'style,' in prompt_text:
                prompt_text = prompt_text.split('style,', 1)[1].strip()

            # 生成歌词文本
            lyrics_text = item.get('lyrics', '[instrumental]')

            # 输出文件路径
            out_audio_path = os.path.join(output_dir, f"{stem}.mp3")
            out_prompt_path = os.path.join(output_dir, f"{stem}_prompt.txt")
            out_lyrics_path = os.path.join(output_dir, f"{stem}_lyrics.txt")

            # 复制或链接音频文件
            if copy_audio:
                shutil.copy2(audio_path, out_audio_path)
            else:
                # 创建相对路径软链接
                rel_path = os.path.relpath(audio_path, output_dir)
                if os.path.exists(out_audio_path):
                    os.remove(out_audio_path)
                os.symlink(rel_path, out_audio_path)

            # 写入 prompt 文件
            with open(out_prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt_text)

            # 写入 lyrics 文件
            with open(out_lyrics_path, 'w', encoding='utf-8') as f:
                f.write(lyrics_text)

            success += 1

        except Exception as e:
            print(f"失败: {item.get('id', 'unknown')} -> {e}")
            fail += 1

    print(f"\n完成: 成功 {success}, 失败 {fail}")
    print(f"输出目录: {output_dir}")

    # 统计
    prompt_files = list(Path(output_dir).glob("*_prompt.txt"))
    lyrics_files = list(Path(output_dir).glob("*_lyrics.txt"))
    audio_files = list(Path(output_dir).glob("*.mp3"))

    print(f"\n文件统计:")
    print(f"  音频文件: {len(audio_files)}")
    print(f"  Prompt 文件: {len(prompt_files)}")
    print(f"  Lyrics 文件: {len(lyrics_files)}")

    return output_dir


if __name__ == "__main__":
    import sys

    metadata_path = sys.argv[1] if len(sys.argv) > 1 else "data/finetune/metadata.jsonl"
    audio_dir = sys.argv[2] if len(sys.argv) > 2 else "data/finetune/audio"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/finetune/training"
    copy_audio = "--copy" in sys.argv

    prepare_training_data(metadata_path, audio_dir, output_dir, copy_audio)
