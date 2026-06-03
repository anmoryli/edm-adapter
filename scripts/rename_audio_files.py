"""统一重命名音频文件为简洁格式，并更新 metadata.jsonl"""

import os
import json
import re
from pathlib import Path


def clean_filename(filename: str) -> str:
    """清理文件名，提取核心信息

    输入: "053 - Avicii - Without You (feat. Sandro Cavazza) (Extended Original Mix) (Extended Version) (Unreleased) [IHrhQ7aZvt8].mp3"
    输出: "053_Avicii_Without_You.mp3"
    """
    # 去除扩展名
    stem = Path(filename).stem
    ext = Path(filename).suffix

    # 提取编号（如果有）
    number_match = re.match(r'^(\d+)\s*[-_.]\s*', stem)
    number = number_match.group(1) if number_match else ""
    if number:
        stem = stem[len(number_match.group(0)):]

    # 分割艺术家和标题
    parts = stem.split(" - ", 1)
    if len(parts) >= 2:
        artist = parts[0].strip()
        title = parts[1].strip()
    else:
        artist = ""
        title = stem

    # 清理标题：去除括号内容、YouTube ID、特殊标记
    # 去除 [...] YouTube ID
    title = re.sub(r'\[.*?\]', '', title)
    # 去除 (feat. xxx), (ft. xxx)
    title = re.sub(r'\(feat\.?\s*[^)]+\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\(ft\.?\s*[^)]+\)', '', title, flags=re.IGNORECASE)
    # 去除 (Remix), (Edit), (Mix) 等
    title = re.sub(r'\((?:Extended|Original|Remix|Edit|Mix|Version|Unreleased|Official|Audio|Video|Remaster|Radio|Live|Acoustic|Acappella|Instrumental)[^)]*\)', '', title, flags=re.IGNORECASE)
    # 去除其他括号内容
    title = re.sub(r'\([^)]*\)', '', title)
    # 去除特殊字符，保留中文、英文、数字、空格
    title = re.sub(r'[^\w\s一-鿿]', '', title)
    # 合并多个空格为一个
    title = re.sub(r'\s+', ' ', title).strip()

    # 清理艺术家名
    artist = re.sub(r'[^\w\s一-鿿&]', '', artist)
    artist = re.sub(r'\s+', ' ', artist).strip()

    # 构建新文件名
    parts = []
    if number:
        parts.append(number)
    if artist:
        parts.append(artist)
    if title:
        parts.append(title)

    # 如果清理后为空，使用原始文件名
    if not parts:
        clean_name = stem
    else:
        clean_name = "_".join(parts)

    # 替换空格为下划线，去除连续下划线
    clean_name = clean_name.replace(" ", "_")
    clean_name = re.sub(r'_+', '_', clean_name)
    clean_name = clean_name.strip('_')

    # 限制文件名长度
    if len(clean_name) > 100:
        clean_name = clean_name[:100]

    return clean_name + ext


def rename_audio_files(audio_dir: str, metadata_path: str, dry_run: bool = True):
    """重命名音频文件并更新元数据

    Args:
        audio_dir: 音频文件目录
        metadata_path: metadata.jsonl 路径
        dry_run: 如果为 True，只显示预览不实际重命名
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

    # 读取现有元数据
    metadata_list = []
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    metadata_list.append(json.loads(line))

    # 创建 ID 到元数据的映射
    metadata_by_id = {item['id']: item for item in metadata_list}

    # 扫描音频文件
    rename_map = {}  # {old_path: new_path}
    new_metadata = []

    for filename in sorted(os.listdir(audio_dir)):
        if Path(filename).suffix.lower() not in audio_extensions:
            continue

        old_path = os.path.join(audio_dir, filename)
        new_name = clean_filename(filename)
        new_path = os.path.join(audio_dir, new_name)

        # 避免重名
        if new_path in rename_map.values():
            base = Path(new_name).stem
            ext = Path(new_name).suffix
            counter = 1
            while new_path in rename_map.values():
                new_name = f"{base}_{counter}{ext}"
                new_path = os.path.join(audio_dir, new_name)
                counter += 1

        rename_map[old_path] = new_path

        # 更新元数据
        old_id = Path(filename).stem
        if old_id in metadata_by_id:
            meta = metadata_by_id[old_id].copy()
            meta['id'] = Path(new_name).stem
            meta['audio_path'] = f"data/finetune/audio/{new_name}"
            new_metadata.append(meta)
        else:
            # 创建新元数据
            new_metadata.append({
                'id': Path(new_name).stem,
                'audio_path': f"data/finetune/audio/{new_name}",
                'duration': 180.0,
                'prompt': 'electronic music, EDM',
                'lyrics': '[instrumental]',
                'artist': '',
                'title': '',
                'genre': 'house',
                'bpm': 128,
                'key': 'C',
                'energy': 'moderate',
            })

    # 显示预览
    print(f"找到 {len(rename_map)} 个文件需要重命名\n")
    print("预览（前 20 个）:")
    print("-" * 80)

    for i, (old, new) in enumerate(list(rename_map.items())[:20]):
        old_name = os.path.basename(old)
        new_name = os.path.basename(new)
        try:
            print(f"{old_name}")
            print(f"  -> {new_name}")
            print()
        except UnicodeEncodeError:
            # 处理特殊字符
            print(f"[文件 {i+1}]")
            print(f"  -> {new_name}")
            print()

    if len(rename_map) > 20:
        print(f"... 还有 {len(rename_map) - 20} 个文件")

    # 执行重命名
    if not dry_run:
        print("\n开始重命名...")
        success = 0
        fail = 0

        for old_path, new_path in rename_map.items():
            try:
                os.rename(old_path, new_path)
                success += 1
            except Exception as e:
                print(f"失败: {os.path.basename(old_path)} -> {e}")
                fail += 1

        print(f"\n完成: 成功 {success}, 失败 {fail}")

        # 保存更新后的元数据
        if new_metadata:
            # 备份原文件
            backup_path = metadata_path + '.backup'
            if os.path.exists(metadata_path):
                os.rename(metadata_path, backup_path)
                print(f"原元数据已备份至: {backup_path}")

            with open(metadata_path, 'w', encoding='utf-8') as f:
                for item in new_metadata:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"元数据已更新: {metadata_path}")

    return rename_map, new_metadata


if __name__ == "__main__":
    import sys

    audio_dir = sys.argv[1] if len(sys.argv) > 1 else "data/finetune/audio"
    metadata_path = sys.argv[2] if len(sys.argv) > 2 else "data/finetune/metadata.jsonl"
    dry_run = "--apply" not in sys.argv

    if dry_run:
        print("=== 预览模式（不会实际修改）===")
        print("添加 --apply 参数来执行重命名\n")
    else:
        print("=== 执行模式 ===\n")

    rename_audio_files(audio_dir, metadata_path, dry_run)
