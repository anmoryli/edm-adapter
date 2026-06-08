"""EDM-Adapter Gradio Demo: ACE-Step 音乐生成 + 音源分离"""

import os
import sys
import random
import json
import threading
import uuid
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gradio as gr

from src.audio_io import load_audio, save_audio, normalize_audio, edm_post_process, get_audio_duration
from src.audio_features import extract_all_features
from src.mel_watermark import save_ai_watermarked_mel
from src.generation import (
    load_acestep_model,
    reset_acestep_model_cache,
    apply_finetuned_weights,
    generate_acestep,
    generate_acestep_reference_style,
    get_acestep_max_duration,
    score_reference_candidate,
    REFERENCE_MELODY_PROMPT,
    REFERENCE_RECONSTRUCTION_PROMPT,
    REFERENCE_STYLE_PROMPT,
)
from src.reports.paper_report import build_paper_report_html


# ============================================================
# 风格模板（ACE-Step 使用标签 + 自然语言混合格式）
# ============================================================

STYLE_PROMPTS = {
    "techno": {
        "tags": "dark techno, industrial, hypnotic, 128 BPM",
        "desc": "Pounding four-on-the-floor kick drum, deep rumbling sub bass, eerie arpeggiated synthesizer melodies, industrial warehouse atmosphere, minor key",
    },
    "house": {
        "tags": "house music, groovy, warm, 124 BPM",
        "desc": "Punchy four-on-the-floor kick, funky bass guitar, bright catchy synth melodies, shuffling hi-hats, uplifting major key, soulful and danceable",
    },
    "trap": {
        "tags": "trap, aggressive, dark, 140 BPM",
        "desc": "Booming 808 sub bass, sharp snare hits, rapid-fire hi-hat rolls, dark eerie synth melodies, cinematic with heavy low end",
    },
    "ambient": {
        "tags": "ambient, atmospheric, dreamy, 85 BPM",
        "desc": "Evolving synth pads, gentle piano melodies, soft drones, lush reverb washes, ethereal and floating, no percussion, purely textural",
    },
    "drum_and_bass": {
        "tags": "drum and bass, energetic, fast, 170 BPM",
        "desc": "Fast breakbeat patterns, heavy rolling bassline, rapid hi-hats, catchy synth lead hooks, high energy driving rhythm",
    },
    "future_bass": {
        "tags": "future bass, emotional, euphoric, 150 BPM",
        "desc": "Bright supersaw chord stabs, wobbly sidechained bass, punchy drums with snare builds, euphoric synth melodies, lush and uplifting",
    },
    "trance": {
        "tags": "trance, uplifting, euphoric, 138 BPM",
        "desc": "Soaring euphoric synth melodies, driving pulsing bassline, four-on-the-floor kick, open hi-hats, lush string pad chord progressions",
    },
    "dubstep": {
        "tags": "dubstep, heavy bass, aggressive, 140 BPM",
        "desc": "Massive wobble bass growls, metallic screechy synth leads, heavy kick drums, sharp snares, dark atmospheric tension, intense drops",
    },
}

MOOD_OPTIONS = {
    "黑暗": "dark, moody, ominous",
    "高能": "energetic, high energy, powerful",
    "梦幻": "dreamy, ethereal, atmospheric",
    "凶猛": "aggressive, hard-hitting, intense",
    "放松": "relaxing, calm, peaceful, gentle",
    "迷幻": "hypnotic, repetitive, trance-like",
    "狂喜": "euphoric, uplifting, joyful",
    "忧郁": "melancholic, sad, emotional, bittersweet",
}

STYLE_BPM = {
    "techno": 128, "house": 124, "trap": 140, "ambient": 85,
    "drum_and_bass": 170, "future_bass": 150, "trance": 138, "dubstep": 140,
}

STYLE_NAMES = {
    "techno": "techno",
    "house": "house",
    "trap": "trap",
    "ambient": "ambient electronic",
    "drum_and_bass": "drum and bass",
    "future_bass": "future bass",
    "trance": "trance",
    "dubstep": "dubstep",
}

STYLE_LABELS = {
    "techno": "Techno 暗黑工业",
    "house": "House 浩室",
    "trap": "Trap 陷阱",
    "ambient": "Ambient 氛围",
    "drum_and_bass": "Drum & Bass 鼓打贝斯",
    "future_bass": "Future Bass 未来贝斯",
    "trance": "Trance 出神",
    "dubstep": "Dubstep 回响贝斯",
}

MOOD_LABELS = {
    "黑暗": "黑暗 Dark",
    "高能": "高能 Energetic",
    "梦幻": "梦幻 Dreamy",
    "凶猛": "凶猛 Aggressive",
    "放松": "放松 Relaxing",
    "迷幻": "迷幻 Hypnotic",
    "狂喜": "狂喜 Euphoric",
    "忧郁": "忧郁 Melancholic",
}

EXAMPLE_PROMPTS = [
    [
        "uplifting progressive house, 128 BPM, bright piano intro, emotional supersaw drop, festival mainstage energy",
        STYLE_LABELS["house"],
        128,
        MOOD_LABELS["狂喜"],
        45,
        "",
    ],
    [
        "melodic EDM, 126 BPM, acoustic guitar plucks, warm vocal chops, clean sidechain groove, hopeful summer atmosphere",
        STYLE_LABELS["future_bass"],
        126,
        MOOD_LABELS["高能"],
        45,
        "",
    ],
    [
        "dark warehouse techno, 130 BPM, heavy kick, rolling bass, hypnotic arpeggio, cold industrial reverb",
        STYLE_LABELS["techno"],
        130,
        MOOD_LABELS["黑暗"],
        40,
        "",
    ],
    [
        "ambient electronic intro, 85 BPM, wide pads, soft piano motif, distant shimmer, cinematic build",
        STYLE_LABELS["ambient"],
        85,
        MOOD_LABELS["梦幻"],
        35,
        "",
    ],
]

LORA_TEST_PROMPT = (
    "Avicii style, house music, groovy, warm, soulful, high energy, powerful, "
    "intense, peak time, 129 BPM, A key, heavy bass, bright piano plucks, "
    "uplifting festival lead, sidechain pumping, melodic progressive house drop, "
    "clean supersaw chords, professional EDM mix"
)

REFERENCE_MODE_LABELS = {
    "style_timbre": "风格音色（新旋律）",
    "reconstruct": "参考重构（最稳）",
    "melody": "旋律保留",
    "style": "风格迁移",
    "variation": "强变化",
}

# ACE-Step 单次最大时长
MAX_DURATION = 240.0


def build_prompt(user_desc: str, style: str, mood: str, bpm: int) -> str:
    """Build a minimally opinionated prompt.

    用户写了提示词时，用户提示词必须是主体；这里仅追加质量、结构和混音约束。
    风格下拉只在用户提示词为空时兜底，避免默认模板盖过 LoRA / 用户风格。
    """
    user_text = (user_desc or "").strip()
    if user_text:
        prompt = user_text
        lower = user_text.lower()
        if "bpm" not in lower and bpm:
            prompt += f", around {int(bpm)} BPM"
    else:
        style_name = STYLE_NAMES.get(style, style)
        mood_desc = MOOD_OPTIONS.get(mood, mood)
        prompt = f"{style_name}, {mood_desc}, around {int(bpm)} BPM"

    prompt = prompt.rstrip(" .")
    prompt += (
        ". Follow the prompt closely. Coherent musical structure, clean mix, balanced dynamics, "
        "high quality audio, no clipping, no random speech or vocals unless lyrics are provided."
    )
    return prompt


# ============================================================
# 模型管理
# ============================================================

_acestep_pipeline = None
_current_model_key = None
_current_model_kind = None
_current_lora_weight = None
_last_model_load_summary = ""


def _parse_optional_timeout_seconds(value):
    text = (value or "").strip().lower()
    if text in {"", "0", "-1", "none", "null", "inf", "infinite", "unlimited", "无限", "无限制"}:
        return None
    seconds = int(float(text))
    if seconds <= 0:
        return None
    return max(30, seconds)


def _format_timeout_label(timeout_seconds):
    if timeout_seconds is None:
        return "无限制"
    if timeout_seconds % 3600 == 0:
        return f"{timeout_seconds // 3600} 小时"
    if timeout_seconds >= 60:
        return f"{timeout_seconds // 60} 分钟"
    return f"{timeout_seconds} 秒"


# 模型存放目录
def _parse_float_env(name: str, default: float, min_value: float, max_value: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return float(np.clip(value, min_value, max_value))


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
FINETUNED_DIR = os.path.join(PROJECT_ROOT, "outputs", "finetuned")
FINETUNED_MODEL_PATH = os.path.join(FINETUNED_DIR, "model.pt")
FINETUNE_LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "finetune", "logs")
AVICII_LORA_ROOT = os.path.join(PROJECT_ROOT, "outputs", "avicii_local_lora")
SECTION15_COMPARE_DIR = os.path.join(AVICII_LORA_ROOT, "generations", "section15_triple_compare")
EDM_CONTROL_LORA_ROOT = os.path.join(PROJECT_ROOT, "outputs", "edm_control_lora")
WEB_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "web_generations")
TASK_INDEX_PATH = os.path.join(WEB_OUTPUT_DIR, "tasks.json")
TASK_LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "generation_tasks.log")
MAX_PARALLEL_JOBS = max(1, int(os.environ.get("EDM_MAX_PARALLEL_JOBS", "2")))
TRUE_MUSIC_IMAGE_PATH = os.path.join(PROJECT_ROOT, "true_music.jpg")
GSD_VIDEO_PATH = os.path.join(PROJECT_ROOT, "gsd.mp4")
VOICE_MODEL_DIR = os.path.join(PROJECT_ROOT, "1761704195865-bk9wgc-tomori1_e12_s2664")
VOICE_CONVERTER_CMD_ENV = "EDM_VOICE_CONVERTER_CMD"
VOICE_CONVERTER_TIMEOUT_SEC = _parse_optional_timeout_seconds(os.environ.get("EDM_VOICE_CONVERTER_TIMEOUT_SEC", "0"))
VOICE_REVERB_MIX = _parse_float_env("EDM_VOICE_REVERB_MIX", 0.14, 0.0, 0.35)
VOICE_REVERB_SECONDS = _parse_float_env("EDM_VOICE_REVERB_SECONDS", 0.85, 0.15, 2.20)
VOICE_REVERB_PRE_DELAY_MS = _parse_float_env("EDM_VOICE_REVERB_PRE_DELAY_MS", 24.0, 0.0, 80.0)
SEED_VC_DIR = os.path.join(PROJECT_ROOT, "external_tools", "seed-vc")
SEED_VC_WRAPPER_PATH = os.path.join(PROJECT_ROOT, "scripts", "seed_vc_converter.py")
DEFAULT_VOICE_CONVERTER_CMD = (
    f'conda run -n edm-adapter python "{SEED_VC_WRAPPER_PATH}" '
    "--source {input} --output {output} --pitch {pitch} --model-dir {model_dir} --task-dir {task_dir}"
)

_pipeline_cache: dict[str, dict] = {}
_pipeline_cache_lock = threading.RLock()
_pipeline_generation_locks = defaultdict(threading.RLock)
_task_lock = threading.RLock()
_task_executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS, thread_name_prefix="edm-task")
_task_futures = {}

os.makedirs(WEB_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TASK_LOG_PATH), exist_ok=True)


def _format_size(path: str) -> str:
    if not os.path.exists(path):
        return "未找到"
    size = os.path.getsize(path)
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MB"
    return f"{size / 1024:.1f} KB"


def _model_entry(kind: str, path: str = "", detail: str = "") -> dict[str, str]:
    return {"kind": kind, "path": path, "detail": detail}


def _lora_search_roots() -> list[str]:
    return [
        AVICII_LORA_ROOT,
        EDM_CONTROL_LORA_ROOT,
        FINETUNED_DIR,
        CHECKPOINT_DIR,
        FINETUNE_LOG_DIR,
    ]


def _load_lora_manifest(lora_dir: str) -> dict:
    path = Path(lora_dir).resolve()
    for candidate in [path, *path.parents]:
        if candidate == Path(PROJECT_ROOT).resolve().parent:
            break
        manifest_path = candidate / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def _lora_display_name(lora_dir: str) -> str:
    manifest = _load_lora_manifest(lora_dir)
    adapter_name = manifest.get("adapter_name") or os.path.basename(lora_dir)
    step = manifest.get("global_step")
    rel_name = os.path.relpath(lora_dir, PROJECT_ROOT)
    if manifest.get("format") == "avicii_local_lora_v1" or "avicii_local_lora" in rel_name.replace("\\", "/"):
        suffix = f"step={step}" if step is not None else "local"
        return f"Avicii 本地 LoRA ({suffix}, {adapter_name})"
    if step is not None:
        return f"LoRA ({adapter_name}, step={step})"
    return f"LoRA: {rel_name}"


def _find_lora_dirs() -> list[str]:
    """Find ACE-Step LoRA directories that contain pytorch_lora_weights.safetensors."""
    roots = _lora_search_roots()
    found: list[str] = []
    seen = set()
    for root_dir in roots:
        if not os.path.isdir(root_dir):
            continue
        for root, _, files in os.walk(root_dir):
            if "pytorch_lora_weights.safetensors" not in files:
                continue
            full_path = os.path.abspath(root)
            if full_path not in seen:
                seen.add(full_path)
                found.append(full_path)
    found.sort(key=lambda path: (_load_lora_manifest(path).get("global_step") or 0, os.path.getmtime(path)), reverse=True)
    return found


def _find_training_checkpoints() -> list[str]:
    if not os.path.isdir(FINETUNED_DIR):
        return []
    checkpoints = [
        os.path.join(FINETUNED_DIR, name)
        for name in os.listdir(FINETUNED_DIR)
        if name.endswith(".ckpt") and os.path.isfile(os.path.join(FINETUNED_DIR, name))
    ]
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints


def _resolve_ckpt_lora_dir(ckpt_path: str) -> str | None:
    """Find a LoRA inference directory that corresponds to a Lightning .ckpt."""
    ckpt = os.path.abspath(ckpt_path)
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    expected_names = [f"{stem}_lora", stem.replace(".ckpt", "") + "_lora"]
    search_roots = [os.path.dirname(ckpt), FINETUNED_DIR, FINETUNE_LOG_DIR]

    for root_dir in search_roots:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for root, dirs, files in os.walk(root_dir):
            if os.path.basename(root) not in expected_names:
                continue
            if "pytorch_lora_weights.safetensors" in files:
                return os.path.abspath(root)

    return None


def _extract_lora_from_ckpt(ckpt_path: str) -> str:
    """Extract LoRA tensors from a Lightning checkpoint into an inference LoRA directory."""
    from safetensors.torch import save_file

    ckpt_path = os.path.abspath(ckpt_path)
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    output_dir = os.path.join(FINETUNED_DIR, "_ckpt_lora_cache", f"{stem}_lora")
    output_file = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    if os.path.exists(output_file):
        return output_dir

    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    lora_state = {}
    for key, value in state_dict.items():
        if ".lora_A." not in key and ".lora_B." not in key:
            continue
        clean_key = key
        if clean_key.startswith("transformers."):
            clean_key = clean_key[len("transformers."):]
        clean_key = clean_key.replace(".edm_lora.", ".")
        lora_state[clean_key] = value.detach().cpu()

    if not lora_state:
        raise RuntimeError(f"未在 checkpoint 中找到 LoRA 权重: {ckpt_path}")

    os.makedirs(output_dir, exist_ok=True)
    save_file(lora_state, output_file)
    return output_dir


def scan_available_models() -> dict[str, dict[str, str]]:
    """扫描可用模型（基线 + 全量微调权重 + LoRA 检查点）

    返回 {显示名称: {kind, path, detail}} 字典，kind 支持 base/full/lora/ckpt。
    """
    models: dict[str, dict[str, str]] = {}

    for lora_dir in _find_lora_dirs():
        rel_name = os.path.relpath(lora_dir, PROJECT_ROOT)
        weight_path = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")
        label = _lora_display_name(lora_dir)
        if label in models:
            label = f"{label} - {rel_name}"
        models[label] = _model_entry(
            "lora",
            lora_dir,
            f"ACE-Step LoRA 推理权重，大小 {_format_size(weight_path)}，路径 {rel_name}",
        )

    if os.path.exists(FINETUNED_MODEL_PATH):
        models["旧全量模型: outputs/finetuned/model.pt"] = _model_entry(
            "full",
            FINETUNED_DIR,
            f"完整 transformer 权重，大小 {_format_size(FINETUNED_MODEL_PATH)}；默认不在网页加载，避免内存峰值过高",
        )

    for ckpt_path in _find_training_checkpoints():
        rel_name = os.path.relpath(ckpt_path, PROJECT_ROOT)
        lora_dir = _resolve_ckpt_lora_dir(ckpt_path)
        detail = f"Lightning 完整训练 checkpoint，大小 {_format_size(ckpt_path)}"
        if lora_dir:
            detail += f"；推理将使用对应 LoRA: {os.path.relpath(lora_dir, PROJECT_ROOT)}"
        else:
            detail += "；首次推理会从 ckpt 抽取 LoRA 权重"
        models[f"完整 CKPT: {rel_name}"] = _model_entry("ckpt", ckpt_path, detail)

    models["ACE-Step v1-3.5B 基线模型"] = _model_entry("base", "", "未叠加微调权重")

    return models


def _model_choices() -> list[str]:
    return list(scan_available_models().keys())


def _default_model_choice(choices: list[str] | None = None) -> str:
    choices = choices if choices is not None else _model_choices()
    return choices[0] if choices else "ACE-Step v1-3.5B 基线模型"


def _section15_compare_summary() -> str:
    compare_dir = Path(SECTION15_COMPARE_DIR)
    if not compare_dir.exists():
        return ""
    wav_count = len(list(compare_dir.glob("*.wav")))
    mel_count = len(list(compare_dir.glob("*_mel_ai_watermark.png")))
    sidecar_files = sorted(compare_dir.glob("triple_compare_seed42_15s_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sidecar_files:
        return f"15秒段落级二次微调对比目录：`{SECTION15_COMPARE_DIR}`（WAV {wav_count} 个，AI水印 Mel {mel_count} 张）"
    latest = sidecar_files[0]
    timestamp = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return (
        "15秒段落级二次微调已完成：已生成 intro / breakdown / drop 的 "
        f"baseline、step=960 reference、step=1200 candidate 同 seed 对比；"
        f"WAV {wav_count} 个，AI水印 Mel {mel_count} 张，最新索引 `{latest.name}`（{timestamp}）。"
    )


def get_model_status_markdown() -> str:
    """生成网页上展示的模型状态说明。"""
    available = scan_available_models()
    default_model = next(iter(available.keys()), "未找到可用模型")
    lora_dirs = _find_lora_dirs()
    latest_lora = lora_dirs[0] if lora_dirs else ""
    latest_manifest = _load_lora_manifest(latest_lora) if latest_lora else {}
    latest_weight = os.path.join(latest_lora, "pytorch_lora_weights.safetensors") if latest_lora else ""
    latest_rel = os.path.relpath(latest_lora, PROJECT_ROOT) if latest_lora else "未发现 LoRA"
    latest_step = latest_manifest.get("global_step", "未知")
    section15_status = _section15_compare_summary()

    lines = [
        "### 当前模型",
        f"- 默认选择：**{default_model}**",
        f"- 最新 LoRA：`{latest_rel}`" + (f"（step={latest_step}，{_format_size(latest_weight)}）" if latest_lora else ""),
        f"- 可选模型：{len(available)} 个；其中 LoRA {len(lora_dirs)} 个，Base 1 个。",
        "- 对比逻辑：勾选“同时提交基线模型任务”后，会用同一 seed 生成当前模型和 ACE-Step base 两首。",
        "- 授权音色转换：单独页签，上传歌曲后调用 Seed-VC/外部 VC 命令，任务目录保存转换音频、日志和诊断图。",
        f"- 输出目录：`{WEB_OUTPUT_DIR}`",
    ]
    if section15_status:
        lines.insert(3, f"- 段落级二次微调：{section15_status}")
    return "\n".join(lines)


def get_finetuned_readme_markdown() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "docs", "AVICII_LOCAL_FINETUNE_PLAN.md"),
        os.path.join(PROJECT_ROOT, "docs", "REFERENCE_STYLE_AND_FINETUNE_TECHNICAL_REPORT.md"),
        os.path.join(FINETUNED_DIR, "EDM_LORA_README.md"),
        os.path.join(FINETUNED_DIR, "DEMO_README.md"),
    ]
    for readme_path in candidates:
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                return f.read()
    return "当前版本使用 `outputs/avicii_local_lora/` 下的 LoRA bundle；未找到额外 Markdown 说明。"


def _resolve_seed(seed) -> int:
    """Use -1/empty seed for a fresh random generation."""
    try:
        seed_value = int(seed)
    except (TypeError, ValueError):
        seed_value = -1
    if seed_value < 0:
        return random.randint(0, 2**32 - 1)
    return seed_value


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _task_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slugify(value: str, fallback: str = "track", limit: int = 48) -> str:
    chars = []
    for ch in str(value or "").lower():
        if ch.isalnum() and ch.isascii():
            chars.append(ch)
        elif ch in {" ", "-", "_", "."}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return (slug or fallback)[:limit].strip("_") or fallback


def _task_output_dir(task_id: str) -> str:
    return os.path.join(WEB_OUTPUT_DIR, task_id)


def _task_json_path(task_id: str) -> str:
    return os.path.join(_task_output_dir(task_id), "task.json")


def _write_json_atomic(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_task_index_unlocked() -> dict:
    tasks = {}
    if os.path.exists(TASK_INDEX_PATH):
        try:
            with open(TASK_INDEX_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            tasks = data.get("tasks", data if isinstance(data, dict) else {})
        except Exception:
            tasks = {}

    for task_file in Path(WEB_OUTPUT_DIR).glob("*/task.json"):
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                task = json.load(f)
            task_id = task.get("id") or task_file.parent.name
            tasks[task_id] = task
        except Exception:
            continue
    return tasks


def _save_task_index_unlocked(tasks: dict):
    _write_json_atomic(TASK_INDEX_PATH, {"updated_at": _now_iso(), "tasks": tasks})


def _save_task_unlocked(task: dict, tasks: dict | None = None):
    if tasks is None:
        tasks = _load_task_index_unlocked()
    tasks[task["id"]] = task
    _write_json_atomic(_task_json_path(task["id"]), task)
    _save_task_index_unlocked(tasks)


def _get_task(task_id: str | None) -> dict | None:
    if not task_id:
        return None
    with _task_lock:
        return _load_task_index_unlocked().get(task_id)


def _update_task(task_id: str, **updates) -> dict | None:
    with _task_lock:
        tasks = _load_task_index_unlocked()
        task = tasks.get(task_id)
        if not task:
            return None
        task.update(updates)
        task["updated_at"] = _now_iso()
        _save_task_unlocked(task, tasks)
        return task


def _append_task_log(task_id: str, message: str, level: str = "INFO"):
    line = f"[{_now_iso()}] [{level}] {message}"
    with _task_lock:
        tasks = _load_task_index_unlocked()
        task = tasks.get(task_id)
        if not task:
            return
        output_dir = task.get("output_dir") or _task_output_dir(task_id)
        os.makedirs(output_dir, exist_ok=True)
        task_log_path = os.path.join(output_dir, "task.log")
        with open(task_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        with open(TASK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{task_id} {line}\n")
        try:
            with open(task_log_path, "r", encoding="utf-8") as f:
                log_tail = "\n".join(f.read().splitlines()[-200:])
        except Exception:
            log_tail = line
        task["log_path"] = task_log_path
        task["log_tail"] = log_tail
        task["updated_at"] = _now_iso()
        tasks[task_id] = task
        _save_task_unlocked(task, tasks)
    print(f"{task_id} {line}")


def _base_model_key() -> str:
    for key, info in scan_available_models().items():
        if info.get("kind") == "base":
            return key
    return "ACE-Step v1-3.5B 基线模型"


def _lora_style_prefix_for_load(load_info: dict | None) -> str:
    if not load_info or not load_info.get("lora_loaded"):
        return ""
    lora_path = load_info.get("lora_path") or ""
    if not lora_path or lora_path == "none":
        return ""
    manifest = _load_lora_manifest(lora_path)
    return (manifest.get("style_prefix") or manifest.get("trigger_word") or "").strip(" ,")


def _apply_lora_style_prefix(prompt: str, load_info: dict | None) -> str:
    prefix = _lora_style_prefix_for_load(load_info)
    text = (prompt or "").strip()
    if not prefix:
        return text
    if text.lower().startswith(prefix.lower()):
        return text
    return f"{prefix}, {text}" if text else prefix


def _pipeline_cache_key(model_kind: str, model_path: str, lora_weight: float) -> str:
    if model_kind in {"lora", "ckpt"}:
        return f"{model_kind}:{os.path.abspath(model_path)}:{float(lora_weight):.4f}"
    if model_kind == "full":
        return f"full:{os.path.abspath(model_path)}"
    return "base"


def _validate_lora_dir(path: str) -> tuple[bool, str]:
    if not path:
        return False, "LoRA 路径为空"
    weight_path = os.path.join(path, "pytorch_lora_weights.safetensors")
    if not os.path.exists(weight_path):
        return False, f"未找到 LoRA 权重文件: {weight_path}"
    return True, weight_path


def _task_choices() -> list[str]:
    with _task_lock:
        tasks = _load_task_index_unlocked()
    ordered = sorted(tasks.values(), key=lambda item: item.get("created_at", ""), reverse=True)
    return [task["id"] for task in ordered]


TASK_TABLE_HEADERS = [
    "任务ID",
    "状态",
    "进度",
    "角色",
    "类型",
    "模型",
    "LoRA请求",
    "LoRA已加载",
    "seed",
    "时长",
    "音频数",
    "创建时间",
    "完成时间",
    "输出目录",
    "提醒",
]

AUDIO_TABLE_HEADERS = [
    "标签",
    "角色",
    "模型",
    "文件名",
    "大小",
    "路径",
]


def _task_table_rows() -> list[list[str]]:
    with _task_lock:
        tasks = _load_task_index_unlocked()
    ordered = sorted(tasks.values(), key=lambda item: item.get("created_at", ""), reverse=True)
    rows = []
    for task in ordered[:200]:
        outputs = task.get("outputs") or {}
        files = outputs.get("files") or []
        warnings = task.get("warnings") or []
        rows.append([
            task.get("id", ""),
            task.get("status", ""),
            task.get("progress_label") or f"{task.get('progress_percent', 0)}%",
            task.get("compare_role", ""),
            task.get("kind", ""),
            task.get("model_key", ""),
            "是" if task.get("lora_requested") else "否",
            "是" if task.get("lora_loaded") else "否",
            str(task.get("seed", "")),
            str(task.get("duration", "")),
            str(len(files)),
            task.get("created_at", ""),
            task.get("finished_at", ""),
            task.get("output_dir", ""),
            " | ".join(warnings),
        ])
    return rows


def _existing_path(path: str | None) -> str | None:
    return path if path and os.path.exists(path) else None


def _display_role(task: dict | None) -> str:
    role = (task or {}).get("compare_role", "")
    if role == "selected_model":
        return "选择模型"
    if role == "baseline":
        return "基线"
    return role or "普通任务"


def _file_size_label(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return "未生成"
    size = os.path.getsize(path)
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MB"
    return f"{size / 1024:.1f} KB"


def _audio_entries_for_task(task: dict | None) -> list[dict[str, str]]:
    if not task:
        return []

    outputs = task.get("outputs") or {}
    stems = outputs.get("stems") or {}
    role = _display_role(task)
    model_key = task.get("model_key", "")
    entries: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(label: str, path: str | None):
        if not path or not os.path.exists(path):
            return
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            return
        seen.add(abs_path)
        entries.append({
            "label": label,
            "role": role,
            "model": model_key,
            "filename": os.path.basename(abs_path),
            "size": _file_size_label(abs_path),
            "path": abs_path,
        })

    add("完整混音", outputs.get("mix"))

    stem_labels = {
        "drums": "分离音轨 - 鼓 Drums",
        "bass": "分离音轨 - 贝斯 Bass",
        "vocals": "分离音轨 - 人声/旋律 Vocals",
        "other": "分离音轨 - 其他 Other",
        "accompaniment": "分离伴奏 - Drums+Bass+Other",
    }
    for stem_name, stem_path in stems.items():
        add(stem_labels.get(stem_name, f"分离音轨 - {stem_name}"), stem_path)

    for index, path in enumerate(outputs.get("files") or [], start=1):
        if path and os.path.exists(path):
            label = "批量音频" if task.get("kind") == "batch" else "其他音频"
            add(f"{label} {index:02d}", path)

    return entries


def _audio_table_rows(task: dict | None) -> list[list[str]]:
    return [
        [entry["label"], entry["role"], entry["model"], entry["filename"], entry["size"], entry["path"]]
        for entry in _audio_entries_for_task(task)
    ]


def _default_audio_selection(task: dict | None) -> tuple[str | None, str]:
    entries = _audio_entries_for_task(task)
    if not entries:
        return None, "该任务还没有可播放音频。"
    entry = entries[0]
    note = f"{entry['label']} | {entry['role']} | {entry['model']}\n{entry['path']}"
    return entry["path"], note


def _progress_frames_for_task(task: dict | None) -> list[str]:
    if not task:
        return []
    outputs = task.get("outputs") or {}
    visuals = []
    for key in ("process_overview", "trace_plot", "progress_image"):
        path = outputs.get(key)
        if path and os.path.exists(path):
            visuals.append(path)
    frames = [path for path in outputs.get("progress_frames", []) if path and os.path.exists(path)]
    if frames:
        if len(frames) <= 24:
            sampled = frames
        else:
            idx = np.linspace(0, len(frames) - 1, 24, dtype=int)
            sampled = [frames[i] for i in idx]
        visuals.extend(sampled)
    return visuals


def _waveform_for_task(task: dict | None) -> str | None:
    if not task:
        return None
    outputs = task.get("outputs") or {}
    return _existing_path(outputs.get("waveform"))


def _mel_for_task(task: dict | None) -> str | None:
    if not task:
        return None
    outputs = task.get("outputs") or {}
    mel_path = _existing_path(outputs.get("mel_spectrogram"))
    if mel_path:
        return mel_path
    for path in outputs.get("mel_spectrograms") or []:
        existing = _existing_path(path)
        if existing:
            return existing
    return None


def _save_ai_mel_for_audio(audio_path: str | None, output_dir: str, unique: str, prefix: str = "mel") -> str | None:
    if not audio_path or not os.path.exists(audio_path):
        return None
    name = f"{prefix}_{unique}_ai_watermark.png"
    return save_ai_watermarked_mel(audio_path, os.path.join(output_dir, name))


def _task_detail_values(task_id: str | None):
    task = _get_task(task_id)
    if not task:
        return None, None, None, None, None, "", "未选择任务。", None, None, None, [], [], None, "未选择任务。"

    outputs = task.get("outputs") or {}
    stems = outputs.get("stems") or {}
    files = [path for path in outputs.get("files", []) if os.path.exists(path)]
    mix_path = _existing_path(outputs.get("mix")) or (files[0] if files else None)
    selected_audio, selected_note = _default_audio_selection(task)
    status_parts = [
        f"任务: {task.get('id')}",
        f"状态: {task.get('status')}",
        f"进度: {task.get('progress_label') or str(task.get('progress_percent', 0)) + '%'}",
        f"角色: {_display_role(task)}",
        f"模型: {task.get('model_key')}",
        f"输出目录: {task.get('output_dir')}",
    ]
    if task.get("warnings"):
        status_parts.append("提醒: " + " | ".join(task.get("warnings", [])))
    if task.get("error"):
        status_parts.append("错误: " + task.get("error"))
    if task.get("log_tail"):
        status_parts.append("")
        status_parts.append(task["log_tail"])

    return (
        mix_path,
        _existing_path(stems.get("drums")),
        _existing_path(stems.get("bass")),
        _existing_path(stems.get("vocals")),
        _existing_path(stems.get("other")),
        task.get("analysis", ""),
        "\n".join(status_parts),
        _progress_frames_for_task(task),
        _waveform_for_task(task),
        _mel_for_task(task),
        files,
        _audio_table_rows(task),
        selected_audio,
        selected_note,
    )


def _mix_for_task(task: dict | None) -> str | None:
    if not task:
        return None
    outputs = task.get("outputs") or {}
    files = [path for path in outputs.get("files", []) if os.path.exists(path)]
    return _existing_path(outputs.get("mix")) or (files[0] if files else None)


def _compare_generation_values(task_ids: list[str]) -> tuple[dict | None, dict | None, str]:
    if not task_ids:
        return None, None, "未提交对比任务。"

    tasks = [_get_task(task_id) for task_id in task_ids]
    selected = next((task for task in tasks if task and task.get("compare_role") == "selected_model"), None)
    baseline = next((task for task in tasks if task and task.get("compare_role") == "baseline"), None)
    if selected is None:
        selected = tasks[0] if tasks and tasks[0] else None
    if baseline is None:
        baseline = next(
            (task for task in tasks if task and task.get("model_kind") == "base" and task.get("id") != (selected or {}).get("id")),
            None,
        )

    lines = []
    if selected:
        lines.append(
            f"选择模型音频: {selected.get('id')} | {selected.get('model_key')} | "
            f"{selected.get('status')} | {_mix_for_task(selected) or '等待生成'}"
        )
    if baseline:
        lines.append(
            f"基线对比音频: {baseline.get('id')} | {baseline.get('model_key')} | "
            f"{baseline.get('status')} | {_mix_for_task(baseline) or '等待生成'}"
        )
    elif len(task_ids) == 1:
        lines.append("没有单独的基线任务：当前选择的已经是基线模型，或未勾选基线对比。")
    else:
        lines.append("基线任务尚未找到，请看下方任务表确认状态。")

    return selected, baseline, "\n".join(lines)


def _parse_task_ids(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).replace("\n", ",").split(",") if part.strip()]


def refresh_task_dashboard(selected_task_id: str | None = None):
    choices = _task_choices()
    value = selected_task_id if selected_task_id in choices else (choices[0] if choices else None)
    return (_task_table_rows(), gr.update(choices=choices, value=value), *_task_detail_values(value))


def refresh_task_table_only():
    return _task_table_rows()


def load_selected_task_detail(selected_task_id: str | None = None):
    choices = _task_choices()
    value = selected_task_id if selected_task_id in choices else (choices[0] if choices else None)
    return _task_detail_values(value)


def select_task_from_table(evt: gr.SelectData):
    rows = _task_table_rows()
    row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    try:
        row_index = int(row_index)
    except (TypeError, ValueError):
        row_index = 0
    task_id = rows[row_index][0] if 0 <= row_index < len(rows) else None
    choices = _task_choices()
    value = task_id if task_id in choices else None
    return (gr.update(choices=choices, value=value), *_task_detail_values(value))


def select_audio_from_table(task_id: str | None, evt: gr.SelectData):
    task = _get_task(task_id)
    rows = _audio_table_rows(task)
    row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    try:
        row_index = int(row_index)
    except (TypeError, ValueError):
        row_index = 0
    if not rows or row_index < 0 or row_index >= len(rows):
        return None, "没有选中有效音频。"
    label, role, model, filename, size, path = rows[row_index]
    return path, f"{label} | {role} | {model} | {size}\n{path}"


def poll_generation_tasks(task_ids_text: str | None):
    task_ids = _parse_task_ids(task_ids_text)
    if not task_ids:
        return None, None, "还没有提交任务。", None, None, None, None, "", "还没有提交任务。", [], None, None, [], None, None, _task_table_rows()

    primary = task_ids[0]
    selected_task, baseline_task, compare_text = _compare_generation_values(task_ids)
    selected_mix = _mix_for_task(selected_task)
    baseline_mix = _mix_for_task(baseline_task)
    _, drums, bass, vocals, other, analysis, _, _, _, *_ = _task_detail_values(primary)
    status_blocks = []
    for task_id in task_ids:
        task = _get_task(task_id)
        if not task:
            status_blocks.append(f"{task_id}: 未找到")
            continue
        block = [
            f"任务: {task_id}",
            f"状态: {task.get('status')}",
            f"模型: {task.get('model_key')}",
        ]
        if task.get("warnings"):
            block.append("提醒: " + " | ".join(task.get("warnings", [])))
        if task.get("error"):
            block.append("错误: " + task.get("error"))
        if task.get("log_tail"):
            block.append(task["log_tail"])
        status_blocks.append("\n".join(block))
    return (
        selected_mix,
        baseline_mix,
        compare_text,
        drums,
        bass,
        vocals,
        other,
        analysis,
        "\n\n".join(status_blocks),
        _progress_frames_for_task(selected_task),
        _waveform_for_task(selected_task),
        _mel_for_task(selected_task),
        _progress_frames_for_task(baseline_task),
        _waveform_for_task(baseline_task),
        _mel_for_task(baseline_task),
        _task_table_rows(),
    )


def poll_batch_tasks(task_ids_text: str | None):
    task_ids = _parse_task_ids(task_ids_text)
    if not task_ids:
        return [], "还没有提交批量任务。", _task_table_rows()
    task = _get_task(task_ids[0])
    if not task:
        return [], "未找到任务。", _task_table_rows()
    files = [path for path in (task.get("outputs") or {}).get("files", []) if os.path.exists(path)]
    _, _, _, _, _, _, status_text, *_ = _task_detail_values(task_ids[0])
    return files, status_text, _task_table_rows()


def poll_reference_style_tasks(task_ids_text: str | None):
    task_ids = _parse_task_ids(task_ids_text)
    if not task_ids:
        return None, "还没有提交无训练参考音频任务。", "", [], None, None, [], _task_table_rows(), None, None, None, None
    task = _get_task(task_ids[0])
    if not task:
        return None, "未找到任务。", "", [], None, None, [], _task_table_rows(), None, None, None, None
    mix, drums, bass, vocals, other, analysis, status_text, progress, waveform, mel, files, *_ = _task_detail_values(task_ids[0])
    return mix, status_text, analysis, progress, waveform, mel, files, _task_table_rows(), drums, bass, vocals, other


def _matching_task_file(task: dict | None, prefixes: tuple[str, ...]) -> str | None:
    if not task:
        return None
    outputs = task.get("outputs") or {}
    for path in outputs.get("files") or []:
        if not path:
            continue
        name = os.path.basename(path)
        if any(name.startswith(prefix) for prefix in prefixes):
            existing = _existing_path(path)
            if existing:
                return existing
    output_dir = task.get("output_dir")
    if output_dir and os.path.isdir(output_dir):
        for prefix in prefixes:
            for path in sorted(Path(output_dir).glob(f"{prefix}*")):
                if path.is_file():
                    return str(path)
    return None


def _voice_converted_vocal_path(task: dict | None) -> str | None:
    outputs = (task or {}).get("outputs") or {}
    return (
        _existing_path(outputs.get("converted_vocal_reverb"))
        or _existing_path(outputs.get("converted_vocal"))
        or _matching_task_file(task, ("converted_vocal_reverb", "converted_vocal"))
    )


def _voice_accompaniment_path(task: dict | None) -> str | None:
    outputs = (task or {}).get("outputs") or {}
    stems = outputs.get("stems") or {}
    return (
        _existing_path(outputs.get("accompaniment"))
        or _existing_path(stems.get("accompaniment"))
        or _matching_task_file(task, ("accompaniment", "voice_accompaniment"))
    )


def _voice_conversion_detail_values(task_ids_text: str | None):
    task_ids = _parse_task_ids(task_ids_text)
    if not task_ids:
        return None, None, None, None, None, None, None, "还没有提交授权音色转换任务。", "", None, None, [], _task_table_rows()
    task = _get_task(task_ids[0])
    if not task:
        return None, None, None, None, None, None, None, "未找到任务。", "", None, None, [], _task_table_rows()
    mix, drums, bass, vocals, other, analysis, status_text, progress, waveform, mel, files, *_ = _task_detail_values(task_ids[0])
    converted_vocal = _voice_converted_vocal_path(task)
    accompaniment = _voice_accompaniment_path(task)
    return (
        mix,
        converted_vocal,
        vocals,
        accompaniment,
        drums,
        bass,
        other,
        status_text,
        analysis,
        waveform,
        mel,
        files,
        _task_table_rows(),
    )


def poll_voice_conversion_tasks(task_ids_text: str | None):
    return _voice_conversion_detail_values(task_ids_text)


def remix_voice_conversion_task(task_ids_text: str | None):
    task_ids = _parse_task_ids(task_ids_text)
    if not task_ids:
        return _voice_conversion_detail_values(task_ids_text)

    task_id = task_ids[0]
    task = _get_task(task_id)
    if not task:
        return _voice_conversion_detail_values(task_ids_text)

    outputs = task.get("outputs") or {}
    stems = outputs.get("stems") or {}
    output_dir = task.get("output_dir") or _task_dir(task_id)
    os.makedirs(output_dir, exist_ok=True)

    converted_vocal = _voice_converted_vocal_path(task)
    accompaniment = _voice_accompaniment_path(task)
    if not converted_vocal:
        _append_task_log(task_id, "无法重新合成：没有找到换音色后人声。", "WARN")
        return _voice_conversion_detail_values(task_ids_text)

    reverb_vocal = _existing_path(outputs.get("converted_vocal_reverb"))
    if not reverb_vocal:
        dry_vocal = _existing_path(outputs.get("converted_vocal_dry")) or converted_vocal
        reverb_candidate = os.path.join(output_dir, f"converted_vocal_reverb_{task_id}.wav")
        try:
            reverb_vocal = _apply_vocal_reverb_file(dry_vocal, reverb_candidate)
            outputs["converted_vocal_dry"] = dry_vocal
            outputs["converted_vocal_reverb"] = reverb_vocal
            _append_task_log(
                task_id,
                f"已给换音色后人声添加轻混响: wet={VOICE_REVERB_MIX:.2f}, tail={VOICE_REVERB_SECONDS:.2f}s",
            )
        except Exception as exc:
            reverb_vocal = converted_vocal
            _append_task_log(task_id, f"人声混响处理失败，改用原换音色人声重混: {exc}", "WARN")
    converted_vocal = reverb_vocal

    if not accompaniment:
        accompaniment_candidate = os.path.join(output_dir, f"accompaniment_{task_id}.wav")
        try:
            accompaniment = _build_accompaniment_from_stems(stems, accompaniment_candidate)
            if accompaniment:
                stems["accompaniment"] = accompaniment
        except Exception as exc:
            _append_task_log(task_id, f"重新生成伴奏失败: {exc}", "WARN")

    if not accompaniment:
        _append_task_log(task_id, "无法重新合成：没有可用伴奏，请先启用 Demucs 并成功分离 drums/bass/other。", "WARN")
        return _voice_conversion_detail_values(task_ids_text)

    remix_path = os.path.join(output_dir, f"voice_conversion_mix_{task_id}.wav")
    try:
        vocal_gain_db = float((task.get("params") or {}).get("vocal_gain_db", 1.5) or 0.0)
        _mix_converted_vocal_with_accompaniment(converted_vocal, accompaniment, remix_path, vocal_gain_db=vocal_gain_db)
    except Exception as exc:
        _append_task_log(task_id, f"重新合成新歌失败: {exc}", "ERROR")
        return _voice_conversion_detail_values(task_ids_text)

    files = list(outputs.get("files") or [])
    for path in (remix_path, converted_vocal, accompaniment):
        if path and os.path.exists(path) and path not in files:
            files.append(path)
    try:
        mel_path = _save_ai_mel_for_audio(remix_path, output_dir, task_id, prefix="mel_voice_conversion_mix")
    except Exception as exc:
        mel_path = None
        _append_task_log(task_id, f"重新合成后 Mel 图谱生成失败: {exc}", "WARN")
    outputs.update({
        "mix": remix_path,
        "converted_vocal": converted_vocal,
        "converted_vocal_reverb": converted_vocal,
        "accompaniment": accompaniment,
        "stems": stems,
        "files": files,
    })
    if mel_path:
        outputs["mel_spectrogram"] = mel_path
    _update_task(task_id, outputs=outputs, updated_at=_now_iso(), status_text=(task.get("status_text") or "") + f"\n已重新合成新歌: {remix_path}")
    _append_task_log(task_id, f"已重新合成新歌: {remix_path}")
    return _voice_conversion_detail_values(task_ids_text)


def _reset_pipeline():
    global _acestep_pipeline, _current_model_key, _current_model_kind, _current_lora_weight
    _acestep_pipeline = None
    _current_model_key = None
    _current_model_kind = None
    _current_lora_weight = None
    reset_acestep_model_cache()
    with _pipeline_cache_lock:
        _pipeline_cache.clear()


def get_pipeline(model_key: str = "", lora_weight: float = 1.0):
    """获取或加载 ACE-Step pipeline；不同模型配置使用独立缓存，避免并发任务互相切换 LoRA。"""
    global _acestep_pipeline, _current_model_key, _current_model_kind, _current_lora_weight, _last_model_load_summary

    available = scan_available_models()
    if not available:
        raise RuntimeError("没有找到可用模型")

    if model_key not in available:
        model_key = next(iter(available.keys()))

    model_info = available[model_key]
    model_kind = model_info["kind"]
    model_path = model_info.get("path", "")
    requested_lora_weight = float(lora_weight)

    if model_kind == "full" and os.environ.get("EDM_ALLOW_FULL_MODEL") != "1":
        raise RuntimeError(
            "旧全量 model.pt 默认禁止在网页加载，避免内存峰值导致服务退出。"
            "请改选“推荐 LoRA”或“完整 CKPT”。如确实要测试旧全量，"
            "先设置环境变量 EDM_ALLOW_FULL_MODEL=1 再启动网页。"
        )

    single_pipeline_cache = os.environ.get("EDM_SINGLE_PIPELINE_CACHE") == "1"
    cache_key = "single" if single_pipeline_cache else _pipeline_cache_key(model_kind, model_path, requested_lora_weight)
    with _pipeline_cache_lock:
        cached = _pipeline_cache.get(cache_key)
        if cached and not single_pipeline_cache:
            _last_model_load_summary = cached["summary"]
            _current_model_key = model_key
            _current_model_kind = model_kind
            _current_lora_weight = cached.get("lora_weight")
            return cached["pipeline"]

        if cached and single_pipeline_cache:
            pipeline = cached["pipeline"]
        else:
            pipeline = load_acestep_model(
                checkpoint_dir="",  # 空字符串会自动用 models/ace-step/
                device="auto",
                cpu_offload=True,
                dtype="bfloat16",
                cache_key=cache_key,
            )

        warnings = []
        lora_requested = model_kind in {"lora", "ckpt"}
        lora_loaded = False
        effective_lora_path = "none"
        effective_lora_weight = 1.0

        if model_kind == "lora":
            ok, detail = _validate_lora_dir(model_path)
            if not ok:
                warnings.append(f"前端选择了 LoRA，但没有加载：{detail}")
                raise RuntimeError(warnings[-1])
            print(f"加载 LoRA: {model_path} weight={requested_lora_weight}")
            pipeline.load_lora(model_path, requested_lora_weight)
            effective_lora_path = model_path
            effective_lora_weight = requested_lora_weight
            lora_loaded = getattr(pipeline, "lora_path", "none") != "none"
            if requested_lora_weight == 0:
                warnings.append("已加载 LoRA，但 LoRA 强度为 0，实际效果等同未使用 LoRA。")
            if not lora_loaded:
                warnings.append("前端选择了 LoRA，但 pipeline 当前没有激活 LoRA；本次不会按微调模型生成。")
            summary = f"已叠加 LoRA: {model_path} | weight={requested_lora_weight:.2f}"
        elif model_kind == "ckpt":
            lora_dir = _resolve_ckpt_lora_dir(model_path)
            source = "对应 LoRA"
            if not lora_dir:
                lora_dir = _extract_lora_from_ckpt(model_path)
                source = "从 CKPT 抽取 LoRA"
            ok, detail = _validate_lora_dir(lora_dir)
            if not ok:
                warnings.append(f"前端选择了 CKPT/LoRA，但没有加载：{detail}")
                raise RuntimeError(warnings[-1])
            print(f"加载完整 CKPT 对应 LoRA: {lora_dir} weight={requested_lora_weight}")
            pipeline.load_lora(lora_dir, requested_lora_weight)
            effective_lora_path = lora_dir
            effective_lora_weight = requested_lora_weight
            lora_loaded = getattr(pipeline, "lora_path", "none") != "none"
            if requested_lora_weight == 0:
                warnings.append("已加载 CKPT 对应 LoRA，但 LoRA 强度为 0，实际效果等同未使用 LoRA。")
            if not lora_loaded:
                warnings.append("前端选择了 CKPT/LoRA，但 pipeline 当前没有激活 LoRA；本次不会按微调模型生成。")
            summary = (
                f"已选择完整 CKPT: {model_path} | {source}: {lora_dir} "
                f"| weight={requested_lora_weight:.2f}"
            )
        elif model_kind == "full":
            result = apply_finetuned_weights(pipeline, model_path)
            summary = (
                "已加载全量微调权重: "
                f"{result['path']} | missing={result['missing_keys']} | unexpected={result['unexpected_keys']}"
            )
        else:
            pipeline.load_lora("none", 1.0)
            summary = "使用 ACE-Step 基线模型"
            if requested_lora_weight != 1.0:
                warnings.append("当前选择的是基线模型，LoRA 强度滑块不会生效。")

        load_info = {
            "cache_key": cache_key,
            "model_key": model_key,
            "model_kind": model_kind,
            "lora_requested": lora_requested,
            "lora_loaded": lora_loaded,
            "lora_path": effective_lora_path,
            "lora_weight": effective_lora_weight,
            "warnings": warnings,
            "summary": summary,
        }
        pipeline._edm_adapter_load_info = load_info
        _pipeline_cache[cache_key] = {"pipeline": pipeline, **load_info}
        _last_model_load_summary = summary
        _current_model_key = model_key
        _current_model_kind = model_kind
        _current_lora_weight = effective_lora_weight if lora_loaded else None
        return pipeline


# ============================================================
# 音源分离
# ============================================================

def separate_stems(audio_path: str, output_dir: str):
    """用 Demucs 分离音轨（鼓 / 贝斯 / 人声 / 其他）"""
    try:
        from src.demucs_wrapper import separate_stems as _separate
        return _separate(audio_path, output_dir)
    except Exception as e:
        print(f"Demucs 分离失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _voice_model_files(model_dir: str = VOICE_MODEL_DIR) -> dict[str, list[str]]:
    root = Path(model_dir)
    if not root.exists():
        return {"ckpt": [], "pth": [], "refs": []}
    refs_dir = root / "参考"
    return {
        "ckpt": [str(path) for path in sorted(root.glob("*.ckpt"))],
        "pth": [str(path) for path in sorted(root.glob("*.pth"))],
        "refs": [str(path) for path in sorted(refs_dir.glob("*")) if path.is_file()] if refs_dir.exists() else [],
    }


def get_voice_conversion_status_markdown() -> str:
    files = _voice_model_files()
    converter_cmd = os.environ.get(VOICE_CONVERTER_CMD_ENV, "").strip()
    seed_vc_ready = os.path.exists(SEED_VC_DIR) and os.path.exists(SEED_VC_WRAPPER_PATH)
    lines = [
        "## 授权音色转换",
        "",
        "当前入口用于你有权使用的目标音色模型。上传歌曲后可先分离人声，再调用外部 VC/RVC/SVC 推理命令完成音色转换和混音。",
        "",
        f"- 目标模型目录: `{VOICE_MODEL_DIR}`",
        f"- `.ckpt`: {len(files['ckpt'])} 个",
        f"- `.pth`: {len(files['pth'])} 个",
        f"- 参考音频: {len(files['refs'])} 个",
    ]
    if files["ckpt"] and files["pth"]:
        lines.append("- 本地文件形态更像 GPT-SoVITS TTS 组合；它不能直接完成“上传一首歌然后换声线”的 SVC 推理。")
    if converter_cmd:
        lines.append(f"- 外部转换器: 已配置 `{VOICE_CONVERTER_CMD_ENV}`，Web 进程只负责排队和调用，不常驻加载转换模型。")
    elif seed_vc_ready:
        lines.append("- 外部转换器: 已自动接入 Seed-VC 零样本 SVC；未配置环境变量时会使用默认 Seed-VC 桥接命令。")
        lines.append("- 当前默认启用 30 秒预览、30 步歌声转换、轻量去沙哑、源人声力度包络保留和可调人声重混增益。")
        lines.append("- CPU 跑整首歌会非常慢；如需整首请把最大转换时长设为 0，并降低步数或使用 CUDA。")
    else:
        lines.append(f"- 外部转换器: 未配置 `{VOICE_CONVERTER_CMD_ENV}`；提交后会给出配置诊断，不会加载不明权重。")
    lines.append("")
    lines.append("命令模板可使用 `{input}`、`{output}`、`{pitch}`、`{model_dir}`、`{task_dir}`、`{steps}`、`{cfg}` 占位符。")
    return "\n".join(lines)


def _quote_cmd_value(value: str) -> str:
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'


def _render_voice_converter_cmd(
    template: str,
    input_path: str,
    output_path: str,
    pitch_shift: int,
    model_dir: str,
    task_dir: str,
    diffusion_steps: int = 30,
    inference_cfg_rate: float = 0.75,
) -> str:
    replacements = {
        "{input}": _quote_cmd_value(input_path),
        "{output}": _quote_cmd_value(output_path),
        "{pitch}": str(int(pitch_shift)),
        "{model_dir}": _quote_cmd_value(model_dir),
        "{task_dir}": _quote_cmd_value(task_dir),
        "{diffusion_steps}": str(int(diffusion_steps)),
        "{steps}": str(int(diffusion_steps)),
        "{cfg}": f"{float(inference_cfg_rate):.4f}",
    }
    command = template
    for placeholder, value in replacements.items():
        command = command.replace(placeholder, value)
    return command


def _copy_uploaded_audio(source_audio: str, output_dir: str) -> str:
    suffix = os.path.splitext(source_audio)[1] or ".wav"
    target = os.path.join(output_dir, f"uploaded_song{suffix}")
    shutil.copy2(source_audio, target)
    return target


def _as_stereo_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    if audio.shape[0] > 2:
        audio = audio[:2, :]
    return audio.astype(np.float32, copy=False)


def _sum_audio_paths(paths: list[str], sr: int = 44100, gains: list[float] | None = None) -> tuple[np.ndarray, int]:
    parts: list[np.ndarray] = []
    max_len = 0
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        audio, loaded_sr = load_audio(path, sr=sr, mono=False)
        sr = loaded_sr
        audio = _as_stereo_audio(audio)
        parts.append(audio)
        max_len = max(max_len, audio.shape[-1])
    if not parts or max_len <= 0:
        raise ValueError("没有可合成的音频文件")

    mix = np.zeros((2, max_len), dtype=np.float32)
    if gains is None:
        gains = [1.0] * len(parts)
    for audio, gain in zip(parts, gains):
        mix[:, : audio.shape[-1]] += audio[:, :max_len] * float(gain)
    return normalize_audio(mix, peak_db=-1.0), sr


def _apply_sos(audio: np.ndarray, sos) -> np.ndarray:
    from scipy.signal import sosfilt, sosfiltfilt

    try:
        return sosfiltfilt(sos, audio, axis=-1).astype(np.float32)
    except ValueError:
        return sosfilt(sos, audio, axis=-1).astype(np.float32)


def _vocal_dehiss_filter(audio: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """Lightweight vocal cleanup for Seed-VC artifacts.

    This targets high-frequency hiss and de-essing without changing timing or
    pitch. It is intentionally conservative; stronger timbre changes should be
    handled by the converter/reference, not by EQ.
    """
    from scipy.signal import butter

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return audio.astype(np.float32, copy=False)

    y = _as_stereo_audio(audio).copy()
    nyquist = sr * 0.5

    hp_hz = min(95.0, 45.0 + 35.0 * strength)
    y = _apply_sos(y, butter(2, hp_hz, btype="highpass", fs=sr, output="sos"))

    deess_low = min(5200.0, nyquist * 0.45)
    deess_high = min(10500.0, nyquist * 0.86)
    if deess_high > deess_low + 200:
        sibilance = _apply_sos(y, butter(2, [deess_low, deess_high], btype="bandpass", fs=sr, output="sos"))
        y = y - (0.10 + 0.22 * strength) * sibilance

    hiss_cut = min(9000.0, nyquist * 0.80)
    if strength >= 0.25 and hiss_cut < nyquist * 0.95:
        hiss = _apply_sos(y, butter(2, hiss_cut, btype="highpass", fs=sr, output="sos"))
        y = y - (0.08 + 0.08 * strength) * hiss

    lowpass_hz = min(nyquist * 0.92, 17500.0 - 4200.0 * strength)
    if lowpass_hz > 6000:
        y = _apply_sos(y, butter(2, lowpass_hz, btype="lowpass", fs=sr, output="sos"))

    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _smoothed_envelope(audio: np.ndarray, sr: int, window_seconds: float = 0.055) -> np.ndarray:
    mono = np.mean(_as_stereo_audio(audio), axis=0)
    window = max(128, int(sr * window_seconds))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.sqrt(np.convolve(mono * mono, kernel, mode="same") + 1e-9).astype(np.float32)


def _match_source_vocal_dynamics(converted: np.ndarray, source: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """Blend source singing dynamics back into the converted vocal."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return converted.astype(np.float32, copy=False)

    y = _as_stereo_audio(converted).copy()
    source_env = _smoothed_envelope(source, sr)
    target_len = y.shape[-1]
    if len(source_env) != target_len:
        old_x = np.linspace(0.0, 1.0, num=len(source_env), dtype=np.float32)
        new_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
        source_env = np.interp(new_x, old_x, source_env).astype(np.float32)

    converted_env = _smoothed_envelope(y, sr)
    source_ref = float(np.percentile(source_env, 90) + 1e-5)
    converted_ref = float(np.percentile(converted_env, 90) + 1e-5)
    source_norm = source_env / source_ref
    converted_norm = converted_env / converted_ref
    gain = (source_norm + 0.05) / (converted_norm + 0.05)
    gain = np.clip(gain, 0.68, 1.55)

    smooth_window = max(256, int(sr * 0.12))
    kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
    gain = np.convolve(gain, kernel, mode="same")
    gain = 1.0 + strength * (gain - 1.0)
    return (y * gain[np.newaxis, :]).astype(np.float32)


def _audio_rms_value(audio: np.ndarray) -> float:
    y = _as_stereo_audio(audio)
    return float(np.sqrt(np.mean(y * y) + 1e-12))


def _level_vocal_against_source(processed: np.ndarray, source: np.ndarray | None, peak_db: float = -1.5) -> np.ndarray:
    y = _as_stereo_audio(processed).astype(np.float32, copy=True)
    processed_rms = _audio_rms_value(y)
    source_rms = _audio_rms_value(source) if source is not None else 0.0

    if source_rms >= 0.003:
        target_rms = float(np.clip(source_rms * 1.10, 0.018, 0.11))
        max_gain = 32.0
    else:
        # If the separated vocal is basically empty, do not turn vocoder noise
        # into a loud foreground track.
        target_rms = float(np.clip(max(processed_rms, source_rms * 1.10), 0.0005, 0.006))
        max_gain = 3.0

    gain = float(np.clip(target_rms / (processed_rms + 1e-8), 0.05, max_gain))
    y *= gain

    peak = float(np.max(np.abs(y)) + 1e-8)
    target_peak = 10 ** (peak_db / 20.0)
    if peak > target_peak:
        y *= target_peak / peak
    return y.astype(np.float32)


def _prepare_vocal_for_seed_vc(input_path: str, output_path: str, dehiss_strength: float) -> str:
    audio, sr = load_audio(input_path, sr=44100, mono=False)
    cleaned = _vocal_dehiss_filter(audio, sr, min(0.55, max(0.0, dehiss_strength) * 0.65))
    cleaned = normalize_audio(cleaned, peak_db=-2.0)
    save_audio(output_path, cleaned, sr=sr, subtype="PCM_24")
    return output_path


def _trim_audio_file(input_path: str, output_path: str, max_seconds: float) -> str:
    audio, sr = load_audio(input_path, sr=44100, mono=False)
    max_samples = max(1, int(float(max_seconds) * sr))
    trimmed = audio[..., : min(audio.shape[-1], max_samples)]
    save_audio(output_path, trimmed, sr=sr, subtype="PCM_24")
    return output_path


def _seed_vc_cpu_rtf_estimate(diffusion_steps: int) -> float:
    # Observed on this machine: 8-step CPU singing conversion was about RTF=13.
    # Diffusion inference cost scales roughly with the step count.
    return 13.0 * max(4, int(diffusion_steps)) / 8.0


def _estimate_seed_vc_seconds(duration_seconds: float, diffusion_steps: int) -> float:
    if torch.cuda.is_available():
        return max(30.0, duration_seconds * max(1.5, diffusion_steps / 12.0))
    return 180.0 + duration_seconds * _seed_vc_cpu_rtf_estimate(diffusion_steps)


def _safe_seed_vc_preview_seconds(diffusion_steps: int, timeout_seconds: int) -> int:
    if torch.cuda.is_available():
        return 240
    budget = max(300.0, timeout_seconds * 0.80 - 180.0)
    return max(10, int(budget / _seed_vc_cpu_rtf_estimate(diffusion_steps)))


def _postprocess_converted_vocal(
    raw_path: str,
    source_vocal_path: str | None,
    output_path: str,
    dehiss_strength: float,
    emotion_strength: float,
) -> str:
    converted, sr = load_audio(raw_path, sr=44100, mono=False)
    processed = _vocal_dehiss_filter(converted, sr, dehiss_strength)
    source_vocal = None
    if source_vocal_path and os.path.exists(source_vocal_path) and emotion_strength > 0:
        source_vocal, _ = load_audio(source_vocal_path, sr=sr, mono=False)
        processed = _match_source_vocal_dynamics(processed, source_vocal, sr, emotion_strength)
    elif source_vocal_path and os.path.exists(source_vocal_path):
        source_vocal, _ = load_audio(source_vocal_path, sr=sr, mono=False)
    processed = _level_vocal_against_source(processed, source_vocal, peak_db=-1.5)
    save_audio(output_path, processed, sr=sr, subtype="PCM_24")
    return output_path


def _limit_audio_peak(audio: np.ndarray, peak_db: float = -1.5) -> np.ndarray:
    y = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=True)
    peak = float(np.max(np.abs(y)) + 1e-8)
    target_peak = 10 ** (float(peak_db) / 20.0)
    if peak > target_peak:
        y *= target_peak / peak
    return y.astype(np.float32)


def _apply_light_vocal_reverb(
    audio: np.ndarray,
    sr: int,
    wet_mix: float | None = None,
    room_seconds: float | None = None,
    pre_delay_ms: float | None = None,
) -> np.ndarray:
    wet_mix = VOICE_REVERB_MIX if wet_mix is None else float(wet_mix)
    room_seconds = VOICE_REVERB_SECONDS if room_seconds is None else float(room_seconds)
    pre_delay_ms = VOICE_REVERB_PRE_DELAY_MS if pre_delay_ms is None else float(pre_delay_ms)
    wet_mix = float(np.clip(wet_mix, 0.0, 0.35))
    room_seconds = float(np.clip(room_seconds, 0.15, 2.20))
    pre_delay_ms = float(np.clip(pre_delay_ms, 0.0, 80.0))

    y = _as_stereo_audio(audio).astype(np.float32, copy=True)
    if wet_mix <= 0.0 or y.shape[-1] < 32:
        return y

    from scipy.signal import butter

    n = y.shape[-1]
    ir_len = max(int(sr * room_seconds), int(sr * 0.15))
    pre_delay = min(ir_len - 1, int(sr * pre_delay_ms / 1000.0))
    rng = np.random.default_rng(20260603)
    t = np.arange(ir_len, dtype=np.float32) / float(sr)
    decay = np.exp(-t / max(0.08, room_seconds * 0.32)).astype(np.float32)

    tail = rng.standard_normal((2, ir_len)).astype(np.float32) * decay[np.newaxis, :]
    tail[:, :pre_delay] = 0.0
    lowpass_hz = min(8200.0, sr * 0.44)
    if lowpass_hz > 1200.0:
        tail = _apply_sos(tail, butter(2, lowpass_hz, btype="lowpass", fs=sr, output="sos"))

    ir = tail * 0.10
    early_taps = (
        (0.000, 0.32, 0.24),
        (0.031, 0.20, -0.14),
        (0.057, -0.15, 0.19),
        (0.083, 0.11, 0.13),
        (0.129, -0.08, 0.07),
    )
    for seconds, left_gain, right_gain in early_taps:
        idx = min(ir_len - 1, pre_delay + int(sr * seconds))
        ir[0, idx] += left_gain
        ir[1, idx] += right_gain

    energy = np.sqrt(np.sum(ir * ir, axis=1, keepdims=True) + 1e-8)
    ir = ir / energy * 0.58

    try:
        from scipy.signal import oaconvolve as _convolve
    except ImportError:
        from scipy.signal import fftconvolve as _convolve

    wet = np.zeros_like(y)
    for ch in range(2):
        wet[ch] = _convolve(y[ch], ir[ch], mode="full")[:n].astype(np.float32)

    nyquist = sr * 0.5
    wet = _apply_sos(wet, butter(2, 150.0, btype="highpass", fs=sr, output="sos"))
    wet_lowpass = min(9800.0, nyquist * 0.86)
    if wet_lowpass > 2000.0:
        wet = _apply_sos(wet, butter(2, wet_lowpass, btype="lowpass", fs=sr, output="sos"))

    dry_rms = _audio_rms_value(y)
    wet_rms = _audio_rms_value(wet)
    if dry_rms > 1e-6 and wet_rms > 1e-6:
        wet *= float(np.clip(dry_rms / wet_rms, 0.25, 4.0))

    reverbed = y * (1.0 - wet_mix * 0.05) + wet * wet_mix
    return _limit_audio_peak(reverbed, peak_db=-1.5)


def _apply_vocal_reverb_file(input_path: str, output_path: str) -> str:
    audio, sr = load_audio(input_path, sr=44100, mono=False)
    reverbed = _apply_light_vocal_reverb(audio, sr)
    save_audio(output_path, reverbed, sr=sr, subtype="PCM_24")
    return output_path


def _build_accompaniment_from_stems(stems: dict[str, str], output_path: str) -> str | None:
    stem_paths = [
        stems.get(name)
        for name in ("drums", "bass", "other")
        if stems.get(name) and os.path.exists(stems[name])
    ]
    if not stem_paths:
        return None
    mix, sr = _sum_audio_paths(stem_paths, sr=44100)
    save_audio(output_path, mix, sr=sr)
    return output_path


def _mix_converted_vocal_with_accompaniment(
    converted_vocal_path: str,
    accompaniment_path: str,
    output_path: str,
    vocal_gain_db: float = 0.0,
) -> tuple[str, int]:
    vocal_gain = 10 ** (float(vocal_gain_db) / 20.0)
    mix, sr = _sum_audio_paths([converted_vocal_path, accompaniment_path], sr=44100, gains=[vocal_gain, 1.0])
    save_audio(output_path, mix, sr=sr)
    return output_path, sr


def _mix_converted_vocal_with_stems(
    converted_vocal_path: str,
    stems: dict[str, str],
    output_path: str,
    vocal_gain_db: float = 0.0,
) -> tuple[str, int]:
    stem_paths = [
        stems.get(name)
        for name in ("drums", "bass", "other")
        if stems.get(name) and os.path.exists(stems[name])
    ]
    gains = [10 ** (float(vocal_gain_db) / 20.0), *([1.0] * len(stem_paths))]
    mix, sr = _sum_audio_paths([converted_vocal_path, *stem_paths], sr=44100, gains=gains)
    save_audio(output_path, mix, sr=sr)
    return output_path, sr


def generate_voice_conversion(
    source_audio: str,
    model_dir: str,
    pitch_shift: int,
    use_demucs: bool,
    diffusion_steps: int = 30,
    inference_cfg_rate: float = 0.75,
    dehiss_strength: float = 0.40,
    emotion_strength: float = 0.55,
    vocal_gain_db: float = 1.5,
    max_vocal_seconds: int = 30,
    output_dir: str | None = None,
    task_id: str = "",
    log_callback=None,
):
    status_lines: list[str] = []

    def emit(message: str, level: str = "INFO"):
        status_lines.append(message)
        if log_callback:
            log_callback(message, level)

    if output_dir is None:
        output_dir = os.path.join(WEB_OUTPUT_DIR, f"voice_conversion_{_task_timestamp()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(output_dir, exist_ok=True)

    if not source_audio or not os.path.exists(source_audio):
        return None, {}, "请先上传一首歌。", "请先上传一首歌。", None, [], {}

    model_dir = os.path.abspath(model_dir or VOICE_MODEL_DIR)
    diffusion_steps = int(np.clip(int(diffusion_steps or 30), 4, 50))
    inference_cfg_rate = float(np.clip(float(inference_cfg_rate or 0.75), 0.0, 1.5))
    dehiss_strength = float(np.clip(float(dehiss_strength or 0.0), 0.0, 1.0))
    emotion_strength = float(np.clip(float(emotion_strength or 0.0), 0.0, 1.0))
    vocal_gain_db = float(np.clip(float(vocal_gain_db or 0.0), -9.0, 9.0))
    max_vocal_seconds = int(np.clip(int(max_vocal_seconds or 0), 0, 600))
    uploaded_path = _copy_uploaded_audio(source_audio, output_dir)
    emit(f"输入音频已保存: {uploaded_path}")
    emit(f"目标模型目录: {model_dir}")
    emit(f"Pitch shift: {int(pitch_shift)} semitone(s)")
    emit(f"Seed-VC 扩散步数: {diffusion_steps} | CFG: {inference_cfg_rate:.2f}")
    emit(f"去沙哑强度: {dehiss_strength:.2f} | 情感/力度保留: {emotion_strength:.2f} | 人声重混增益: {vocal_gain_db:+.1f} dB")
    emit(f"换音色后人声轻混响: wet={VOICE_REVERB_MIX:.2f} | tail={VOICE_REVERB_SECONDS:.2f}s | pre-delay={VOICE_REVERB_PRE_DELAY_MS:.0f}ms")
    emit("转换时长: 整首" if max_vocal_seconds <= 0 else f"转换时长上限: {max_vocal_seconds}s（CPU 推荐先做短预览）")

    if task_id:
        _update_task(task_id, progress_percent=10, progress_label="准备输入音频")

    stems = {}
    accompaniment_path = None
    conversion_input = uploaded_path
    conversion_source_for_envelope = uploaded_path
    processed_conversion_input = None
    trim_preview_path = None
    trim_envelope_path = None
    full_accompaniment_path = None
    if use_demucs:
        emit("正在用 Demucs 分离人声/伴奏...")
        if task_id:
            _update_task(task_id, progress_percent=20, progress_label="分离人声")
        stems = separate_stems(uploaded_path, output_dir) or {}
        if stems:
            accompaniment_candidate = os.path.join(output_dir, f"accompaniment_{task_id or _task_timestamp()}.wav")
            try:
                accompaniment_path = _build_accompaniment_from_stems(stems, accompaniment_candidate)
                if accompaniment_path:
                    stems["accompaniment"] = accompaniment_path
                    full_accompaniment_path = accompaniment_path
                    emit(f"已生成分离伴奏: {accompaniment_path}")
                else:
                    emit("Demucs 没有产出 drums/bass/other，无法生成单独伴奏。", "WARN")
            except Exception as exc:
                emit(f"生成分离伴奏失败: {exc}", "WARN")
        if stems.get("vocals") and os.path.exists(stems["vocals"]):
            conversion_input = stems["vocals"]
            conversion_source_for_envelope = stems["vocals"]
            emit(f"使用分离后人声作为转换输入: {conversion_input}")
            if dehiss_strength > 0:
                try:
                    processed_conversion_input = os.path.join(output_dir, f"seed_vc_input_cleaned_{task_id or _task_timestamp()}.wav")
                    conversion_input = _prepare_vocal_for_seed_vc(conversion_input, processed_conversion_input, dehiss_strength)
                    emit(f"已对分离人声做轻量去沙预处理: {conversion_input}")
                except Exception as exc:
                    processed_conversion_input = None
                    conversion_input = stems["vocals"]
                    emit(f"分离人声预处理失败，改用原始 vocals: {exc}", "WARN")
        else:
            emit("Demucs 未产出 vocals，改用原始上传音频作为转换输入。", "WARN")
    else:
        emit("跳过 Demucs，直接把上传音频交给转换器。")

    try:
        conversion_duration = float(get_audio_duration(conversion_input))
    except Exception:
        conversion_duration = 0.0
    original_conversion_duration = conversion_duration
    if max_vocal_seconds > 0 and conversion_duration > max_vocal_seconds:
        trim_preview_path = os.path.join(output_dir, f"seed_vc_input_preview_{task_id or _task_timestamp()}_{max_vocal_seconds}s.wav")
        conversion_input = _trim_audio_file(conversion_input, trim_preview_path, max_vocal_seconds)
        if conversion_source_for_envelope and os.path.exists(conversion_source_for_envelope):
            trim_envelope_path = os.path.join(output_dir, f"seed_vc_envelope_preview_{task_id or _task_timestamp()}_{max_vocal_seconds}s.wav")
            conversion_source_for_envelope = _trim_audio_file(conversion_source_for_envelope, trim_envelope_path, max_vocal_seconds)
        if accompaniment_path and os.path.exists(accompaniment_path):
            full_accompaniment_path = accompaniment_path
            preview_accompaniment = os.path.join(output_dir, f"accompaniment_preview_{task_id or _task_timestamp()}_{max_vocal_seconds}s.wav")
            accompaniment_path = _trim_audio_file(accompaniment_path, preview_accompaniment, max_vocal_seconds)
            stems["accompaniment"] = accompaniment_path
        conversion_duration = float(get_audio_duration(conversion_input))
        emit(f"源人声 {original_conversion_duration:.1f}s 已裁剪为 {conversion_duration:.1f}s 预览，避免 CPU 长时间超时。", "WARN")

    estimated_seconds = _estimate_seed_vc_seconds(conversion_duration or 1.0, diffusion_steps)
    emit(f"Seed-VC 预计耗时: 约 {estimated_seconds / 60.0:.1f} 分钟（当前超时 {_format_timeout_label(VOICE_CONVERTER_TIMEOUT_SEC)}）")
    if VOICE_CONVERTER_TIMEOUT_SEC is not None and estimated_seconds > VOICE_CONVERTER_TIMEOUT_SEC * 0.90:
        safe_seconds = _safe_seed_vc_preview_seconds(diffusion_steps, VOICE_CONVERTER_TIMEOUT_SEC)
        message = (
            f"预计 Seed-VC 会超时：输入 {conversion_duration:.1f}s，steps={diffusion_steps}，"
            f"CPU 估计约 {estimated_seconds / 60.0:.1f} 分钟。\n"
            f"请把“最大转换时长”调到 {safe_seconds}s 以内，或把扩散步数降到 8-12，"
            "或在有 CUDA 的环境运行；也可以提高 EDM_VOICE_CONVERTER_TIMEOUT_SEC 后再跑整首。"
        )
        emit(message, "ERROR")
        with open(os.path.join(output_dir, "status.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(status_lines))
        files = [uploaded_path, *stems.values()]
        for path in (processed_conversion_input, trim_preview_path, trim_envelope_path, full_accompaniment_path):
            if path and os.path.exists(path):
                files.append(path)
        return None, stems, message, "\n".join(status_lines), None, files, {
            "accompaniment": accompaniment_path,
            "full_accompaniment": full_accompaniment_path,
            "processed_conversion_input": processed_conversion_input,
            "preview_conversion_input": trim_preview_path,
        }

    converter_template = os.environ.get(VOICE_CONVERTER_CMD_ENV, "").strip()
    if not converter_template and os.path.exists(SEED_VC_DIR) and os.path.exists(SEED_VC_WRAPPER_PATH):
        converter_template = DEFAULT_VOICE_CONVERTER_CMD
    if not converter_template:
        message = (
            f"未配置 {VOICE_CONVERTER_CMD_ENV}，无法执行真正的音色转换。\n"
            "当前本地 ckpt/pth 更像 GPT-SoVITS TTS 模型，不是直接的歌曲 SVC/RVC 推理入口。\n"
            "请配置一个你有权使用的外部 VC/RVC/SVC 命令模板，或安装 Seed-VC 后再提交。"
        )
        emit(message, "ERROR")
        with open(os.path.join(output_dir, "status.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(status_lines))
        return None, stems, message, "\n".join(status_lines), None, [uploaded_path, *stems.values()], {
            "accompaniment": accompaniment_path,
        }

    converted_vocal_raw_path = os.path.join(output_dir, f"converted_vocal_raw_{task_id or _task_timestamp()}.wav")
    converted_vocal_dry_path = os.path.join(output_dir, f"converted_vocal_dry_{task_id or _task_timestamp()}.wav")
    converted_vocal_path = os.path.join(output_dir, f"converted_vocal_reverb_{task_id or _task_timestamp()}.wav")
    command = _render_voice_converter_cmd(
        converter_template,
        input_path=conversion_input,
        output_path=converted_vocal_raw_path,
        pitch_shift=int(pitch_shift),
        model_dir=model_dir,
        task_dir=output_dir,
        diffusion_steps=diffusion_steps,
        inference_cfg_rate=inference_cfg_rate,
    )
    emit("正在调用外部音色转换器...")
    emit(command)
    if task_id:
        _update_task(task_id, progress_percent=55, progress_label="音色转换")

    proc_env = os.environ.copy()
    proc_env["SEEDVC_DIFFUSION_STEPS"] = str(diffusion_steps)
    proc_env["SEEDVC_CFG_RATE"] = str(inference_cfg_rate)
    proc_env["SEEDVC_TIMEOUT_SEC"] = "0" if VOICE_CONVERTER_TIMEOUT_SEC is None else str(VOICE_CONVERTER_TIMEOUT_SEC)
    try:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            shell=True,
            capture_output=True,
            text=True,
            timeout=VOICE_CONVERTER_TIMEOUT_SEC,
            env=proc_env,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_stdout = exc.stdout or ""
        timeout_stderr = exc.stderr or ""
        if isinstance(timeout_stdout, bytes):
            timeout_stdout = timeout_stdout.decode("utf-8", errors="replace")
        if isinstance(timeout_stderr, bytes):
            timeout_stderr = timeout_stderr.decode("utf-8", errors="replace")
        with open(os.path.join(output_dir, "converter_stdout.txt"), "w", encoding="utf-8", errors="replace") as f:
            f.write(timeout_stdout)
        with open(os.path.join(output_dir, "converter_stderr.txt"), "w", encoding="utf-8", errors="replace") as f:
            f.write(timeout_stderr)
        message = (
            f"Seed-VC 超时：输入 {conversion_duration:.1f}s，steps={diffusion_steps}，"
            f"超过 {_format_timeout_label(VOICE_CONVERTER_TIMEOUT_SEC)} 限制。请缩短最大转换时长、降低步数或使用 CUDA。"
        )
        emit(message, "ERROR")
        return None, stems, message, "\n".join(status_lines), None, [uploaded_path, *stems.values()], {
            "accompaniment": accompaniment_path,
            "converted_vocal_raw": converted_vocal_raw_path,
            "processed_conversion_input": processed_conversion_input,
            "preview_conversion_input": trim_preview_path,
        }
    with open(os.path.join(output_dir, "converter_stdout.txt"), "w", encoding="utf-8", errors="replace") as f:
        f.write(proc.stdout or "")
    with open(os.path.join(output_dir, "converter_stderr.txt"), "w", encoding="utf-8", errors="replace") as f:
        f.write(proc.stderr or "")
    if proc.returncode != 0:
        message = f"外部转换器失败，returncode={proc.returncode}\n{(proc.stderr or proc.stdout or '').strip()}"
        emit(message, "ERROR")
        return None, stems, message, "\n".join(status_lines), None, [uploaded_path, *stems.values()], {
            "accompaniment": accompaniment_path,
        }
    if not os.path.exists(converted_vocal_raw_path):
        message = f"外部转换器执行完成，但未生成输出文件: {converted_vocal_raw_path}"
        emit(message, "ERROR")
        return None, stems, message, "\n".join(status_lines), None, [uploaded_path, *stems.values()], {
            "accompaniment": accompaniment_path,
        }
    try:
        _postprocess_converted_vocal(
            converted_vocal_raw_path,
            conversion_source_for_envelope,
            converted_vocal_dry_path,
            dehiss_strength=dehiss_strength,
            emotion_strength=emotion_strength,
        )
        emit(f"已生成修复后干声人声: {converted_vocal_dry_path}")
    except Exception as exc:
        shutil.copy2(converted_vocal_raw_path, converted_vocal_dry_path)
        emit(f"人声后处理失败，保留原始转换输出: {exc}", "WARN")
    try:
        _apply_vocal_reverb_file(converted_vocal_dry_path, converted_vocal_path)
        emit(f"已给换音色后人声添加轻混响: {converted_vocal_path}")
    except Exception as exc:
        shutil.copy2(converted_vocal_dry_path, converted_vocal_path)
        emit(f"人声混响处理失败，改用干声合成: {exc}", "WARN")

    mix_path = converted_vocal_path
    if use_demucs and accompaniment_path:
        remix_path = os.path.join(output_dir, f"voice_conversion_mix_{task_id or _task_timestamp()}.wav")
        try:
            mix_path, _ = _mix_converted_vocal_with_accompaniment(
                converted_vocal_path,
                accompaniment_path,
                remix_path,
                vocal_gain_db=vocal_gain_db,
            )
            emit(f"已把转换后人声混回伴奏: {mix_path}")
        except Exception as exc:
            emit(f"混回伴奏失败，保留转换后人声: {exc}", "WARN")
    else:
        mix_audio, mix_sr = load_audio(mix_path, sr=44100, mono=False)
        save_audio(mix_path, normalize_audio(mix_audio, peak_db=-1.0), sr=mix_sr)

    if task_id:
        _update_task(task_id, progress_percent=88, progress_label="分析音频")

    try:
        y_mono, sr = load_audio(mix_path, sr=44100, mono=True)
        features = extract_all_features(y_mono, sr)
        analysis = (
            f"BPM (检测): {features['bpm']:.1f}\n"
            f"RMS 能量: {features['rms_mean']:.4f}\n"
            f"低频比例: {features['low_freq_ratio']:.4f}\n"
            f"频谱质心: {features['spectral_centroid_mean']:.1f} Hz\n"
            f"起音密度: {features['onset_density']:.2f} /秒\n"
            f"实际时长: {len(y_mono) / sr:.1f}s\n"
            f"Pitch shift: {int(pitch_shift)} semitone(s)\n"
            f"转换输入时长: {conversion_duration:.1f}s / 原始 {original_conversion_duration:.1f}s\n"
            f"Seed-VC diffusion steps: {diffusion_steps}\n"
            f"Seed-VC CFG: {inference_cfg_rate:.2f}\n"
            f"去沙哑强度: {dehiss_strength:.2f}\n"
            f"情感/力度保留: {emotion_strength:.2f}\n"
            f"人声重混增益: {vocal_gain_db:+.1f} dB\n"
            f"换音色后人声轻混响: wet={VOICE_REVERB_MIX:.2f}, tail={VOICE_REVERB_SECONDS:.2f}s, pre-delay={VOICE_REVERB_PRE_DELAY_MS:.0f}ms"
        )
        waveform_path = _plot_waveform(y_mono, sr, output_dir=output_dir, name=f"waveform_voice_conversion_{task_id or _task_timestamp()}.png")
    except Exception as exc:
        analysis = f"音频分析失败: {exc}"
        waveform_path = None
        emit(analysis, "WARN")

    try:
        mel_path = _save_ai_mel_for_audio(mix_path, output_dir, task_id or _task_timestamp(), prefix="mel_voice_conversion_mix")
        emit(f"完整 Mel 图谱（含末尾频谱负形 AI 水印）已保存: {mel_path}")
    except Exception as exc:
        mel_path = None
        emit(f"Mel 图谱生成失败: {exc}", "WARN")

    with open(os.path.join(output_dir, "analysis.txt"), "w", encoding="utf-8") as f:
        f.write(analysis)
    with open(os.path.join(output_dir, "status.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(status_lines))

    files = []
    seen_files = set()
    for path in [mix_path, converted_vocal_path, converted_vocal_dry_path, converted_vocal_raw_path, processed_conversion_input, trim_preview_path, trim_envelope_path, full_accompaniment_path, uploaded_path, *stems.values()]:
        if not path or not os.path.exists(path):
            continue
        abs_path = os.path.abspath(path)
        if abs_path in seen_files:
            continue
        seen_files.add(abs_path)
        files.append(path)
    if task_id:
        _update_task(task_id, progress_percent=100, progress_label="完成")
    emit("音色转换任务完成。")
    return mix_path, stems, analysis, "\n".join(status_lines), waveform_path, files, {
        "converted_vocal": converted_vocal_path,
        "converted_vocal_reverb": converted_vocal_path,
        "converted_vocal_dry": converted_vocal_dry_path,
        "converted_vocal_raw": converted_vocal_raw_path,
        "processed_conversion_input": processed_conversion_input,
        "preview_conversion_input": trim_preview_path,
        "full_accompaniment": full_accompaniment_path,
        "accompaniment": accompaniment_path,
        "mel_spectrogram": mel_path,
    }


def _latent_progress_array(latents) -> np.ndarray:
    """Convert ACE-Step latent tensor to a compact denoising heatmap."""
    latent = latents.detach().float().cpu().numpy()
    if latent.ndim == 4:
        latent = latent[0]
    if latent.ndim == 3:
        heatmap = np.mean(np.abs(latent), axis=0)
    elif latent.ndim == 2:
        heatmap = np.abs(latent)
    else:
        heatmap = np.atleast_2d(np.abs(latent))
    heatmap = np.log1p(heatmap)
    lo, hi = np.percentile(heatmap, [1, 99])
    if hi > lo:
        heatmap = np.clip((heatmap - lo) / (hi - lo), 0, 1)
    return heatmap.astype(np.float32)


def _use_scientific_plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.45,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
    })


def _latent_trace_summary(latents) -> dict[str, float]:
    latent = latents.detach().float().cpu().numpy()
    values = np.abs(latent).reshape(-1)
    if values.size == 0:
        return {"mean_abs": 0.0, "p95_abs": 0.0, "std_abs": 0.0}
    return {
        "mean_abs": float(np.mean(values)),
        "p95_abs": float(np.percentile(values, 95)),
        "std_abs": float(np.std(values)),
    }


def _save_denoising_trace_plot(records: list[dict], output_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        return ""
    _use_scientific_plot_style()
    trace_path = os.path.join(output_dir, "denoising_trace.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    steps = np.array([r["step"] for r in records], dtype=np.float32)
    mean_abs = np.array([r["mean_abs"] for r in records], dtype=np.float32)
    p95_abs = np.array([r["p95_abs"] for r in records], dtype=np.float32)
    timestep = np.array([r["timestep"] for r in records], dtype=np.float32)

    fig, ax1 = plt.subplots(figsize=(7.4, 3.4))
    ax1.plot(steps, mean_abs, color="#1f77b4", linewidth=1.6, label="mean |latent|")
    ax1.plot(steps, p95_abs, color="#d55e00", linewidth=1.2, linestyle="--", label="p95 |latent|")
    ax1.set_xlabel("Diffusion step")
    ax1.set_ylabel("Latent magnitude")
    ax2 = ax1.twinx()
    ax2.plot(steps, timestep, color="#6b7280", linewidth=1.0, linestyle=":", label="noise timestep")
    ax2.set_ylabel("Noise timestep")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right", fontsize=8)
    ax1.set_title("Denoising trajectory recorded during generation")
    fig.tight_layout()
    path = os.path.join(output_dir, "denoising_trace.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_process_overview(task: dict, output_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _use_scientific_plot_style()
    params = task.get("params") or {}
    kind = task.get("kind", "")
    if kind == "reference_style":
        stages = [
            ("1", "Reference\npreprocess"),
            ("2", "Feature\nanalysis"),
            ("3", "MusicDCAE\nz_ref"),
            ("4", "Latent\nmixing"),
            ("5", "Diffusion\ndenoise"),
            ("6", "Decode +\npostprocess"),
        ]
        subtitle = f"training-free reference audio | strength={params.get('ref_strength', '')} | steps={params.get('infer_steps', '')}"
    elif kind == "song":
        stages = [
            ("1", "Prompt +\nlyrics"),
            ("2", "Model\nload"),
            ("3", "LoRA/base\ncondition"),
            ("4", "Diffusion\ndenoise"),
            ("5", "Decode +\npostprocess"),
            ("6", "Analysis\noutputs"),
        ]
        subtitle = f"{task.get('model_key', '')} | seed={task.get('seed', '')} | duration={task.get('duration', '')}s"
    elif kind == "voice_conversion":
        stages = [
            ("1", "Upload\nsong"),
            ("2", "Demucs\noptional"),
            ("3", "VC/SVC\nexternal"),
            ("4", "Pitch\nshift"),
            ("5", "Remix\nstems"),
            ("6", "Analysis\noutputs"),
        ]
        subtitle = f"{task.get('model_key', '')} | pitch={params.get('pitch_shift', 0)} semitone(s)"
    else:
        stages = [
            ("1", "Batch\nprompts"),
            ("2", "Model\nload"),
            ("3", "Sequential\ngeneration"),
            ("4", "Save\nfiles"),
        ]
        subtitle = f"{task.get('model_key', '')} | batch task"

    fig, ax = plt.subplots(figsize=(8.6, 2.35))
    ax.axis("off")
    ax.text(0.02, 0.92, "Generation process overview", fontsize=11, weight="bold", ha="left")
    ax.text(0.02, 0.80, subtitle, fontsize=8.5, color="#374151", ha="left")
    x0, y, w, h, gap = 0.03, 0.32, 0.13, 0.28, 0.025
    colors = ["#e8f1fb", "#f7efe6", "#e9f6ef", "#f6eaf3", "#eef2f7", "#fff7df"]
    for i, (num, label) in enumerate(stages):
        x = x0 + i * (w + gap)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colors[i % len(colors)], edgecolor="#111827", linewidth=0.8))
        ax.text(x + 0.018, y + h - 0.055, num, fontsize=8.5, weight="bold", ha="left", va="center")
        ax.text(x + w / 2, y + h / 2 - 0.02, label, fontsize=8.2, ha="center", va="center")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap * 0.75, y + h / 2), xytext=(x + w + gap * 0.2, y + h / 2),
                        arrowprops=dict(arrowstyle="->", lw=0.9, color="#1f77b4"))
    ax.text(0.02, 0.08, f"task={task.get('id', '')} | status={task.get('status', '')} | updated={task.get('updated_at', '')}",
            fontsize=7.8, color="#4b5563", ha="left")
    path = os.path.join(output_dir, "generation_process_overview.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_latent_progress_frame(latents, output_dir: str, step: int, total: int, timestep: float) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    progress_dir = os.path.join(output_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)
    heatmap = _latent_progress_array(latents)
    path = os.path.join(progress_dir, f"diffusion_step_{step:03d}.png")
    _use_scientific_plot_style()
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    im = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"Latent denoising step {step}/{total} | noise t={timestep:.1f}")
    ax.set_xlabel("time frames")
    ax.set_ylabel("latent bands")
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.048)
    cb.set_label("log-normalized |z|", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save_progress_contact_sheet(frame_paths: list[str], output_dir: str) -> str:
    if not frame_paths:
        return ""
    latest = frame_paths[-1]
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return latest

    images = []
    selected = frame_paths if len(frame_paths) <= 12 else [frame_paths[i] for i in np.linspace(0, len(frame_paths) - 1, 12, dtype=int)]
    for path in selected:
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img.thumbnail((300, 170))
            images.append((path, img.copy()))
    if not images:
        return latest

    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    cell_w, cell_h = 330, 220
    header_h = 54
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font_title = ImageFont.truetype("times.ttf", 20)
        font_label = ImageFont.truetype("times.ttf", 14)
    except Exception:
        font_title = None
        font_label = None
    draw.text((14, 12), "Sampled latent denoising states", fill=(17, 24, 39), font=font_title)
    for idx, (path, img) in enumerate(images):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h + header_h
        sheet.paste(img, (x + 15, y + 10))
        draw.rectangle((x + 15, y + 10, x + 315, y + 180), outline=(209, 213, 219), width=1)
        draw.text((x + 15, y + 187), os.path.basename(path).replace(".png", ""), fill=(20, 20, 20), font=font_label)
    sheet_path = os.path.join(output_dir, "diffusion_progress_sheet.png")
    sheet.save(sheet_path)
    return sheet_path


def _make_diffusion_progress_callback(task_id: str, output_dir: str, total_steps: int, emit):
    frame_paths: list[str] = []
    trace_records: list[dict] = []
    snapshot_interval = 1
    last_logged_step = -1

    def callback(info: dict):
        nonlocal last_logged_step
        step = int(info.get("step", 0))
        total = int(info.get("total", total_steps) or total_steps or 1)
        percent = int(min(99, max(0, round(step * 90 / max(total, 1)))))
        label = f"扩散采样 {step}/{total} ({percent}%)"
        trace_records.append({
            "step": step,
            "total": total,
            "timestep": float(info.get("timestep", 0.0)),
            **_latent_trace_summary(info["latents"]),
        })

        outputs_update = None
        if step == 0 or step == total or step % snapshot_interval == 0:
            frame_path = _save_latent_progress_frame(
                info["latents"],
                output_dir=output_dir,
                step=step,
                total=total,
                timestep=float(info.get("timestep", 0.0)),
            )
            frame_paths.append(frame_path)
            sheet_path = _save_progress_contact_sheet(frame_paths, output_dir)
            trace_plot = _save_denoising_trace_plot(trace_records, output_dir)
            task = _get_task(task_id)
            outputs = (task or {}).get("outputs", {})
            outputs["progress_image"] = sheet_path
            outputs["progress_frames"] = list(frame_paths)
            if trace_plot:
                outputs["trace_plot"] = trace_plot
            outputs_update = outputs

        if task_id:
            updates = {"progress_percent": percent, "progress_label": label}
            if outputs_update is not None:
                updates["outputs"] = outputs_update
            _update_task(task_id, **updates)

        if step == 0 or step == total or step - last_logged_step >= max(1, total // 10):
            emit(label)
            last_logged_step = step

    return callback


# ============================================================
# 主生成函数
# ============================================================

def generate_song(
    user_prompt: str,
    style: str,
    bpm: int,
    mood: str,
    duration: int,
    seed: int,
    guidance: float,
    infer_steps: int,
    lora_weight: float,
    scheduler_type: str,
    cfg_type: str,
    omega_scale: float,
    guidance_interval: float,
    min_guidance_scale: float,
    use_erg: bool,
    post_process: bool,
    enable_stems: bool,
    lyrics: str,
    model_key: str = "ACE-Step v1-3.5B 基线模型",
    output_dir: str | None = None,
    task_id: str = "",
    log_callback=None,
):
    """主生成流程"""
    status_lines = []

    def emit(message: str, level: str = "INFO"):
        status_lines.append(message)
        if log_callback:
            log_callback(message, level)

    if output_dir is None:
        output_dir = os.path.join(WEB_OUTPUT_DIR, f"adhoc_{_task_timestamp()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        pipeline = get_pipeline(model_key, lora_weight=lora_weight)
    except Exception as e:
        emit(f"模型加载失败: {e}", "ERROR")
        return None, {}, f"模型加载失败: {e}", f"错误: {e}", None, None

    prompt = build_prompt(user_prompt, style, mood, bpm)
    actual_seed = _resolve_seed(seed)
    load_info = getattr(pipeline, "_edm_adapter_load_info", {})
    warnings = load_info.get("warnings", [])
    cache_key = load_info.get("cache_key", "base")
    lora_name_or_path = load_info.get("lora_path", "none")
    effective_lora_weight = load_info.get("lora_weight", 1.0)
    prompt = _apply_lora_style_prefix(prompt, load_info)

    emit(f"提示词: {prompt}")
    emit(f"任务目录: {output_dir}")
    emit(f"模型: {model_key}")
    if task_id:
        _update_task(task_id, progress_percent=3, progress_label="模型已加载，准备生成")
    if _last_model_load_summary:
        emit(f"加载状态: {_last_model_load_summary}")
    emit(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    emit(
        f"时长: {duration}s | 步数: {infer_steps} | guidance={guidance} | "
        f"scheduler={scheduler_type} | cfg={cfg_type} | omega={omega_scale} | "
        f"guidance_interval={guidance_interval} | min_guidance={min_guidance_scale} | "
        f"ERG={'on' if use_erg else 'off'} | seed={actual_seed}"
    )
    if load_info.get("lora_requested"):
        if load_info.get("lora_loaded"):
            emit(f"LoRA 已加载: {lora_name_or_path} | weight={float(effective_lora_weight):.2f}")
        else:
            emit("警告: 前端选择了 LoRA/CKPT，但 pipeline 没有激活 LoRA，本次不会使用 LoRA。", "WARN")
    else:
        emit("LoRA 状态: 未请求，使用基线/全量模型配置。")
    for warning in warnings:
        emit(f"提醒: {warning}", "WARN")

    # 歌词 / 人声处理
    if lyrics and lyrics.strip():
        emit(f"歌词: {lyrics.strip()[:100]}...")
    else:
        lyrics = "[instrumental]"
        emit("模式: 纯器乐（无人声）")

    with open(os.path.join(output_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt + "\n")
    with open(os.path.join(output_dir, "lyrics.txt"), "w", encoding="utf-8") as f:
        f.write(lyrics + "\n")

    # 生成
    try:
        emit("正在生成音频 (ACE-Step)...")
        if task_id:
            _update_task(task_id, progress_percent=5, progress_label="开始扩散采样")
        progress_callback = _make_diffusion_progress_callback(task_id, output_dir, int(infer_steps), emit) if task_id else None
        raw_dir = os.path.join(output_dir, "ace_step_raw")
        with _pipeline_generation_locks[cache_key]:
            audio, sr = generate_acestep(
                pipeline=pipeline,
                prompt=prompt,
                lyrics=lyrics,
                duration=float(duration),
                seed=actual_seed,
                infer_step=infer_steps,
                guidance_scale=guidance,
                scheduler_type=scheduler_type,
                cfg_type=cfg_type,
                omega_scale=omega_scale,
                guidance_interval=guidance_interval,
                min_guidance_scale=min_guidance_scale,
                use_erg=use_erg,
                save_dir=raw_dir,
                lora_name_or_path=lora_name_or_path,
                lora_weight=effective_lora_weight,
                progress_callback=progress_callback,
            )

        audio = normalize_audio(audio, peak_db=-1.0)

        if post_process:
            # 后处理：EQ、立体声增强、限幅
            emit("正在后处理 (EQ / 立体声增强 / 限幅)...")
            if task_id:
                _update_task(task_id, progress_percent=92, progress_label="后处理音频")
            audio = edm_post_process(audio, sr)
        else:
            emit("已跳过 EDM 后处理。")
    except Exception as e:
        emit(f"生成失败: {e}", "ERROR")
        return None, {}, f"生成失败: {e}", f"错误: {e}", None, None

    # 保存
    unique = task_id or f"{_task_timestamp()}_{uuid.uuid4().hex[:8]}"
    model_slug = _slugify(load_info.get("model_kind") or model_key, "model", 24)
    role_slug = _slugify((_get_task(task_id) or {}).get("compare_role", ""), "track", 20) if task_id else "track"
    mix_path = os.path.join(output_dir, f"{role_slug}_{model_slug}_mix_{unique}.wav")
    save_audio(mix_path, audio, sr=sr)
    emit(f"完整混音已保存: {mix_path}")
    if task_id:
        _update_task(task_id, progress_percent=95, progress_label="保存音频并分析")

    # 音频分析
    y_mono = audio[0] if audio.ndim == 2 else audio
    try:
        features = extract_all_features(y_mono, sr)
        analysis = (
            f"BPM (请求): {bpm}\n"
            f"BPM (检测): {features['bpm']:.1f}\n"
            f"BPM 误差: {abs(features['bpm'] - bpm):.1f}\n"
            f"RMS 能量: {features['rms_mean']:.4f}\n"
            f"低频比: {features['low_freq_ratio']:.4f}\n"
            f"频谱质心: {features['spectral_centroid_mean']:.1f} Hz\n"
            f"起音密度: {features['onset_density']:.2f} /秒\n"
            f"实际时长: {audio.shape[-1]/sr:.1f}s"
        )
    except Exception as e:
        analysis = f"音频分析失败: {e}\n实际时长: {audio.shape[-1]/sr:.1f}s"
        emit(f"音频分析失败: {e}", "WARN")
    with open(os.path.join(output_dir, "analysis.txt"), "w", encoding="utf-8") as f:
        f.write(analysis)

    stem_files = {}
    if enable_stems:
        emit("正在分离音轨 (Demucs)...")
        if task_id:
            _update_task(task_id, progress_percent=97, progress_label="分离音轨")
        stems = separate_stems(mix_path, output_dir)

        if stems:
            for name, path in stems.items():
                stem_files[name] = path
            emit(f"分离完成: {', '.join(stems.keys())}")
        else:
            emit("音轨分离失败或未安装 Demucs。", "WARN")
    else:
        emit("已跳过音轨分离。")

    # 波形图
    try:
        waveform_path = _plot_waveform(y_mono, sr, output_dir=output_dir, name=f"waveform_{unique}.png")
        emit(f"波形图已保存: {waveform_path}")
    except Exception as e:
        waveform_path = None
        emit(f"波形图生成失败: {e}", "WARN")

    try:
        mel_path = _save_ai_mel_for_audio(mix_path, output_dir, unique, prefix="mel_full_song")
        emit(f"完整 Mel 图谱（含末尾频谱负形 AI 水印）已保存: {mel_path}")
    except Exception as e:
        mel_path = None
        emit(f"Mel 图谱生成失败: {e}", "WARN")

    emit("生成完成！")
    if task_id:
        _update_task(task_id, progress_percent=100, progress_label="完成")
    status = "\n".join(status_lines)
    with open(os.path.join(output_dir, "status.txt"), "w", encoding="utf-8") as f:
        f.write(status)

    return mix_path, stem_files, analysis, status, waveform_path, mel_path


def _extract_reference_features_text(reference_audio_path: str) -> tuple[dict, str]:
    """Extract compact reference-audio descriptors for logging and evaluation."""
    y_ref, sr_ref = load_audio(reference_audio_path, sr=44100, mono=True)
    original_duration = len(y_ref) / sr_ref
    max_samples = int(sr_ref * 120)
    if len(y_ref) > max_samples:
        y_ref = y_ref[:max_samples]
    features = extract_all_features(y_ref, sr_ref)
    text = (
        f"参考音频 BPM: {features['bpm']:.1f}\n"
        f"参考 RMS 能量: {features['rms_mean']:.4f}\n"
        f"参考低频比例: {features['low_freq_ratio']:.4f}\n"
        f"参考频谱质心: {features['spectral_centroid_mean']:.1f} Hz\n"
        f"参考起音密度: {features['onset_density']:.2f} /秒\n"
        f"参考原始时长: {original_duration:.1f}s\n"
        f"参考分析时长: {len(y_ref) / sr_ref:.1f}s"
    )
    return features, text


def _style_feature_prompt(features: dict) -> str:
    """Convert objective audio features into non-melodic style constraints."""
    if not features:
        return ""

    bpm = float(features.get("bpm") or 0.0)
    low_ratio = float(features.get("low_freq_ratio") or 0.0)
    centroid = float(features.get("spectral_centroid_mean") or 0.0)
    onset_density = float(features.get("onset_density") or 0.0)
    rms = float(features.get("rms_mean") or 0.0)

    traits = []
    if bpm > 1:
        traits.append(f"around {int(round(bpm))} BPM")
    if low_ratio >= 0.32:
        traits.append("bass-heavy low end")
    elif low_ratio >= 0.18:
        traits.append("balanced dance low end")
    else:
        traits.append("light low end")
    if centroid >= 3000:
        traits.append("bright synth-forward tone")
    elif centroid >= 1800:
        traits.append("balanced mid-bright tone")
    else:
        traits.append("warm darker tone")
    if onset_density >= 4.0:
        traits.append("busy rhythmic percussion")
    elif onset_density >= 2.0:
        traits.append("steady dance groove")
    else:
        traits.append("sparse rhythmic motion")
    if rms >= 0.08:
        traits.append("high energy compressed mix")
    elif rms >= 0.03:
        traits.append("moderate energy mix")
    else:
        traits.append("soft dynamic mix")

    return (
        " Match only these non-melodic reference traits: "
        + ", ".join(traits)
        + "; keep the hook, melody, and chord progression newly composed."
    )


def _reference_mode_key(label: str | None) -> str:
    for key, value in REFERENCE_MODE_LABELS.items():
        if value == label:
            return key
    return "style_timbre"


def _reference_default_prompt(mode: str) -> str:
    if mode == "reconstruct":
        return REFERENCE_RECONSTRUCTION_PROMPT
    if mode == "melody":
        return REFERENCE_MELODY_PROMPT
    if mode == "style_timbre":
        return REFERENCE_STYLE_PROMPT
    return (
        "instrumental variation inspired by the reference audio's sound design, arrangement energy, "
        "rhythmic feel, and mix balance; create original melodic material, no vocals"
    )


def _reference_generation_settings(
    mode: str,
    prompt: str,
    ref_strength: float,
    guidance: float,
    infer_steps: int,
) -> tuple[str, float, float, int]:
    prompt = (prompt or "").strip()
    if not prompt:
        prompt = _reference_default_prompt(mode)
    elif mode in {"reconstruct", "melody"} and "melod" not in prompt.lower() and "旋律" not in prompt:
        prompt = f"{prompt}, preserve the reference melodic contour, hook motif, lead phrasing, chord progression energy, and rhythmic groove"
    elif mode == "style_timbre":
        prompt = (
            f"{prompt}, use the reference audio only for sound palette, timbre, drum design, bass tone, groove density, "
            "arrangement energy, stereo width, and mix balance; compose a new melody, new chord progression, and new hook; "
            "do not copy the reference melody; instrumental, no vocals"
        )

    if mode == "style_timbre":
        ref_strength = float(np.clip(float(ref_strength), 0.18, 0.45))
        guidance = float(np.clip(float(guidance), 8.5, 13.0))
        infer_steps = max(int(infer_steps), 140)
    elif mode == "reconstruct":
        ref_strength = max(float(ref_strength), 0.90)
        guidance = min(float(guidance), 5.0)
        infer_steps = max(int(infer_steps), 120)
    elif mode == "melody":
        ref_strength = max(float(ref_strength), 0.82)
        guidance = min(float(guidance), 7.0)
        infer_steps = max(int(infer_steps), 100)
    elif mode == "style":
        ref_strength = max(float(ref_strength), 0.55)
        guidance = min(float(guidance), 11.0)
        infer_steps = max(int(infer_steps), 60)
    else:
        ref_strength = float(ref_strength)
        guidance = float(guidance)
        infer_steps = int(infer_steps)

    return prompt, float(ref_strength), float(guidance), int(infer_steps)


def _reference_effective_steps(requested_steps: int, ref_strength: float) -> int:
    sigma_max = max(0.03, 1.0 - float(ref_strength))
    effective = int(np.ceil(int(requested_steps) / sigma_max))
    return int(np.clip(effective, int(requested_steps), int(requested_steps) * 12))


def generate_reference_style_song(
    reference_audio: str,
    content_prompt: str,
    reference_mode: str,
    duration: int,
    reference_start: float,
    auto_reference_start: bool,
    ref_strength: float,
    seed: int,
    guidance: float,
    infer_steps: int,
    use_style_proxy: bool = True,
    use_demucs_proxy: bool = False,
    candidate_count: int = 2,
    enable_stems: bool = False,
    lyrics: str = "",
    output_dir: str | None = None,
    task_id: str = "",
    log_callback=None,
):
    """Training-free reference-audio generation; always uses the base model and no LoRA."""
    status_lines = []

    def emit(message: str, level: str = "INFO"):
        status_lines.append(message)
        if log_callback:
            log_callback(message, level)

    if output_dir is None:
        output_dir = os.path.join(WEB_OUTPUT_DIR, f"reference_{_task_timestamp()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(output_dir, exist_ok=True)

    if not reference_audio or not os.path.exists(reference_audio):
        message = "请先上传一段参考音频。"
        emit(message, "ERROR")
        return None, {}, message, message, None, None

    base_key = _base_model_key()
    actual_seed = _resolve_seed(seed)
    mode = reference_mode or "style_timbre"
    requested_strength = float(ref_strength)
    requested_guidance = float(guidance)
    requested_steps = int(infer_steps)
    requested_candidate_count = int(candidate_count or 1)
    prompt, ref_strength, guidance, infer_steps = _reference_generation_settings(
        mode,
        content_prompt,
        requested_strength,
        requested_guidance,
        requested_steps,
    )
    planned_internal_steps = _reference_effective_steps(infer_steps, ref_strength)
    if mode != "style_timbre":
        use_style_proxy = False
        use_demucs_proxy = False
        candidate_count = 1
    else:
        candidate_count = int(np.clip(requested_candidate_count, 1, 4))
    lyrics = (lyrics or "").strip()
    lyrics_condition = lyrics or "[instrumental]"

    ref_copy = None
    try:
        suffix = Path(reference_audio).suffix or ".wav"
        ref_copy = os.path.join(output_dir, f"uploaded_reference{suffix}")
        shutil.copy2(reference_audio, ref_copy)
    except Exception as e:
        emit(f"参考音频复制失败，将直接读取临时路径: {e}", "WARN")
        ref_copy = reference_audio

    try:
        ref_features, ref_text = _extract_reference_features_text(ref_copy)
    except Exception as e:
        ref_features, ref_text = {}, f"参考音频特征提取失败: {e}"
        emit(ref_text, "WARN")
    if mode == "style_timbre":
        feature_prompt = _style_feature_prompt(ref_features)
        if feature_prompt:
            prompt = f"{prompt.rstrip(' .')}. {feature_prompt.strip()}"

    try:
        pipeline = get_pipeline(base_key, lora_weight=1.0)
        pipeline.load_lora("none", 1.0)
    except Exception as e:
        emit(f"基础模型加载失败: {e}", "ERROR")
        return None, {}, f"基础模型加载失败: {e}", f"错误: {e}", None, None

    load_info = getattr(pipeline, "_edm_adapter_load_info", {})
    cache_key = load_info.get("cache_key", "base")

    emit("模式: 无训练参考音频生成（base ACE-Step + reference latent conditioning）")
    emit(f"参考模式: {REFERENCE_MODE_LABELS.get(mode, mode)}")
    emit(f"任务目录: {output_dir}")
    emit(f"基础模型: {base_key}")
    emit(f"参考音频: {ref_copy}")
    emit(f"内容约束 prompt: {prompt}")
    emit(f"歌词条件: {lyrics_condition}")
    emit(
        f"ref_strength={float(ref_strength):.2f} (requested {requested_strength:.2f}) | "
        f"seed={actual_seed} | duration={int(duration)}s | steps={int(infer_steps)} "
        f"(requested {requested_steps}, internal≈{planned_internal_steps}) | guidance={float(guidance):.1f} (requested {requested_guidance:.1f})"
    )
    emit(
        "方法: 上传音频先做采样率/声道/时长标准化；风格音色模式会额外构造去旋律参考代理音频，再由 MusicDCAE 编码为参考潜变量；"
        "扩散初始噪声与参考潜变量按强度混合后重新去噪。"
    )
    emit(
        f"清晰化设置: style_proxy={'on' if use_style_proxy else 'off'} | "
        f"demucs_proxy={'on' if use_demucs_proxy else 'off'} | candidates={candidate_count}"
    )
    if _last_model_load_summary:
        emit(f"加载状态: {_last_model_load_summary}")
    if task_id:
        _update_task(task_id, progress_percent=3, progress_label="基础模型已加载，正在分析参考音频")

    with open(os.path.join(output_dir, "reference_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt + "\n")
    with open(os.path.join(output_dir, "reference_features.txt"), "w", encoding="utf-8") as f:
        f.write(ref_text + "\n")

    candidate_records = []
    best_candidate = None
    try:
        emit("正在执行参考音频潜变量条件生成...")
        if task_id:
            _update_task(task_id, progress_percent=5, progress_label="开始参考潜变量扩散采样")
        candidates_dir = os.path.join(output_dir, "candidates")
        os.makedirs(candidates_dir, exist_ok=True)
        for cand_idx in range(candidate_count):
            candidate_seed = int(actual_seed + cand_idx * 1009)
            emit(f"候选 {cand_idx + 1}/{candidate_count}: seed={candidate_seed}")
            if task_id:
                _update_task(
                    task_id,
                    progress_percent=max(5, min(88, int(5 + cand_idx * 80 / max(candidate_count, 1)))),
                    progress_label=f"生成参考候选 {cand_idx + 1}/{candidate_count}",
                )
            progress_callback = (
                _make_diffusion_progress_callback(task_id, output_dir, int(planned_internal_steps), emit)
                if task_id
                else None
            )
            raw_dir = os.path.join(output_dir, "ace_step_reference_raw", f"candidate_{cand_idx + 1:02d}")
            with _pipeline_generation_locks[cache_key]:
                cand_audio, sr, cand_metadata = generate_acestep_reference_style(
                    pipeline=pipeline,
                    prompt=prompt,
                    reference_audio_path=ref_copy,
                    lyrics=lyrics_condition,
                    duration=float(duration),
                    seed=candidate_seed,
                    infer_step=int(infer_steps),
                    guidance_scale=float(guidance),
                    ref_audio_strength=float(ref_strength),
                    reference_start=float(reference_start or 0.0),
                    auto_reference_start=bool(auto_reference_start),
                    use_style_proxy=bool(use_style_proxy),
                    use_demucs_proxy=bool(use_demucs_proxy),
                    save_dir=raw_dir,
                    progress_callback=progress_callback,
                )
            cand_audio = normalize_audio(cand_audio, peak_db=-1.0)
            cand_score = score_reference_candidate(cand_audio, sr, ref_features)
            cand_path = os.path.join(candidates_dir, f"candidate_{cand_idx + 1:02d}_seed{candidate_seed}.wav")
            save_audio(cand_path, cand_audio, sr=sr)
            cand_record = {
                "index": cand_idx + 1,
                "seed": candidate_seed,
                "path": cand_path,
                "score": cand_score,
                "metadata": cand_metadata,
            }
            candidate_records.append(cand_record)
            emit(
                f"候选 {cand_idx + 1} 评分={cand_score['score']:.3f} | "
                f"low={cand_score['low_freq_ratio']:.3f} | centroid={cand_score['spectral_centroid']:.0f}Hz | "
                f"onset={cand_score['onset_density']:.2f}/s"
            )
            if best_candidate is None or cand_score["score"] > best_candidate["score"]["score"]:
                best_candidate = {**cand_record, "audio": cand_audio, "sr": sr}
        if not best_candidate:
            raise RuntimeError("没有成功生成任何候选音频。")
        audio = best_candidate["audio"]
        sr = best_candidate["sr"]
        metadata = best_candidate["metadata"]
        metadata["selected_candidate"] = {
            "index": best_candidate["index"],
            "seed": best_candidate["seed"],
            "path": best_candidate["path"],
            "score": best_candidate["score"],
        }
        metadata["candidate_scores"] = [
            {
                "index": item["index"],
                "seed": item["seed"],
                "path": item["path"],
                "score": item["score"],
            }
            for item in candidate_records
        ]
        emit(f"已选择候选 {best_candidate['index']}/{candidate_count}: seed={best_candidate['seed']}，score={best_candidate['score']['score']:.3f}")
    except Exception as e:
        emit(f"参考音频生成失败: {e}", "ERROR")
        return None, {}, f"参考音频生成失败: {e}", f"错误: {e}", None, None

    unique = task_id or f"{_task_timestamp()}_{uuid.uuid4().hex[:8]}"
    mix_path = os.path.join(output_dir, f"reference_style_mix_{unique}.wav")
    save_audio(mix_path, audio, sr=sr)
    emit(f"生成音频已保存: {mix_path}")
    if task_id:
        _update_task(task_id, progress_percent=95, progress_label="保存音频并分析")

    try:
        y_mono = audio[0] if audio.ndim == 2 else audio
        gen_features = extract_all_features(y_mono, sr)
        analysis = (
            "模式: 无训练参考音频生成\n"
            "训练状态: 未训练、未加载 LoRA、未更新权重\n"
            f"参考模式: {REFERENCE_MODE_LABELS.get(mode, mode)}\n"
            f"参考潜变量强度: {float(ref_strength):.2f}\n"
            f"参考代理音频: {'启用' if use_style_proxy else '关闭'}\n"
            f"Demucs 代理分离: {'启用' if use_demucs_proxy else '关闭'}\n"
            f"候选数量: {candidate_count}，选择候选: {metadata.get('selected_candidate', {}).get('index', 1)}\n"
            f"歌词条件: {lyrics_condition}\n"
            f"参考裁剪起点: {float(reference_start or 0.0):.1f}s\n"
            f"自动选择片段: {'是' if auto_reference_start else '否'}\n"
            f"请求步数: {requested_steps}\n"
            f"内部实际步数: {metadata.get('effective_infer_step', planned_internal_steps)}\n"
            f"seed: {actual_seed}\n\n"
            f"{ref_text}\n\n"
            f"生成 BPM: {gen_features['bpm']:.1f}\n"
            f"生成 RMS 能量: {gen_features['rms_mean']:.4f}\n"
            f"生成低频比例: {gen_features['low_freq_ratio']:.4f}\n"
            f"生成频谱质心: {gen_features['spectral_centroid_mean']:.1f} Hz\n"
            f"生成起音密度: {gen_features['onset_density']:.2f} /秒\n"
            f"生成时长: {audio.shape[-1] / sr:.1f}s\n\n"
            "候选评分:\n"
            + "\n".join(
                f"- candidate {item['index']} seed={item['seed']} score={item['score']['score']:.3f} "
                f"low={item['score']['low_freq_ratio']:.3f} centroid={item['score']['spectral_centroid']:.0f}Hz onset={item['score']['onset_density']:.2f}/s"
                for item in candidate_records
            )
            + "\n\n"
            "智能音频处理链路: 上传音频标准化 -> 参考特征分析 -> MusicDCAE 潜变量编码 -> 参考潜变量加噪 -> 文本条件扩散去噪 -> 波形解码 -> 客观音频特征评估。"
        )
    except Exception as e:
        analysis = f"音频分析失败: {e}\n实际时长: {audio.shape[-1] / sr:.1f}s\n{ref_text}"
        emit(f"音频分析失败: {e}", "WARN")

    metadata["final_path"] = mix_path
    metadata["reference_mode"] = mode
    metadata["requested_ref_strength"] = requested_strength
    metadata["requested_guidance"] = requested_guidance
    metadata["requested_infer_steps"] = requested_steps
    metadata["requested_candidate_count"] = requested_candidate_count
    metadata["effective_candidate_count"] = candidate_count
    metadata["use_style_proxy"] = bool(use_style_proxy)
    metadata["use_demucs_proxy"] = bool(use_demucs_proxy)
    metadata["auto_reference_start_requested"] = bool(auto_reference_start)
    metadata["reference_features"] = ref_features
    with open(os.path.join(output_dir, "reference_generation_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "analysis.txt"), "w", encoding="utf-8") as f:
        f.write(analysis)

    stem_files = {}
    if enable_stems:
        emit("正在分离音轨 (Demucs)...")
        if task_id:
            _update_task(task_id, progress_percent=97, progress_label="分离音轨")
        stems = separate_stems(mix_path, output_dir)
        if stems:
            stem_files.update(stems)
            emit(f"分离完成: {', '.join(stems.keys())}")
        else:
            emit("音轨分离失败或未安装 Demucs。", "WARN")
    else:
        emit("已跳过音轨分离。")

    try:
        waveform_path = _plot_waveform(audio[0] if audio.ndim == 2 else audio, sr, output_dir=output_dir, name=f"waveform_{unique}.png")
        emit(f"波形图已保存: {waveform_path}")
    except Exception as e:
        waveform_path = None
        emit(f"波形图生成失败: {e}", "WARN")

    try:
        mel_path = _save_ai_mel_for_audio(mix_path, output_dir, unique, prefix="mel_reference_style")
        emit(f"完整 Mel 图谱（含末尾频谱负形 AI 水印）已保存: {mel_path}")
    except Exception as e:
        mel_path = None
        emit(f"Mel 图谱生成失败: {e}", "WARN")

    emit("无训练参考音频生成完成。")
    if task_id:
        _update_task(task_id, progress_percent=100, progress_label="完成")
    status = "\n".join(status_lines)
    with open(os.path.join(output_dir, "status.txt"), "w", encoding="utf-8") as f:
        f.write(status)
    return mix_path, stem_files, analysis, status, waveform_path, mel_path


def generate_batch_tracks(
    batch_prompts: str,
    duration: int,
    seed: int,
    guidance: float,
    infer_steps: int,
    lora_weight: float,
    model_key: str,
    output_dir: str | None = None,
    task_id: str = "",
    log_callback=None,
):
    """批量生成微调模型演示音频，不做音轨分离，避免一次任务过慢。"""
    status = []

    def emit(message: str, level: str = "INFO"):
        status.append(message)
        if log_callback:
            log_callback(message, level)

    lines = [line.strip() for line in (batch_prompts or "").splitlines() if line.strip()]
    if not lines:
        lines = [example[0] for example in EXAMPLE_PROMPTS[:4]]

    truncated = len(lines) > 5
    lines = lines[:5]

    try:
        pipeline = get_pipeline(model_key, lora_weight=lora_weight)
    except Exception as e:
        emit(f"模型加载失败: {e}", "ERROR")
        return [], f"模型加载失败: {e}", []

    if output_dir is None:
        output_dir = os.path.join(WEB_OUTPUT_DIR, f"batch_{_task_timestamp()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    mel_paths = []
    model_kind = scan_available_models().get(model_key, {}).get("kind")
    load_info = getattr(pipeline, "_edm_adapter_load_info", {})
    cache_key = load_info.get("cache_key", "base")
    lora_name_or_path = load_info.get("lora_path", "none")
    effective_lora_weight = load_info.get("lora_weight", 1.0)

    emit(f"模型: {model_key}")
    emit(f"加载状态: {_last_model_load_summary}")
    if model_kind in {"lora", "ckpt"}:
        if load_info.get("lora_loaded"):
            emit(f"LoRA 已加载: {lora_name_or_path} | weight={float(effective_lora_weight):.2f}")
        else:
            emit("警告: 前端选择了 LoRA/CKPT，但 pipeline 没有激活 LoRA，本批次不会使用 LoRA。", "WARN")
    else:
        emit("LoRA 状态: 未请求，使用基线/全量模型配置。")
    for warning in load_info.get("warnings", []):
        emit(f"提醒: {warning}", "WARN")
    emit(f"批量数量: {len(lines)}" + ("（已限制为前 5 条）" if truncated else ""))
    emit(f"输出目录: {output_dir}")
    emit("")

    for index, raw_prompt in enumerate(lines, start=1):
        prompt = _apply_lora_style_prefix(raw_prompt, load_info)
        track_seed = int(seed) + index - 1
        emit(f"[{index}/{len(lines)}] seed={track_seed} | {prompt}")
        if task_id:
            percent = int((index - 1) * 100 / max(len(lines), 1))
            _update_task(task_id, progress_percent=percent, progress_label=f"批量生成 {index}/{len(lines)}")
        try:
            raw_dir = os.path.join(output_dir, "ace_step_raw", f"track_{index:02d}")
            with _pipeline_generation_locks[cache_key]:
                audio, sr = generate_acestep(
                    pipeline=pipeline,
                    prompt=prompt,
                    lyrics="[instrumental]",
                    duration=float(duration),
                    seed=track_seed,
                    infer_step=int(infer_steps),
                    guidance_scale=float(guidance),
                    save_dir=raw_dir,
                    lora_name_or_path=lora_name_or_path,
                    lora_weight=effective_lora_weight,
                )
            audio = normalize_audio(audio, peak_db=-1.0)
            audio = edm_post_process(audio, sr)
            unique = task_id or f"{_task_timestamp()}_{uuid.uuid4().hex[:8]}"
            output_path = os.path.join(output_dir, f"track_{index:02d}_seed_{track_seed}_{unique}.wav")
            save_audio(output_path, audio, sr=sr)
            output_paths.append(output_path)
            try:
                mel_path = _save_ai_mel_for_audio(output_path, output_dir, f"track_{index:02d}_seed_{track_seed}_{unique}", prefix="mel_batch")
                if mel_path:
                    mel_paths.append(mel_path)
                    emit(f"  Mel 图谱: {mel_path}")
            except Exception as exc:
                emit(f"  Mel 图谱生成失败: {exc}", "WARN")
            emit(f"  完成: {output_path}")
            if task_id:
                percent = int(index * 100 / max(len(lines), 1))
                _update_task(task_id, progress_percent=percent, progress_label=f"批量生成 {index}/{len(lines)}")
        except Exception as e:
            emit(f"  失败: {e}", "ERROR")

    with open(os.path.join(output_dir, "batch_status.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(status))
    return output_paths, "\n".join(status), mel_paths


def _style_label_to_key(style_label: str) -> str:
    for key, value in STYLE_LABELS.items():
        if value == style_label:
            return key
    return "techno"


def _mood_label_to_key(mood_label: str) -> str:
    for key, value in MOOD_LABELS.items():
        if value == mood_label:
            return key
    return "榛戞殫"


def _create_task(kind: str, params: dict) -> str:
    task_id = f"{_task_timestamp()}_{kind}_{uuid.uuid4().hex[:8]}"
    output_dir = _task_output_dir(task_id)
    model_key = params.get("model_key", "")
    model_info = scan_available_models().get(model_key, {})
    os.makedirs(output_dir, exist_ok=True)
    task = {
        "id": task_id,
        "kind": kind,
        "status": "queued",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "started_at": "",
        "finished_at": "",
        "output_dir": output_dir,
        "params": params,
        "model_key": model_key,
        "compare_role": params.get("compare_role", ""),
        "model_kind": model_info.get("kind", ""),
        "lora_requested": model_info.get("kind") in {"lora", "ckpt"},
        "lora_loaded": False,
        "lora_path": "",
        "warnings": [],
        "seed": params.get("seed", ""),
        "duration": params.get("duration", ""),
        "progress_percent": 0,
        "progress_label": "排队中",
        "outputs": {
            "mix": None,
            "stems": {},
            "process_overview": None,
            "trace_plot": None,
            "progress_image": None,
            "progress_frames": [],
            "waveform": None,
            "files": [],
        },
        "analysis": "",
        "status_text": "",
        "log_tail": "",
        "error": "",
    }
    try:
        task["outputs"]["process_overview"] = _save_process_overview(task, output_dir)
    except Exception:
        pass
    with _task_lock:
        tasks = _load_task_index_unlocked()
        _save_task_unlocked(task, tasks)
    _append_task_log(task_id, f"任务已进入队列: {kind} | model={model_key}")
    return task_id


def _submit_task(task_id: str):
    future = _task_executor.submit(_run_task, task_id)
    _task_futures[task_id] = future
    return future


def _pipeline_load_info_for(model_key: str, lora_weight: float) -> dict:
    try:
        pipeline = get_pipeline(model_key, lora_weight=lora_weight)
        return getattr(pipeline, "_edm_adapter_load_info", {})
    except Exception:
        return {}


def _run_task(task_id: str):
    task = _get_task(task_id)
    if not task:
        return
    try:
        if task.get("kind") == "song":
            _run_song_task(task_id, task)
        elif task.get("kind") == "reference_style":
            _run_reference_style_task(task_id, task)
        elif task.get("kind") == "voice_conversion":
            _run_voice_conversion_task(task_id, task)
        elif task.get("kind") == "batch":
            _run_batch_task(task_id, task)
        else:
            raise ValueError(f"未知任务类型: {task.get('kind')}")
    except Exception as e:
        _append_task_log(task_id, f"任务异常: {e}", "ERROR")
        _update_task(task_id, status="failed", finished_at=_now_iso(), error=str(e))


def _run_song_task(task_id: str, task: dict):
    params = task.get("params", {})
    _update_task(task_id, status="running", started_at=_now_iso(), error="", progress_percent=1, progress_label="加载模型")
    _append_task_log(task_id, f"开始生成；并行工作线程上限={MAX_PARALLEL_JOBS}")
    mix_path, stem_files, analysis, status_text, wave_path, mel_path = generate_song(
        params.get("user_prompt", ""),
        params.get("style", "techno"),
        int(params.get("bpm", 128)),
        params.get("mood", "榛戞殫"),
        int(params.get("duration", 60)),
        int(params.get("seed", -1)),
        float(params.get("guidance", 15.0)),
        int(params.get("infer_steps", 60)),
        float(params.get("lora_weight", 1.0)),
        params.get("scheduler_type", "euler"),
        params.get("cfg_type", "apg"),
        float(params.get("omega_scale", 10.0)),
        float(params.get("guidance_interval", 0.5)),
        float(params.get("min_guidance_scale", 3.0)),
        bool(params.get("use_erg", True)),
        bool(params.get("post_process", True)),
        bool(params.get("enable_stems", False)),
        params.get("lyrics", ""),
        params.get("model_key", _base_model_key()),
        output_dir=task.get("output_dir"),
        task_id=task_id,
        log_callback=lambda message, level="INFO": _append_task_log(task_id, message, level),
    )

    if not mix_path:
        _update_task(task_id, status="failed", finished_at=_now_iso(), error=status_text, status_text=status_text, progress_label="失败")
        return

    load_info = _pipeline_load_info_for(params.get("model_key", ""), float(params.get("lora_weight", 1.0)))
    warnings = list(load_info.get("warnings", []))
    if load_info.get("lora_requested") and not load_info.get("lora_loaded"):
        warnings.append("前端选择了 LoRA/CKPT，但本任务没有激活 LoRA。")
    audio_files = [path for path in [mix_path, *stem_files.values()] if path and os.path.exists(path)]
    current_task = _get_task(task_id) or {}
    outputs = current_task.get("outputs", {})
    overview_path = _save_process_overview({**task, "status": "completed"}, task.get("output_dir"))
    outputs.update({"mix": mix_path, "stems": stem_files, "waveform": wave_path, "mel_spectrogram": mel_path, "files": audio_files})
    outputs["process_overview"] = overview_path
    _update_task(
        task_id,
        status="completed",
        finished_at=_now_iso(),
        outputs=outputs,
        analysis=analysis,
        status_text=status_text,
        lora_requested=bool(load_info.get("lora_requested")),
        lora_loaded=bool(load_info.get("lora_loaded")),
        lora_path=load_info.get("lora_path", ""),
        warnings=warnings,
    )
    _append_task_log(task_id, f"任务完成；音频文件数={len(audio_files)}")


def _run_reference_style_task(task_id: str, task: dict):
    params = task.get("params", {})
    _update_task(task_id, status="running", started_at=_now_iso(), error="", progress_percent=1, progress_label="加载基础模型")
    _append_task_log(task_id, f"开始无训练参考音频生成；并行工作线程上限={MAX_PARALLEL_JOBS}")
    mix_path, stem_files, analysis, status_text, wave_path, mel_path = generate_reference_style_song(
        params.get("reference_audio", ""),
        params.get("content_prompt", ""),
        params.get("reference_mode", "reconstruct"),
        int(params.get("duration", 30)),
        float(params.get("reference_start", 0.0)),
        bool(params.get("auto_reference_start", True)),
        float(params.get("ref_strength", 0.90)),
        int(params.get("seed", -1)),
        float(params.get("guidance", 4.5)),
        int(params.get("infer_steps", 120)),
        bool(params.get("use_style_proxy", True)),
        bool(params.get("use_demucs_proxy", False)),
        int(params.get("candidate_count", 2)),
        bool(params.get("enable_stems", False)),
        params.get("lyrics", ""),
        output_dir=task.get("output_dir"),
        task_id=task_id,
        log_callback=lambda message, level="INFO": _append_task_log(task_id, message, level),
    )

    if not mix_path:
        _update_task(task_id, status="failed", finished_at=_now_iso(), error=status_text, status_text=status_text, progress_label="失败")
        return

    candidate_files = []
    candidates_dir = os.path.join(task.get("output_dir"), "candidates")
    if os.path.isdir(candidates_dir):
        candidate_files = [
            os.path.join(candidates_dir, name)
            for name in sorted(os.listdir(candidates_dir))
            if name.lower().endswith(".wav")
        ]
    audio_files = [path for path in [mix_path, *candidate_files, *stem_files.values()] if path and os.path.exists(path)]
    current_task = _get_task(task_id) or {}
    outputs = current_task.get("outputs", {})
    overview_path = _save_process_overview({**task, "status": "completed"}, task.get("output_dir"))
    outputs.update({"mix": mix_path, "stems": stem_files, "waveform": wave_path, "mel_spectrogram": mel_path, "files": audio_files})
    outputs["process_overview"] = overview_path
    _update_task(
        task_id,
        status="completed",
        finished_at=_now_iso(),
        outputs=outputs,
        analysis=analysis,
        status_text=status_text,
        lora_requested=False,
        lora_loaded=False,
        lora_path="",
        warnings=[],
    )
    _append_task_log(task_id, f"无训练参考音频任务完成；音频文件数={len(audio_files)}")


def _run_voice_conversion_task(task_id: str, task: dict):
    params = task.get("params", {})
    _update_task(task_id, status="running", started_at=_now_iso(), error="", progress_percent=1, progress_label="准备音色转换")
    _append_task_log(task_id, f"开始授权音色转换；并行工作线程上限={MAX_PARALLEL_JOBS}")
    mix_path, stem_files, analysis, status_text, wave_path, audio_files, extra_outputs = generate_voice_conversion(
        params.get("source_audio", ""),
        params.get("model_dir", VOICE_MODEL_DIR),
        int(params.get("pitch_shift", 0)),
        bool(params.get("use_demucs", True)),
        diffusion_steps=int(params.get("diffusion_steps", 30) or 30),
        inference_cfg_rate=float(params.get("inference_cfg_rate", 0.75) or 0.75),
        dehiss_strength=float(params.get("dehiss_strength", 0.40) or 0.0),
        emotion_strength=float(params.get("emotion_strength", 0.55) or 0.0),
        vocal_gain_db=float(params.get("vocal_gain_db", 1.5) or 0.0),
        max_vocal_seconds=int(params.get("max_vocal_seconds", 30) or 0),
        output_dir=task.get("output_dir"),
        task_id=task_id,
        log_callback=lambda message, level="INFO": _append_task_log(task_id, message, level),
    )

    current_task = _get_task(task_id) or {}
    outputs = current_task.get("outputs", {})
    if not mix_path:
        outputs.update({"stems": stem_files, "waveform": wave_path, "files": audio_files, **extra_outputs})
        _update_task(
            task_id,
            status="failed",
            finished_at=_now_iso(),
            outputs=outputs,
            analysis=analysis,
            status_text=status_text,
            error=analysis or status_text,
            lora_requested=False,
            lora_loaded=False,
            warnings=["未完成音色转换；查看任务日志和状态诊断。"],
            progress_label="失败",
        )
        return

    overview_path = _save_process_overview({**task, "status": "completed"}, task.get("output_dir"))
    outputs.update({"mix": mix_path, "stems": stem_files, "waveform": wave_path, "files": audio_files, **extra_outputs})
    outputs["process_overview"] = overview_path
    _update_task(
        task_id,
        status="completed",
        finished_at=_now_iso(),
        outputs=outputs,
        analysis=analysis,
        status_text=status_text,
        lora_requested=False,
        lora_loaded=False,
        lora_path="",
        warnings=[],
    )
    _append_task_log(task_id, f"授权音色转换任务完成；音频文件数={len(audio_files)}")


def _run_batch_task(task_id: str, task: dict):
    params = task.get("params", {})
    _update_task(task_id, status="running", started_at=_now_iso(), error="", progress_percent=1, progress_label="加载模型")
    _append_task_log(task_id, f"开始批量生成；并行工作线程上限={MAX_PARALLEL_JOBS}")
    output_paths, status_text, mel_paths = generate_batch_tracks(
        params.get("batch_prompts", ""),
        int(params.get("duration", 30)),
        int(params.get("seed", 42)),
        float(params.get("guidance", 15.0)),
        int(params.get("infer_steps", 50)),
        float(params.get("lora_weight", 1.0)),
        params.get("model_key", _base_model_key()),
        output_dir=task.get("output_dir"),
        task_id=task_id,
        log_callback=lambda message, level="INFO": _append_task_log(task_id, message, level),
    )

    load_info = _pipeline_load_info_for(params.get("model_key", ""), float(params.get("lora_weight", 1.0)))
    warnings = list(load_info.get("warnings", []))
    if load_info.get("lora_requested") and not load_info.get("lora_loaded"):
        warnings.append("前端选择了 LoRA/CKPT，但本任务没有激活 LoRA。")
    output_paths = [path for path in output_paths if path and os.path.exists(path)]
    status = "completed" if output_paths else "failed"
    _update_task(
        task_id,
        status=status,
        finished_at=_now_iso(),
        outputs={
            "mix": output_paths[0] if output_paths else None,
            "stems": {},
            "waveform": None,
            "mel_spectrogram": mel_paths[0] if mel_paths else None,
            "mel_spectrograms": mel_paths,
            "files": output_paths,
        },
        status_text=status_text,
        lora_requested=bool(load_info.get("lora_requested")),
        lora_loaded=bool(load_info.get("lora_loaded")),
        lora_path=load_info.get("lora_path", ""),
        warnings=warnings,
        error="" if output_paths else "批量任务没有成功生成任何音频。",
        progress_percent=100 if output_paths else 0,
        progress_label="完成" if output_paths else "失败",
    )
    _append_task_log(task_id, f"批量任务结束；状态={status}；音频文件数={len(output_paths)}")


def submit_generation_tasks(
    user_prompt,
    style_label,
    bpm,
    mood_label,
    duration,
    lyrics,
    seed,
    guidance,
    infer_steps,
    lora_weight,
    scheduler_type,
    cfg_type,
    omega_scale,
    guidance_interval,
    min_guidance_scale,
    use_erg,
    post_process,
    enable_stems,
    model_key,
    compare_baseline,
):
    actual_seed = _resolve_seed(seed)
    params = {
        "user_prompt": user_prompt or "",
        "style": _style_label_to_key(style_label),
        "bpm": int(bpm),
        "mood": _mood_label_to_key(mood_label),
        "duration": int(duration),
        "lyrics": lyrics or "",
        "seed": actual_seed,
        "guidance": float(guidance),
        "infer_steps": int(infer_steps),
        "lora_weight": float(lora_weight),
        "scheduler_type": scheduler_type or "euler",
        "cfg_type": cfg_type or "apg",
        "omega_scale": float(omega_scale),
        "guidance_interval": float(guidance_interval),
        "min_guidance_scale": float(min_guidance_scale),
        "use_erg": bool(use_erg),
        "post_process": bool(post_process),
        "enable_stems": bool(enable_stems),
        "model_key": model_key,
        "compare_role": "selected_model",
    }

    task_ids = []
    primary_id = _create_task("song", params)
    task_ids.append(primary_id)
    _submit_task(primary_id)

    base_key = _base_model_key()
    if compare_baseline and model_key != base_key:
        base_params = dict(params)
        base_params["model_key"] = base_key
        base_params["lora_weight"] = 1.0
        base_params["compare_role"] = "baseline"
        base_id = _create_task("song", base_params)
        task_ids.append(base_id)
        _submit_task(base_id)

    status_text = (
        "已提交任务: " + ", ".join(task_ids) + "\n"
        f"输出会持久化到: {WEB_OUTPUT_DIR}\n"
        "可以在当前页等待轮询，也可以到“任务队列”查看全部任务和音频。"
    )
    return (
        ", ".join(task_ids),
        None,
        None,
        status_text,
        None,
        None,
        None,
        None,
        "",
        status_text,
        [],
        None,
        None,
        [],
        None,
        None,
        _task_table_rows(),
    )


def submit_reference_style_task(
    reference_audio,
    content_prompt,
    reference_mode_label,
    duration,
    reference_start,
    auto_reference_start,
    ref_strength,
    seed,
    guidance,
    infer_steps,
    use_style_proxy,
    use_demucs_proxy,
    candidate_count,
    lyrics,
    enable_stems,
):
    if not reference_audio:
        message = "请先上传参考音频。"
        return "", None, message, "", [], None, [], _task_table_rows(), None, None, None, None

    actual_seed = _resolve_seed(seed)
    mode = _reference_mode_key(reference_mode_label)
    params = {
        "reference_audio": reference_audio,
        "content_prompt": content_prompt or "",
        "reference_mode": mode,
        "duration": int(duration),
        "reference_start": float(reference_start or 0.0),
        "auto_reference_start": bool(auto_reference_start),
        "ref_strength": float(ref_strength),
        "seed": actual_seed,
        "guidance": float(guidance),
        "infer_steps": int(infer_steps),
        "use_style_proxy": bool(use_style_proxy),
        "use_demucs_proxy": bool(use_demucs_proxy),
        "candidate_count": int(candidate_count or 1),
        "lyrics": lyrics or "",
        "enable_stems": bool(enable_stems),
        "model_key": _base_model_key(),
        "compare_role": "reference_audio",
    }
    task_id = _create_task("reference_style", params)
    _submit_task(task_id)
    status_text = (
        f"已提交无训练参考音频任务: {task_id}\n"
        f"输出会持久化到: {WEB_OUTPUT_DIR}\n"
        "该流程使用基础模型 + 上传音频潜变量条件，不加载 LoRA，也不会训练或覆盖微调模型。"
    )
    return task_id, None, status_text, "", [], None, [], _task_table_rows(), None, None, None, None


def submit_voice_conversion_task(
    source_audio,
    model_dir,
    pitch_shift,
    use_demucs,
    diffusion_steps,
    inference_cfg_rate,
    dehiss_strength,
    emotion_strength,
    vocal_gain_db,
    max_vocal_seconds,
):
    if not source_audio:
        message = "请先上传一首歌。"
        return "", None, None, None, None, None, None, None, message, "", None, None, [], _task_table_rows()

    target_dir = os.path.abspath(model_dir or VOICE_MODEL_DIR)
    if not os.path.isdir(target_dir):
        message = f"目标模型目录不存在: {target_dir}"
        return "", None, None, None, None, None, None, None, message, "", None, None, [], _task_table_rows()

    params = {
        "source_audio": source_audio,
        "model_dir": target_dir,
        "pitch_shift": int(pitch_shift or 0),
        "use_demucs": bool(use_demucs),
        "diffusion_steps": int(diffusion_steps or 30),
        "inference_cfg_rate": float(inference_cfg_rate or 0.75),
        "dehiss_strength": float(dehiss_strength or 0.0),
        "emotion_strength": float(emotion_strength or 0.0),
        "vocal_gain_db": float(vocal_gain_db or 0.0),
        "max_vocal_seconds": int(max_vocal_seconds or 0),
        "model_key": "授权目标音色模型",
        "compare_role": "authorized_voice_conversion",
        "duration": "",
        "seed": "",
    }
    task_id = _create_task("voice_conversion", params)
    _submit_task(task_id)
    status_text = (
        f"已提交授权音色转换任务 {task_id}\n"
        f"输出会持久化到 {WEB_OUTPUT_DIR}\n"
        f"外部转换器: {'环境变量已配置' if os.environ.get(VOICE_CONVERTER_CMD_ENV, '').strip() else 'Seed-VC 默认桥接命令'}\n"
        f"质量参数: steps={int(diffusion_steps or 30)}, cfg={float(inference_cfg_rate or 0.75):.2f}, 去沙={float(dehiss_strength or 0.0):.2f}, 情感={float(emotion_strength or 0.0):.2f}, 最大时长={int(max_vocal_seconds or 0)}s"
    )
    return task_id, None, None, None, None, None, None, None, status_text, "", None, None, [], _task_table_rows()


def update_reference_mode_defaults(reference_mode_label):
    mode = _reference_mode_key(reference_mode_label)
    if mode == "style_timbre":
        return (
            REFERENCE_STYLE_PROMPT,
            gr.update(value=0.32, label="参考潜变量强度（风格音色新旋律建议 0.25-0.40；0.80+ 会明显复制旋律）"),
            gr.update(value=10.0, label="文本引导系数（风格音色新旋律建议 9-12）"),
            gr.update(value=140, label="推理步数"),
        )
    if mode == "reconstruct":
        return (
            REFERENCE_RECONSTRUCTION_PROMPT,
            gr.update(value=0.90, label="参考潜变量强度（重构建议 0.85+，会保留原旋律）"),
            gr.update(value=4.5, label="文本引导系数（参考重构建议 3-5）"),
            gr.update(value=120, label="推理步数"),
        )
    if mode == "melody":
        return (
            REFERENCE_MELODY_PROMPT,
            gr.update(value=0.82, label="参考潜变量强度（旋律保留建议 0.75-0.90）"),
            gr.update(value=6.0, label="文本引导系数（旋律保留建议 5-7）"),
            gr.update(value=100, label="推理步数"),
        )
    return (
        _reference_default_prompt(mode),
        gr.update(value=0.60, label="参考潜变量强度"),
        gr.update(value=10.0, label="文本引导系数"),
        gr.update(value=80, label="推理步数"),
    )


def submit_batch_task(batch_prompts, duration, seed, guidance, infer_steps, lora_weight, model_key):
    task_seed = _resolve_seed(seed)
    params = {
        "batch_prompts": batch_prompts or "",
        "duration": int(duration),
        "seed": task_seed,
        "guidance": float(guidance),
        "infer_steps": int(infer_steps),
        "lora_weight": float(lora_weight),
        "model_key": model_key,
    }
    task_id = _create_task("batch", params)
    _submit_task(task_id)
    return task_id, [], f"已提交批量任务: {task_id}\n输出会持久化到: {WEB_OUTPUT_DIR}", _task_table_rows()


def _plot_waveform(y: np.ndarray, sr: int, output_dir: str | None = None, name: str = "waveform.png") -> str:
    """Generate waveform, spectrogram, and loudness diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = WEB_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    if y.size == 0:
        raise ValueError("音频为空，无法绘制波形")

    _use_scientific_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.2), gridspec_kw={"height_ratios": [1.0, 1.25, 0.9]})
    t = np.arange(len(y)) / sr

    axes[0].plot(t, y, linewidth=0.35, color="#1f77b4")
    axes[0].set_xlim(0, max(float(t[-1]), 0.01))
    axes[0].set_ylim(-1, 1)
    axes[0].set_title("Waveform and level envelope")
    axes[0].set_ylabel("Amplitude")
    frame = max(256, int(0.050 * sr))
    hop = max(128, frame // 4)
    rms = np.array([], dtype=np.float32)
    centers = np.array([], dtype=np.float32)
    if len(y) >= frame:
        rms_values, center_values = [], []
        for start in range(0, len(y) - frame + 1, hop):
            seg = y[start:start + frame]
            rms_values.append(float(np.sqrt(np.mean(seg ** 2) + 1e-12)))
            center_values.append((start + frame / 2) / sr)
        rms = np.asarray(rms_values, dtype=np.float32)
        centers = np.asarray(center_values, dtype=np.float32)
        if rms.size and np.max(rms) > 0:
            axes[0].plot(centers, np.clip(rms / np.max(rms), 0, 1), color="#d55e00", linewidth=1.0, label="normalized RMS")
            axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    if len(y) < 256:
        axes[1].plot(t, y, linewidth=0.3, color="#7c3aed")
        axes[1].set_ylabel("Amplitude")
    else:
        n_fft = min(2048, int(2 ** np.floor(np.log2(len(y)))))
        axes[1].specgram(y, NFFT=n_fft, Fs=sr, noverlap=n_fft // 2, cmap="viridis")
        axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Spectrogram")

    if rms.size:
        env_db = 20 * np.log10(np.maximum(rms, 1e-6))
        axes[2].plot(centers, env_db, color="#009e73", linewidth=1.2)
        axes[2].fill_between(centers, env_db, np.min(env_db), color="#009e73", alpha=0.12)
    peak = float(np.max(np.abs(y)))
    rms_all = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    crest = peak / max(rms_all, 1e-8)
    axes[2].set_title(f"Loudness envelope | peak={peak:.3f}, RMS={rms_all:.3f}, crest={crest:.2f}")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Level (dB)")
    axes[2].set_xlim(0, max(float(t[-1]), 0.01))

    plt.tight_layout()
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ============================================================
# Gradio 界面（中文）
# ============================================================

APP_CSS = """
.app-title h1 { margin-bottom: 0.2rem; }
.model-status { border-left: 4px solid #2563eb; padding-left: 0.9rem; }
.model-status code { white-space: normal; word-break: break-all; }
.quick-note { border-left: 4px solid #16a34a; padding-left: 0.9rem; }
.top-summary-row {
  align-items: stretch;
  gap: 1.25rem;
  margin-bottom: 0.75rem;
}
.top-summary-text { min-width: 0; }
.top-summary-media {
  display: flex;
  align-items: stretch;
}
.top-reference-image {
  width: 100%;
  min-height: 240px;
  max-height: 300px;
  overflow: hidden;
  margin: 0;
}
.top-reference-image img {
  width: 100% !important;
  height: 100% !important;
  min-height: 240px !important;
  max-height: 300px !important;
  object-fit: cover !important;
  object-position: 50% 48% !important;
  border-radius: 8px;
}
.voice-status { border-left: 4px solid #7c3aed; padding-left: 0.9rem; }
.voice-video video {
  width: 100% !important;
  max-height: 260px !important;
  object-fit: cover !important;
  border-radius: 8px;
}
.voice-video .download,
.voice-video .share-button {
  display: none !important;
}
.mel-full-image img {
  width: 100% !important;
  height: auto !important;
  object-fit: contain !important;
}
@media (max-width: 820px) {
  .top-reference-image {
    min-height: 180px;
    max-height: 220px;
  }
  .top-reference-image img {
    min-height: 180px !important;
    max-height: 220px !important;
  }
}
textarea, input, .wrap { font-size: 14px; }
.paper-wrap {
  max-width: 980px;
  margin: 0 auto;
  padding: 34px 46px;
  background: #fbfaf6 !important;
  color: #111827 !important;
  font-family: "Noto Serif SC", "Source Han Serif SC", "Songti SC", "SimSun", "Microsoft YaHei", serif;
  font-size: 16px;
  line-height: 1.58;
  border: 1px solid #d6d3c8;
  box-shadow: 0 1px 6px rgba(15, 23, 42, 0.08);
}
.paper-wrap, .paper-wrap * { color: #111827 !important; }
.paper-title-block { text-align: center; border-bottom: 1px solid #d1d5db; padding-bottom: 18px; margin-bottom: 22px; }
.paper-title-block h1 { font-size: 28px; line-height: 1.16; margin: 8px 0 12px; font-weight: 700; }
.paper-venue, .paper-date, .paper-affiliation { color: #4b5563 !important; font-size: 13px; }
.paper-authors { font-size: 16px; font-weight: 600; margin-bottom: 4px; }
.paper-wrap h2 { font-size: 19px; margin: 24px 0 8px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }
.paper-wrap p { text-align: justify; margin: 8px 0 12px; }
.paper-abstract { border: 1px solid #d1d5db; padding: 12px 16px; background: #f3f4f6 !important; }
.paper-abstract h2 { border: none; margin-top: 0; }
.paper-table-caption { text-align: center; font-weight: 600; margin: 18px 0 6px; }
.paper-table-wrap { overflow-x: auto; margin-bottom: 10px; }
.paper-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.paper-table th, .paper-table td { border: 1px solid #9ca3af; padding: 7px 8px; vertical-align: top; }
.paper-table th { background: #e5e7eb !important; font-weight: 700; }
.paper-table td { background: #fffef9 !important; }
.paper-figure { margin: 22px auto; text-align: center; }
.paper-figure img { max-width: 100%; border: 1px solid #d1d5db; background: #ffffff !important; }
.paper-figure figcaption { font-size: 14px; color: #374151 !important; margin-top: 7px; font-style: italic; }
.paper-wrap pre { background: #0f172a !important; color: #e5e7eb !important; padding: 12px 14px; border-radius: 4px; overflow-x: auto; font-size: 13px; }
.paper-wrap pre, .paper-wrap pre * { color: #e5e7eb !important; }
.paper-refs li { margin-bottom: 5px; }
"""

def build_ui():
    """构建 Gradio 界面"""

    # 加载报告数据
    report_dir = os.path.join(PROJECT_ROOT, "report")
    img_dir = os.path.join(PROJECT_ROOT, "data", "reports")

    report_texts = {}
    report_files = {
        "技术方案.md": ["技术方案.md", "#U6280#U672f#U65b9#U6848.md"],
        "实验报告.md": ["实验报告.md", "#U5b9e#U9a8c#U62a5#U544a.md"],
        "答辩PPT提纲.md": ["答辩PPT提纲.md", "#U7b54#U8fa9PPT#U63d0#U7eb2.md"],
    }
    for title, candidates in report_files.items():
        path = next((os.path.join(report_dir, name) for name in candidates
                     if os.path.exists(os.path.join(report_dir, name))), None)
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                report_texts[title] = f.read()

    report_images = {}
    for name in ["genre_distribution.png", "bpm_distribution.png", "baseline_vs_finetuned_metrics.png",
                 "baseline_bpm_by_prompt.png", "baseline_summary.png"]:
        path = os.path.join(img_dir, name)
        if os.path.exists(path):
            report_images[name] = path

    with gr.Blocks(title="EDM-Adapter 音乐生成", analytics_enabled=False) as demo:
        with gr.Row(elem_classes=["top-summary-row"]):
            with gr.Column(scale=5, min_width=520, elem_classes=["top-summary-text"]):
                gr.Markdown(
                    "# EDM-Adapter: 音乐生成实验台\n"
                    "ACE-Step base + 本地 Avicii LoRA 微调 + 授权音色转换",
                    elem_classes=["app-title"],
                )
                model_status_panel = gr.Markdown(get_model_status_markdown(), elem_classes=["model-status"])
                gr.Markdown(
                    f"微调生成在“生成音乐”页选择 LoRA；音色切换在“授权音色转换”页上传歌曲并调用 Seed-VC/外部 VC 命令。任务队列默认并行数为 {MAX_PARALLEL_JOBS}。勾选基线对比会同时提交 LoRA 与 baseline 两个任务，并分别保存音频、日志和 latent 可视化。",
                    elem_classes=["quick-note"],
                )
            if os.path.exists(TRUE_MUSIC_IMAGE_PATH):
                with gr.Column(scale=2, min_width=280, elem_classes=["top-summary-media"]):
                    gr.Image(
                        value=TRUE_MUSIC_IMAGE_PATH,
                        show_label=False,
                        interactive=False,
                        container=False,
                        elem_classes=["top-reference-image"],
                    )

        with gr.Tabs():
            # ---- 生成音乐 ----
            with gr.Tab("生成音乐"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 歌曲描述")
                        user_prompt = gr.Textbox(
                            label="描述你想要的音乐（自然语言）",
                            placeholder="例如: Martin Garrix 风格的大房间浩室，明亮的超级锯齿波主旋律，厚重的底鼓，节日氛围",
                            lines=3,
                        )

                        gr.Markdown("### 风格与参数")
                        gr.Markdown("有自定义提示词时，系统只追加短质量约束；风格模板只在提示词为空时兜底。")
                        style = gr.Dropdown(
                            choices=list(STYLE_LABELS.values()),
                            value=STYLE_LABELS["techno"],
                            label="音乐风格",
                        )
                        bpm = gr.Slider(
                            60,
                            180,
                            value=128,
                            step=1,
                            label="BPM 速度（用户指定，不随风格自动修改）",
                        )
                        mood = gr.Dropdown(
                            choices=list(MOOD_LABELS.values()),
                            value=MOOD_LABELS["黑暗"],
                            label="情绪氛围",
                        )
                        duration = gr.Slider(5, 240, value=10, step=5, label="时长（秒）")

                        gr.Markdown("### 人声与歌词")
                        lyrics = gr.Textbox(
                            label="歌词（留空则生成纯器乐）",
                            placeholder="支持中英日韩等语言，留空为纯器乐模式",
                            lines=4,
                        )

                        gr.Markdown("### 模型选择")
                        model_choices = _model_choices()
                        model_selector = gr.Dropdown(
                            choices=model_choices,
                            value=_default_model_choice(model_choices),
                            label="选择模型",
                            info="默认优先使用最新 LoRA；也可切回基线或选择完整 CKPT。",
                        )
                        refresh_btn = gr.Button("刷新模型列表", size="sm")
                        lora_test_prompt = gr.Textbox(
                            label="LoRA 明显效果测试 Prompt",
                            value=LORA_TEST_PROMPT,
                            lines=3,
                            interactive=True,
                        )
                        use_lora_prompt_btn = gr.Button("使用这个 Prompt", size="sm")

                        gr.Markdown("### 生成参数")
                        seed = gr.Number(value=-1, label="随机种子（-1 表示每次随机）", precision=0)
                        guidance = gr.Slider(1.0, 30.0, value=15.0, step=0.5,
                                             label="引导系数（越高越严格遵循提示词）")
                        infer_steps = gr.Slider(10, 150, value=60, step=5,
                                                label="推理步数（越多质量越好，越慢）")
                        lora_weight = gr.Slider(0.0, 3.0, value=1.2, step=0.05,
                                                label="LoRA 强度（LoRA / 完整 CKPT 生效）")
                        with gr.Accordion("高级采样参数", open=False):
                            scheduler_type = gr.Dropdown(
                                choices=["euler", "heun"],
                                value="euler",
                                label="采样器",
                            )
                            cfg_type = gr.Dropdown(
                                choices=["apg", "cfg"],
                                value="apg",
                                label="CFG 类型",
                            )
                            omega_scale = gr.Slider(1.0, 20.0, value=10.0, step=0.5, label="Omega scale")
                            guidance_interval = gr.Slider(0.10, 1.00, value=0.50, step=0.05, label="Guidance interval")
                            min_guidance_scale = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label="最小 guidance")
                            use_erg = gr.Checkbox(value=True, label="启用 ERG 增强条件")
                            post_process = gr.Checkbox(value=True, label="启用 EDM 后处理（EQ / 立体声增强 / 限幅）")
                        enable_stems = gr.Checkbox(value=False, label="生成后分离鼓 / 贝斯 / 人声 / 其他（更慢）")
                        compare_baseline = gr.Checkbox(value=False, label="同时提交基线模型任务用于对比")

                        gr.Markdown(
                            "示例 Prompt：\n"
                            + "\n".join(f"- `{example[0]}`" for example in EXAMPLE_PROMPTS[:3])
                        )

                        gen_btn = gr.Button("开始生成", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### 生成结果")
                        current_task_ids = gr.Textbox(label="当前任务ID", interactive=False)
                        status = gr.Textbox(label="状态", interactive=False, lines=8)
                        compare_info = gr.Textbox(label="对比说明", interactive=False, lines=4)
                        full_mix = gr.Audio(label="选择模型音频（微调 / 当前模型）", type="filepath")
                        baseline_mix = gr.Audio(label="基线对比音频（ACE-Step 原始基线）", type="filepath")
                        analysis = gr.Textbox(label="音频分析", interactive=False, lines=8)
                        progress_visual = gr.Gallery(
                            label="选择模型生成过程可视化（流程概览 / denoising 曲线 / latent 关键帧）",
                            columns=4,
                            object_fit="contain",
                            preview=True,
                        )
                        waveform = gr.Image(label="选择模型音频诊断图（waveform / spectrogram / loudness）")
                        mel_spectrogram = gr.Image(
                            label="选择模型完整 Mel 图谱（末尾频谱负形 AI生成 水印）",
                            elem_classes=["mel-full-image"],
                        )
                        baseline_progress_visual = gr.Gallery(
                            label="基线生成过程可视化（流程概览 / denoising 曲线 / latent 关键帧）",
                            columns=4,
                            object_fit="contain",
                            preview=True,
                        )
                        baseline_waveform = gr.Image(label="基线音频诊断图（waveform / spectrogram / loudness）")
                        baseline_mel_spectrogram = gr.Image(
                            label="基线完整 Mel 图谱（末尾频谱负形 AI生成 水印）",
                            elem_classes=["mel-full-image"],
                        )
                        generation_task_table = gr.Dataframe(
                            headers=TASK_TABLE_HEADERS,
                            value=_task_table_rows(),
                            label="任务列表",
                            interactive=False,
                            wrap=True,
                        )

                        gr.Markdown("### 分离音轨（鼓 / 贝斯 / 人声 / 其他）")
                        gr.Markdown("由 Demucs 自动分离，可单独播放或下载")

                        stem_drums = gr.Audio(label="鼓 Drums", type="filepath", interactive=False)
                        stem_bass = gr.Audio(label="贝斯 Bass", type="filepath", interactive=False)
                        stem_vocals = gr.Audio(label="人声/旋律 Vocals", type="filepath", interactive=False)
                        stem_other = gr.Audio(label="其他 (和弦/Pad)", type="filepath", interactive=False)

                gen_btn.click(
                    fn=submit_generation_tasks,
                    inputs=[
                        user_prompt, style, bpm, mood, duration, lyrics, seed, guidance,
                        infer_steps, lora_weight, scheduler_type, cfg_type, omega_scale,
                        guidance_interval, min_guidance_scale, use_erg, post_process,
                        enable_stems, model_selector, compare_baseline,
                    ],
                    outputs=[
                        current_task_ids, full_mix, baseline_mix, compare_info,
                        stem_drums, stem_bass, stem_vocals, stem_other,
                        analysis, status, progress_visual, waveform,
                        mel_spectrogram, baseline_progress_visual, baseline_waveform,
                        baseline_mel_spectrogram, generation_task_table,
                    ],
                )

                generation_timer = gr.Timer(3)
                generation_timer.tick(
                    fn=poll_generation_tasks,
                    inputs=[current_task_ids],
                    outputs=[
                        full_mix, baseline_mix, compare_info,
                        stem_drums, stem_bass, stem_vocals, stem_other,
                        analysis, status, progress_visual, waveform,
                        mel_spectrogram, baseline_progress_visual, baseline_waveform,
                        baseline_mel_spectrogram, generation_task_table,
                    ],
                )

                # 刷新模型列表
                def refresh_models():
                    choices = _model_choices()
                    return gr.update(choices=choices, value=_default_model_choice(choices)), get_model_status_markdown()

                refresh_btn.click(fn=refresh_models, inputs=[], outputs=[model_selector, model_status_panel])

                def use_lora_prompt():
                    return (
                        LORA_TEST_PROMPT,
                        STYLE_LABELS["house"],
                        129,
                        MOOD_LABELS["高能"],
                        10,
                        "",
                    )

                use_lora_prompt_btn.click(
                    fn=use_lora_prompt,
                    inputs=[],
                    outputs=[user_prompt, style, bpm, mood, duration, lyrics],
                )

            # ---- 授权音色转换 ----
            with gr.Tab("授权音色转换"):
                voice_model_status = gr.Markdown(get_voice_conversion_status_markdown(), elem_classes=["voice-status"])
                with gr.Row():
                    with gr.Column(scale=1):
                        voice_source_audio = gr.Audio(label="上传歌曲", type="filepath", sources=["upload"])
                        voice_model_dir = gr.Textbox(
                            label="目标音色模型目录",
                            value=VOICE_MODEL_DIR,
                            lines=2,
                        )
                        voice_pitch_shift = gr.Slider(
                            -24,
                            24,
                            value=0,
                            step=1,
                            label="Pitch shift（半音）",
                        )
                        voice_use_demucs = gr.Checkbox(value=True, label="先用 Demucs 分离人声再转换")
                        voice_diffusion_steps = gr.Slider(
                            4,
                            50,
                            value=30,
                            step=1,
                            label="Seed-VC 扩散步数（30-50 更少沙沙声但更慢）",
                        )
                        voice_cfg_rate = gr.Slider(
                            0.0,
                            1.5,
                            value=0.75,
                            step=0.05,
                            label="Seed-VC CFG（0.6-0.9 通常较稳）",
                        )
                        voice_dehiss_strength = gr.Slider(
                            0.0,
                            1.0,
                            value=0.40,
                            step=0.05,
                            label="去沙哑/齿音强度",
                        )
                        voice_emotion_strength = gr.Slider(
                            0.0,
                            1.0,
                            value=0.55,
                            step=0.05,
                            label="情感/力度保留强度",
                        )
                        voice_vocal_gain = gr.Slider(
                            -6.0,
                            6.0,
                            value=1.5,
                            step=0.5,
                            label="人声重混增益 dB",
                        )
                        voice_max_seconds = gr.Slider(
                            0,
                            240,
                            value=30,
                            step=5,
                            label="最大转换时长（秒，0=整首；CPU 建议 30-60）",
                        )
                        voice_refresh_btn = gr.Button("刷新模型状态", size="sm")
                        voice_convert_btn = gr.Button("开始授权音色转换", variant="primary", size="lg")
                        if os.path.exists(GSD_VIDEO_PATH):
                            gr.Video(
                                value=GSD_VIDEO_PATH,
                                label="音色转换视频",
                                autoplay=True,
                                loop=True,
                                include_audio=False,
                                interactive=False,
                                height=260,
                                elem_id="gsd-loop-video",
                                elem_classes=["voice-video"],
                            )
                            gr.HTML(
                                value="",
                                js_on_load="""
const muteGsdVideo = () => {
  const video = document.querySelector('#gsd-loop-video video');
  if (!video) return;
  video.muted = true;
  video.defaultMuted = true;
  video.loop = true;
  video.autoplay = true;
  video.playsInline = true;
  video.setAttribute('muted', '');
  video.setAttribute('loop', '');
  video.setAttribute('autoplay', '');
  video.setAttribute('playsinline', '');
  video.play().catch(() => {});
};
muteGsdVideo();
setTimeout(muteGsdVideo, 500);
setTimeout(muteGsdVideo, 1500);
""",
                            )

                    with gr.Column(scale=2):
                        voice_task_ids = gr.Textbox(label="当前音色转换任务ID", interactive=False)
                        voice_status = gr.Textbox(label="音色转换状态", interactive=False, lines=12)
                        voice_mix = gr.Audio(label="新歌成品（换音色人声 + 伴奏）", type="filepath", interactive=False)
                        voice_remix_btn = gr.Button("重新合成新歌", size="sm")
                        voice_converted_vocal = gr.Audio(label="换音色后人声", type="filepath", interactive=False)
                        with gr.Row():
                            voice_vocals = gr.Audio(label="原始分离人声 Vocals", type="filepath", interactive=False)
                            voice_accompaniment = gr.Audio(label="分离伴奏（Drums + Bass + Other）", type="filepath", interactive=False)
                        with gr.Accordion("分离音轨明细", open=False):
                            with gr.Row():
                                voice_drums = gr.Audio(label="Drums", type="filepath", interactive=False)
                                voice_bass = gr.Audio(label="Bass", type="filepath", interactive=False)
                                voice_other = gr.Audio(label="Other", type="filepath", interactive=False)
                        voice_analysis = gr.Textbox(label="音频分析", interactive=False, lines=8)
                        voice_waveform = gr.Image(label="音频诊断图（waveform / spectrogram / loudness）")
                        voice_mel_spectrogram = gr.Image(
                            label="新歌完整 Mel 图谱（末尾频谱负形 AI生成 水印）",
                            elem_classes=["mel-full-image"],
                        )
                        voice_files = gr.File(label="该任务全部音频文件", file_count="multiple", type="filepath")
                        voice_task_table = gr.Dataframe(
                            headers=TASK_TABLE_HEADERS,
                            value=_task_table_rows(),
                            label="任务列表",
                            interactive=False,
                            wrap=True,
                        )

                voice_refresh_btn.click(fn=get_voice_conversion_status_markdown, inputs=[], outputs=[voice_model_status])
                voice_convert_btn.click(
                    fn=submit_voice_conversion_task,
                    inputs=[
                        voice_source_audio,
                        voice_model_dir,
                        voice_pitch_shift,
                        voice_use_demucs,
                        voice_diffusion_steps,
                        voice_cfg_rate,
                        voice_dehiss_strength,
                        voice_emotion_strength,
                        voice_vocal_gain,
                        voice_max_seconds,
                    ],
                    outputs=[
                        voice_task_ids,
                        voice_mix,
                        voice_converted_vocal,
                        voice_vocals,
                        voice_accompaniment,
                        voice_drums,
                        voice_bass,
                        voice_other,
                        voice_status,
                        voice_analysis,
                        voice_waveform,
                        voice_mel_spectrogram,
                        voice_files,
                        voice_task_table,
                    ],
                )
                voice_remix_btn.click(
                    fn=remix_voice_conversion_task,
                    inputs=[voice_task_ids],
                    outputs=[
                        voice_mix,
                        voice_converted_vocal,
                        voice_vocals,
                        voice_accompaniment,
                        voice_drums,
                        voice_bass,
                        voice_other,
                        voice_status,
                        voice_analysis,
                        voice_waveform,
                        voice_mel_spectrogram,
                        voice_files,
                        voice_task_table,
                    ],
                )
                voice_timer = gr.Timer(3)
                voice_timer.tick(
                    fn=poll_voice_conversion_tasks,
                    inputs=[voice_task_ids],
                    outputs=[
                        voice_mix,
                        voice_converted_vocal,
                        voice_vocals,
                        voice_accompaniment,
                        voice_drums,
                        voice_bass,
                        voice_other,
                        voice_status,
                        voice_analysis,
                        voice_waveform,
                        voice_mel_spectrogram,
                        voice_files,
                        voice_task_table,
                    ],
                )

            with gr.Tab("批量演示"):
                gr.Markdown("## 微调模型批量生成")
                gr.Markdown("每行一个完整提示词，最多一次生成 5 条。此模式不做 Demucs 分离，适合快速对比微调模型在不同风格提示词下的输出。")

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_prompts = gr.Textbox(
                            label="批量提示词",
                            value="\n".join(example[0] for example in EXAMPLE_PROMPTS[:4]),
                            lines=8,
                        )
                        batch_model_choices = _model_choices()
                        batch_model_selector = gr.Dropdown(
                            choices=batch_model_choices,
                            value=_default_model_choice(batch_model_choices),
                            label="选择模型",
                            info="默认使用最新 LoRA，也可选择完整 CKPT",
                        )
                        batch_duration = gr.Slider(5, 90, value=10, step=5, label="单条时长（秒）")
                        batch_seed = gr.Number(value=42, label="起始随机种子", precision=0)
                        batch_guidance = gr.Slider(5.0, 25.0, value=15.0, step=0.5, label="引导系数")
                        batch_steps = gr.Slider(20, 100, value=50, step=5, label="推理步数")
                        batch_lora_weight = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA 强度（LoRA / 完整 CKPT 生效）")
                        batch_btn = gr.Button("开始批量生成", variant="primary")

                    with gr.Column(scale=1):
                        batch_task_ids = gr.Textbox(label="当前批量任务ID", interactive=False)
                        batch_status = gr.Textbox(label="批量状态", interactive=False, lines=14)
                        batch_files = gr.File(label="生成结果 WAV", file_count="multiple", type="filepath")
                        batch_task_table = gr.Dataframe(
                            headers=TASK_TABLE_HEADERS,
                            value=_task_table_rows(),
                            label="任务列表",
                            interactive=False,
                            wrap=True,
                        )

                batch_btn.click(
                    fn=submit_batch_task,
                    inputs=[batch_prompts, batch_duration, batch_seed, batch_guidance, batch_steps, batch_lora_weight, batch_model_selector],
                    outputs=[batch_task_ids, batch_files, batch_status, batch_task_table],
                )

                batch_timer = gr.Timer(3)
                batch_timer.tick(
                    fn=poll_batch_tasks,
                    inputs=[batch_task_ids],
                    outputs=[batch_files, batch_status, batch_task_table],
                )

            # ---- 任务队列 ----
            with gr.Tab("任务队列"):
                gr.Markdown("## 全部生成任务和音频")
                queue_task_choices = _task_choices()
                with gr.Row():
                    task_selector = gr.Dropdown(
                        choices=queue_task_choices,
                        value=queue_task_choices[0] if queue_task_choices else None,
                        label="选择任务",
                        scale=4,
                    )
                    queue_refresh_btn = gr.Button("刷新任务列表并加载详情", size="sm", scale=1)
                queue_table = gr.Dataframe(
                    headers=TASK_TABLE_HEADERS,
                    value=_task_table_rows(),
                    label="任务队列",
                    interactive=False,
                    wrap=True,
                )
                queue_status = gr.Textbox(label="任务日志", interactive=False, lines=14)
                queue_analysis = gr.Textbox(label="音频分析", interactive=False, lines=8)
                queue_mix = gr.Audio(label="完整混音 / 首个音频", type="filepath")
                with gr.Row():
                    queue_drums = gr.Audio(label="鼓 Drums", type="filepath", interactive=False)
                    queue_bass = gr.Audio(label="贝斯 Bass", type="filepath", interactive=False)
                with gr.Row():
                    queue_vocals = gr.Audio(label="人声/旋律 Vocals", type="filepath", interactive=False)
                    queue_other = gr.Audio(label="其他 Other", type="filepath", interactive=False)
                queue_progress = gr.Gallery(
                    label="该任务完整生成过程（全部 diffusion steps）",
                    columns=4,
                    object_fit="contain",
                    preview=True,
                )
                queue_waveform = gr.Image(label="Waveform / Spectrogram")
                queue_mel_spectrogram = gr.Image(
                    label="完整 Mel 图谱（末尾频谱负形 AI生成 水印）",
                    elem_classes=["mel-full-image"],
                )
                queue_audio_table = gr.Dataframe(
                    headers=AUDIO_TABLE_HEADERS,
                    value=[],
                    label="音频清单（可点击某一行播放）",
                    interactive=False,
                    wrap=True,
                )
                queue_selected_audio = gr.Audio(label="清单选中音频", type="filepath")
                queue_selected_note = gr.Textbox(label="清单选中说明", interactive=False, lines=3)
                queue_files = gr.File(label="该任务全部音频", file_count="multiple", type="filepath")

                queue_outputs = [
                    queue_table, task_selector, queue_mix, queue_drums, queue_bass, queue_vocals,
                    queue_other, queue_analysis, queue_status, queue_progress, queue_waveform,
                    queue_mel_spectrogram, queue_files, queue_audio_table, queue_selected_audio,
                    queue_selected_note,
                ]
                queue_detail_outputs = [
                    queue_mix, queue_drums, queue_bass, queue_vocals,
                    queue_other, queue_analysis, queue_status, queue_progress, queue_waveform,
                    queue_mel_spectrogram, queue_files, queue_audio_table, queue_selected_audio,
                    queue_selected_note,
                ]
                queue_refresh_btn.click(
                    fn=refresh_task_dashboard,
                    inputs=[task_selector],
                    outputs=queue_outputs,
                )
                queue_table.select(
                    fn=select_task_from_table,
                    inputs=[],
                    outputs=[task_selector, *queue_detail_outputs],
                )
                task_selector.change(
                    fn=load_selected_task_detail,
                    inputs=[task_selector],
                    outputs=queue_detail_outputs,
                )
                queue_audio_table.select(
                    fn=select_audio_from_table,
                    inputs=[task_selector],
                    outputs=[queue_selected_audio, queue_selected_note],
                )
                queue_timer = gr.Timer(3)
                queue_timer.tick(
                    fn=refresh_task_table_only,
                    inputs=[],
                    outputs=[queue_table],
                )

            # ---- 微调说明 ----
            with gr.Tab("微调说明"):
                gr.Markdown("## 当前微调与生成说明")
                gr.Markdown(get_model_status_markdown())
                gr.Markdown(get_finetuned_readme_markdown())

            # ---- 技术报告 ----
            with gr.Tab("技术报告") as report_tab:
                gr.Markdown("## 论文式技术报告")
                gr.Markdown(
                    "该页面按论文结构展示：标题、摘要、关键词、编号章节、编号表格、编号图、实验协议和参考文献。"
                    "图表由当前项目的 `dataset/`、`outputs/web_generations/` 和音色转换任务日志自动生成。"
                )
                report_refresh_btn = gr.Button("刷新论文报告与图表", size="sm")
                paper_report = gr.HTML(value=build_paper_report_html(PROJECT_ROOT))
                report_refresh_btn.click(
                    fn=lambda: build_paper_report_html(PROJECT_ROOT),
                    inputs=[],
                    outputs=[paper_report],
                )

                if report_texts:
                    with gr.Accordion("旧版 Markdown 报告归档", open=False):
                        if report_images:
                            with gr.Row():
                                for name, path in report_images.items():
                                    title = name.replace("_", " ").replace(".png", "").title()
                                    gr.Image(value=path, label=title)
                        for title, content in report_texts.items():
                            gr.Markdown("---")
                            gr.Markdown(f"## {title.replace('.md', '')}")
                            gr.Markdown(content)

        gr.Markdown("---\n**EDM-Adapter** | ACE-Step + LoRA + Seed-VC | 电子音乐生成与授权音色转换系统")

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=32, default_concurrency_limit=16)
    demo.launch(
        server_name=os.environ.get("EDM_GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("EDM_GRADIO_PORT", "7860")),
        share=True,
        inbrowser=False,
        show_error=True,
        enable_monitoring=False,
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )
