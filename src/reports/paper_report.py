"""Render a paper-style technical report for the Gradio UI."""

from __future__ import annotations

import base64
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _table(headers: list[str], rows: Iterable[Iterable], caption: str) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>"
        )
    return (
        f'<div class="paper-table-caption">{html.escape(caption)}</div>'
        '<div class="paper-table-wrap"><table class="paper-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _img_data_uri(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _figure(path: Path, caption: str) -> str:
    return (
        '<figure class="paper-figure">'
        f'<img src="{_img_data_uri(path)}" alt="{html.escape(caption)}" />'
        f'<figcaption>{html.escape(caption)}</figcaption>'
        "</figure>"
    )


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_architecture(path: Path) -> None:
    if path.exists():
        return
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axis("off")
    boxes = [
        ("Caption\nPrompt", 0.05, 0.62, 0.16, 0.20),
        ("EDM Metadata\nsection, BPM,\nenergy, subgenre", 0.05, 0.24, 0.16, 0.26),
        ("Control Curve\nT x 36", 0.28, 0.24, 0.16, 0.26),
        ("UMT5 Text\nEmbeddings", 0.28, 0.62, 0.16, 0.20),
        ("Control Token\nProjector", 0.50, 0.24, 0.17, 0.26),
        ("Adapter Router\nshared + experts", 0.50, 0.62, 0.17, 0.20),
        ("ACE-Step Diffusion\nTransformer", 0.74, 0.46, 0.20, 0.24),
        ("Generated EDM\nAudio", 0.74, 0.14, 0.20, 0.18),
    ]
    for text, x, y, w, h in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor="#f8fafc", edgecolor="#0f172a", linewidth=1.2)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.21, 0.72), (0.28, 0.72)),
        ((0.21, 0.37), (0.28, 0.37)),
        ((0.44, 0.37), (0.50, 0.37)),
        ((0.67, 0.37), (0.74, 0.53)),
        ((0.44, 0.72), (0.50, 0.72)),
        ((0.67, 0.72), (0.74, 0.58)),
        ((0.84, 0.46), (0.84, 0.32)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.4, color="#1d4ed8"))
    ax.text(
        0.5,
        0.04,
        "Structure-aware routing modulates LoRA experts; control tokens inject latent-frame EDM attributes.",
        ha="center",
        va="center",
        fontsize=9,
        color="#334155",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset_distribution(path: Path, stats: dict) -> None:
    if path.exists():
        return
    plt = _ensure_matplotlib()
    subgenre = stats.get("subgenre", {})
    section = stats.get("section", {})
    energy = stats.get("energy", {})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    top_sub = sorted(subgenre.items(), key=lambda kv: kv[1])[-9:]
    axes[0].barh([k for k, _ in top_sub], [v for _, v in top_sub], color="#2563eb")
    axes[0].set_title("Subgenre")
    axes[0].set_xlabel("clips")

    sec_items = sorted(section.items(), key=lambda kv: kv[1], reverse=True)
    axes[1].bar([k for k, _ in sec_items], [v for _, v in sec_items], color="#16a34a")
    axes[1].set_title("Section")
    axes[1].set_ylabel("clips")
    axes[1].tick_params(axis="x", rotation=35)

    energy_items = sorted(energy.items(), key=lambda kv: kv[1], reverse=True)
    axes[2].bar([k for k, _ in energy_items], [v for _, v in energy_items], color="#dc2626")
    axes[2].set_title("Energy")
    axes[2].tick_params(axis="x", rotation=20)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_control_curve(path: Path, project_root: Path, metadata_rows: list[dict]) -> None:
    if path.exists():
        return
    import torch

    plt = _ensure_matplotlib()
    schema = _read_json(project_root / "dataset" / "controls" / "schema.json", {})
    feature_names = schema.get("feature_names", [])
    selected = next((row for row in metadata_rows if row.get("section") == "drop"), None)
    if selected is None and metadata_rows:
        selected = metadata_rows[0]
    if selected is None:
        return
    control_path = project_root / "dataset" / str(selected.get("control_path", "")).replace("\\", "/")
    payload = torch.load(control_path, map_location="cpu")
    curve = payload["control"].float().numpy()
    idx = {name: i for i, name in enumerate(feature_names)}
    x = np.arange(curve.shape[0])

    fig, ax = plt.subplots(figsize=(10.6, 3.8))
    for name, label, color in [
        ("energy_value", "energy", "#dc2626"),
        ("bpm_norm", "BPM norm", "#2563eb"),
        ("beat_phase_sin", "beat phase sin", "#7c3aed"),
        ("low_freq_ratio_norm", "low-frequency ratio", "#059669"),
        ("onset_density_norm", "onset density", "#ea580c"),
        ("loop_end_marker", "loop end marker", "#0f172a"),
    ]:
        if name in idx:
            ax.plot(x, curve[:, idx[name]], label=label, linewidth=1.5, color=color)
    ax.set_title(
        f"Control curve example: {selected.get('section')} / {selected.get('subgenre')} / {selected.get('bpm')} BPM"
    )
    ax.set_xlabel("latent frame")
    ax.set_ylabel("normalized value")
    ax.set_ylim(-1.08, 1.08)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_pipeline(path: Path) -> None:
    if path.exists():
        return
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.axis("off")
    steps = [
        "Raw audio",
        "Clean\nand slice",
        "Metadata\nlabels",
        "DCAE\nlatents",
        "Control\ncurves",
        "EDM-\nStructLoRA",
        "Evaluation\nmetrics",
    ]
    xs = np.linspace(0.06, 0.88, len(steps))
    for i, (x, step) in enumerate(zip(xs, steps)):
        ax.add_patch(plt.Circle((x, 0.55), 0.055, facecolor="#eff6ff", edgecolor="#1d4ed8", lw=1.3))
        ax.text(x, 0.55, str(i + 1), ha="center", va="center", fontsize=11, weight="bold", color="#1d4ed8")
        ax.text(x, 0.30, step, ha="center", va="center", fontsize=9)
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.065, 0.55), xytext=(x + 0.065, 0.55), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.5, 0.84, "Reproducible EDM adaptation pipeline", ha="center", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _ensure_figures(project_root: Path, stats: dict, metadata_rows: list[dict]) -> dict[str, Path]:
    asset_dir = project_root / "report" / "paper_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "architecture": asset_dir / "fig1_architecture.png",
        "dataset": asset_dir / "fig2_dataset_distribution.png",
        "control": asset_dir / "fig3_control_curve.png",
        "pipeline": asset_dir / "fig4_pipeline.png",
    }
    _plot_architecture(figures["architecture"])
    _plot_dataset_distribution(figures["dataset"], stats)
    _plot_control_curve(figures["control"], project_root, metadata_rows)
    _plot_pipeline(figures["pipeline"])
    return figures


def _top_items(counter: dict, n: int = 6) -> str:
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return "; ".join(f"{key}: {value}" for key, value in items)


def _dataset_summary(project_root: Path, stats: dict, metadata_rows: list[dict]) -> dict:
    preprocess_report = project_root / "dataset" / "reports" / "preprocess_report.md"
    raw_files = stats.get("source_file_count", 0)
    processed = stats.get("processed_source_count", 0)
    clips = stats.get("sample_count", len(metadata_rows))
    controls = len(list((project_root / "dataset" / "controls").glob("*.pt")))
    latents = len(list((project_root / "dataset" / "latents").glob("*.pt")))
    split = stats.get("split", {})
    durations = [float(row.get("duration") or 0.0) for row in metadata_rows[:2000]]
    avg_duration = float(np.mean(durations)) if durations else 0.0
    return {
        "raw_files": raw_files,
        "processed_sources": processed,
        "clips": clips,
        "controls": controls,
        "latents": latents,
        "train": split.get("train", 0),
        "val": split.get("val", 0),
        "test": split.get("test", 0),
        "avg_duration": f"{avg_duration:.2f}s",
        "report": preprocess_report.exists(),
    }


def _status_rows(project_root: Path) -> list[list[str]]:
    paths = [
        ("控制曲线 schema", project_root / "dataset" / "controls" / "schema.json"),
        ("ACE latent 缓存", project_root / "dataset" / "latents"),
        ("训练集 HF Dataset", project_root / "outputs" / "datasets" / "edm_control_lora_train"),
        ("验证集 HF Dataset", project_root / "outputs" / "datasets" / "edm_control_lora_val"),
        ("测试集 HF Dataset", project_root / "outputs" / "datasets" / "edm_control_lora_test"),
        ("训练入口脚本", project_root / "scripts" / "train_edm_control_lora.py"),
        ("生成入口脚本", project_root / "scripts" / "generate_edm_control_lora.py"),
        ("本地 Avicii LoRA 训练脚本", project_root / "scripts" / "train_avicii_local_lora.py"),
        ("本地 Avicii LoRA 生成脚本", project_root / "scripts" / "generate_avicii_local_lora.py"),
        ("无训练参考音频生成脚本", project_root / "scripts" / "generate_reference_style.py"),
    ]
    rows = []
    for name, path in paths:
        rows.append([name, "已存在" if path.exists() else "缺失", path.relative_to(project_root).as_posix()])
    return rows


def build_paper_report_html(project_root: str | Path) -> str:
    project_root = Path(project_root)
    generated_paper = project_root / "report" / "intelligent_speech_processing_paper" / "paper_web.html"
    if generated_paper.exists():
        return generated_paper.read_text(encoding="utf-8")
    try:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, str(project_root / "report" / "intelligent_speech_processing_paper" / "build_isp_paper.py")],
            cwd=str(project_root),
            check=True,
            timeout=180,
        )
    except Exception:
        pass
    if generated_paper.exists():
        return generated_paper.read_text(encoding="utf-8")
    return """
<article class="paper-wrap">
  <header class="paper-title-block">
    <div class="paper-venue">科研原型技术报告</div>
    <h1>音乐生成与授权音色转换系统</h1>
    <div class="paper-affiliation">请先运行 report/intelligent_speech_processing_paper/build_isp_paper.py 生成最新论文页面。</div>
  </header>
</article>
"""

    stats = _read_json(project_root / "dataset" / "reports" / "label_statistics.json", {})
    metadata_rows = _read_jsonl(project_root / "dataset" / "metadata.jsonl", limit=6016)
    summary = _dataset_summary(project_root, stats, metadata_rows)
    figures = _ensure_figures(project_root, stats, metadata_rows)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    dataset_table = _table(
        ["项目", "数值"],
        [
            ["原始音频文件数", summary["raw_files"]],
            ["成功处理的源文件数", summary["processed_sources"]],
            ["训练质量切片数", summary["clips"]],
            ["ACE-Step DCAE latent 数", summary["latents"]],
            ["latent 对齐控制曲线数", summary["controls"]],
            ["训练 / 验证 / 测试", f'{summary["train"]} / {summary["val"]} / {summary["test"]}'],
            ["平均切片时长", summary["avg_duration"]],
            ["路径规范", "POSIX 风格相对路径，便于 Linux 服务器复现"],
        ],
        "表 1. 数据集构建统计。",
    )
    label_table = _table(
        ["标签类别", "分布摘要"],
        [
            ["子风格", _top_items(stats.get("subgenre", {}), 7)],
            ["音乐段落", _top_items(stats.get("section", {}), 7)],
            ["能量等级", _top_items(stats.get("energy", {}), 4)],
            ["情绪标签", _top_items(stats.get("mood", {}), 7)],
            ["质量标签", _top_items(stats.get("quality", {}), 4)],
        ],
        "表 2. 用于 adapter 路由的元数据标签分布。",
    )
    method_table = _table(
        ["组件", "实现方式", "作用"],
        [
            ["共享 LoRA", "r=32，rsLoRA，作用于 attention 相关目标模块", "学习 EDM 域整体偏移。"],
            ["段落专家 LoRA", "intro / build-up / drop / breakdown / outro / loop 等 r=8 专家", "学习不同音乐段落的生成偏移。"],
            ["属性专家 LoRA", "energy 与 subgenre 专家", "增强能量和 EDM 子风格的可控性。"],
            ["Adapter 路由器", "由 section、energy、subgenre、BPM、置信度计算权重", "按样本动态激活参数高效专家。"],
            ["控制条件器", "T x 36 控制曲线映射为 8 个文本条件 token", "注入节拍相位、低频、onset、loop 边界和置信度信号。"],
        ],
        "表 3. EDM-StructLoRA 方法组件。",
    )
    local_lora_table = _table(
        ["维度", "本地低资源 LoRA 微调设定", "科研意义"],
        [
            ["训练对象", "冻结 ACE-Step MusicDCAE 与 UMT5 文本编码器，仅在 Transformer 末端少量模块注入 LoRA", "在 CPU/低资源环境下进行真实参数更新，而不是只改 prompt。"],
            ["可训练规模", "52 个可训练张量，约 1.12M 参数；基础系统总参数约 3.9B", "把训练量控制在总参数的极小比例，降低内存与时间成本。"],
            ["训练范围", "最后 2 个 Transformer block、conditioning 相关模块与 final layer LoRA", "优先修改最接近声学解码与条件融合的生成路径。"],
            ["优化目标", "使用参考曲目的 ACE-Step DCAE latent 与文本条件进行扩散噪声预测损失优化", "使 adapter 学习目标音乐域的节奏、音色、和声与 drop 组织偏移。"],
            ["推理方式", "加载 LoRA bundle，并用固定 seed 与 ACE-Step base 做对照生成", "保证微调效果可以通过同 seed、同 prompt、不同 adapter 权重进行可复现实验。"],
        ],
        "表 4. 本地低资源 LoRA 微调配置。",
    )
    reference_table = _table(
        ["阶段", "处理过程", "输出"],
        [
            ["音频标准化", "上传音频被重采样到 48 kHz、统一为双声道，并按目标时长裁剪、循环或截断", "与 MusicDCAE 编码长度对齐的 reference_style_input.wav。"],
            ["智能音频分析", "提取 BPM、RMS 能量、低频比例、频谱质心、起音密度与时长等客观声学特征", "用于日志、结果解释和参考/生成对比，不作为唯一风格来源。"],
            ["潜变量编码", "ACE-Step MusicDCAE 将参考波形编码为 z_ref", "保留节奏密度、音色轮廓、混音能量与局部结构的参考潜变量。"],
            ["扩散初始化", "以 ref_strength 控制参考潜变量与随机噪声的混合：0.35-0.55 主要迁移音色/混音质感，0.80+ 会显著保留原旋律", "无需训练的 reference-conditioned 初始 latent。"],
            ["条件去噪与解码", "基础 ACE-Step Transformer 在文本条件下重新去噪，再由 DCAE 解码为波形", "生成新的音频文件、扩散过程图、波形/频谱图与客观指标。"],
        ],
        "表 5. 无训练参考音频生成的智能音频处理流程。",
    )
    ablation_table = _table(
        ["编号", "系统", "实验目的"],
        [
            ["A0", "ACE-Step base", "未微调文本生成音乐基线。"],
            ["A1", "普通 LoRA r=64", "参数高效域适配基线。"],
            ["A2", "富元数据 caption LoRA", "验证仅靠 caption 工程是否足够。"],
            ["A3", "Section-aware LoRA", "隔离段落专家 adapter 的贡献。"],
            ["A4", "Section + attribute routed LoRA", "验证动态 adapter 路由的贡献。"],
            ["A5", "完整 EDM-StructLoRA", "router + control tokens + confidence-aware sampling。"],
        ],
        "表 6. 消融实验设计。",
    )
    metrics_table = _table(
        ["指标", "定义", "预期证明"],
        [
            ["BPM 误差", "|目标 BPM - 生成音频估计 BPM|", "属性条件路由后误差更低。"],
            ["Onset 密度误差", "目标与生成 onset density 的差异", "energy / section 改变会带来节奏密度变化。"],
            ["低频能量比例", "低频段能量占比", "drop 段低频应强于 intro / breakdown。"],
            ["Loop 相似度", "首尾 embedding 或频谱相似度", "loop prompt 应产生更连贯的边界。"],
            ["FAD / embedding distance", "生成分布与参考 EDM 分布的距离", "适配后保持音质并贴近目标域。"],
            ["人工盲听", "风格匹配、drop 冲击、loop 可用性、音质", "补充自动指标无法覆盖的主观证据。"],
        ],
        "表 7. 可控 EDM 生成的评估指标。",
    )
    status_table = _table(
        ["产物", "状态", "路径"],
        _status_rows(project_root),
        "表 8. 当前实现的可复现状态。",
    )

    return f"""
<article class="paper-wrap">
  <header class="paper-title-block">
    <div class="paper-venue">科研原型技术报告 · ACE-Step 微调与无训练参考音频生成</div>
    <h1>EDM-Adapter: LoRA Domain Adaptation and Training-Free Reference-Conditioned Music Generation</h1>
    <div class="paper-authors">EDM-Adapter 项目</div>
    <div class="paper-affiliation">文本生成音乐扩散 Transformer 的参数高效适配与参考潜变量条件生成</div>
    <div class="paper-date">生成时间：{html.escape(now)}</div>
  </header>

  <section class="paper-abstract">
    <h2>摘要</h2>
    <p>
      本报告给出 EDM-Adapter 的两条互不冲突的生成路线。第一条路线是监督式参数高效微调：
      在冻结 ACE-Step MusicDCAE 与 UMT5 文本编码器的前提下，仅训练 Transformer 末端少量 LoRA
      张量，从而让基础模型学习目标 EDM/Avicii-like 数据域的节奏、音色和编曲偏移。第二条路线是
      无训练参考音频生成：用户上传一段参考音频后，系统将其标准化并编码为 MusicDCAE 潜变量，
      再用 ref_strength 控制参考潜变量与随机噪声的混合，在基础 ACE-Step 上完成重新去噪生成。
      两条路线分别覆盖“可持续积累的风格 adapter”和“即时上传参考音频的零训练生成”场景。
    </p>
    <p><strong>关键词：</strong>文本生成音乐；LoRA；无训练生成；参考音频条件；智能音频处理；扩散 Transformer；ACE-Step。</p>
  </section>

  <section>
    <h2>1. 引言</h2>
    <p>
      文本生成音乐模型在零样本提示词条件下可以产生可听音频，但很难稳定复现某一数据域中
      反复出现的制作习惯。LoRA 微调通过真实参数更新解决长期风格适配问题；无训练参考音频
      生成则解决用户临时上传一段音频、希望立即得到相近声学结构与制作质感的问题。
    </p>
    <p>
      因此，本系统将两类方法并列实现：微调入口加载独立 LoRA bundle；无训练入口固定使用
      ACE-Step base，并只通过参考音频潜变量影响扩散初态。前端和任务队列中二者分离，避免
      LoRA 权重、缓存 pipeline 与上传参考音频流程互相污染。
    </p>
  </section>

  <section>
    <h2>2. 方法</h2>
    {_figure(figures["architecture"], "图 1. EDM-StructLoRA 总体结构：adapter 路由与控制 token 条件注入。")}
    {method_table}
    {local_lora_table}
    {reference_table}
    <p>
      对于训练 batch B，路由器根据 section、energy、subgenre、BPM、自动标签置信度和样本质量
      计算 adapter 权重，并将对应混合权重作用于 ACE-Step Transformer 的 attention 模块。
      控制条件器将每条 T x 36 控制曲线映射为 8 个可学习条件 token，并拼接到 UMT5 文本
      embedding 后面。
    </p>
    <p>
      本地低资源微调采用更保守的工程实例：只训练最后两个 Transformer block、conditioning
      模块和 final layer 的 LoRA 参数，实际可训练参数约 1.12M。该设置牺牲一部分上限能力，
      但能在本机 CPU 环境中完成真实反向传播，并保存可复用的 LoRA bundle。
    </p>
    <p>
      无训练参考音频生成不估计“艺术家标签 prompt”，也不更新权重。其核心是参考音频 latent
      conditioning：给定参考潜变量 z_ref 和随机噪声 ε，系统用 ref_strength 控制初始 latent
      的参考保留比例，再由基础模型在文本条件下扩散去噪。默认“风格音色（新旋律）”模式把强度
      限制在较低区间，并显式要求新旋律、新和弦和无人声；“参考重构”模式才使用高强度以保留
      上传音频的旋律、节奏密度和混音能量。
    </p>
  </section>

  <section>
    <h2>3. 数据集与控制资产</h2>
    {dataset_table}
    {label_table}
    {_figure(figures["dataset"], "图 2. 数据集子风格、段落和能量标签分布。")}
    {_figure(figures["control"], "图 3. 控制条件器使用的 latent 对齐 EDM 控制曲线示例。")}
  </section>

  <section>
    <h2>4. 训练与生成流程</h2>
    {_figure(figures["pipeline"], "图 4. 从原始音频到可控性评估的可复现实验流程。")}
    <p>
      监督式微调流程首先将本地参考曲目清洗为训练 clip，缓存 ACE-Step DCAE latent，并用小规模
      LoRA 在冻结基础模型上优化扩散噪声预测目标。生成时，前端可以选择 LoRA/CKPT 或 ACE-Step
      base，并可用相同 seed 进行基线对照。
    </p>
    <pre><code>python scripts/train_edm_control_lora.py \\
  --dataset-path outputs/datasets/edm_control_lora_train \\
  --checkpoint-dir models/ace-step/ACE-Step-v1-3.5B \\
  --config config/edm_control_lora.json \\
  --max-steps 10000 \\
  --learning-rate 1e-4 \\
  --precision bf16-mixed \\
  --accumulate-grad-batches 8 \\
  --devices 1</code></pre>
    <pre><code>python scripts/generate_edm_control_lora.py \\
  --prompt "uplifting piano chords, warm sidechain bass, wide supersaw lead" \\
  --lora-bundle outputs/edm_control_lora/logs/.../checkpoints/epoch=0-step=10000_edm_control_lora \\
  --duration 8 \\
  --section drop \\
  --energy high \\
  --subgenre "melodic house" \\
  --bpm 128</code></pre>
    <pre><code>python scripts/train_avicii_local_lora.py \\
  --config config/avicii_local_lora.json \\
  --max-steps 120 \\
  --device cpu</code></pre>
    <pre><code>python scripts/generate_reference_style.py \\
  --reference-audio path/to/uploaded_reference.wav \\
  --prompt "instrumental original track, same sound palette and mix energy, new melody" \\
  --duration 15 \\
  --ref-strength 0.45 \\
  --guidance-scale 9.0 \\
  --seed 42 \\
  --infer-step 100</code></pre>
    <p>
      无训练参考音频流程只调用基础模型，不读取 LoRA bundle。前端上传音频后，系统保存原始上传副本、
      生成标准化参考片段、记录参考/生成客观音频特征，并在任务目录中保存最终 wav、元数据 JSON、
      diffusion 进度图和 waveform/spectrogram。
    </p>
  </section>

  <section>
    <h2>5. 实验协议</h2>
    {ablation_table}
    {metrics_table}
    <p>
      当前报告只展示已完成的数据、方法和实验协议，不填入尚未训练得到的结果。完成 GPU 训练后，
      应对表 6 中每个系统报告表 7 指标，并补充人工盲听实验。对于无训练参考音频流程，还应固定
      seed、prompt 和参考片段，比较 ref_strength = 0.35/0.45/0.55/0.85 时的音色相似度、
      旋律新颖度、节奏保留和主观音质。
    </p>
  </section>

  <section>
    <h2>6. 实现状态</h2>
    {status_table}
    <p>
      本地实现已包含三类可运行入口：EDM-StructLoRA 研究型训练入口、本机低资源 LoRA 微调入口、
      以及无训练参考音频生成入口。前端任务队列将 LoRA 微调生成和参考音频生成作为不同 task kind
      管理；后者固定使用 base pipeline，并显式卸载 LoRA，避免与微调模型冲突。
    </p>
  </section>

  <section>
    <h2>7. 参考文献</h2>
    <ol class="paper-refs">
      <li>ACE-Step: A Step Towards Music Generation Foundation Model.</li>
      <li>Hu 等：LoRA: Low-Rank Adaptation of Large Language Models.</li>
      <li>Music ControlNet: Multiple Time-varying Controls for Music Generation.</li>
      <li>MusiConGen: Rhythm and Chord Control for Transformer-Based Text-to-Music Generation.</li>
      <li>MuseControlLite: Lightweight Control for Music Generation.</li>
      <li>DoRA、rsLoRA、AdaLoRA 与 X-LoRA 等参数高效适配基线。</li>
    </ol>
  </section>
</article>
"""
