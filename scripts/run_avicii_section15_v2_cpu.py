"""Run the section-aware Avicii LoRA v2 CPU pipeline sequentially."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "avicii_section15_v2"
DEFAULT_HF_DATASET = PROJECT_ROOT / "outputs" / "datasets" / "avicii_section15_lora_train"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "avicii_local_lora" / "logs"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "avicii_local_lora" / "section15_v2_runs"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_dataset_path(dataset_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return dataset_root / value.replace("\\", "/")


def field_files_complete(dataset_root: Path, field: str) -> bool:
    rows = read_jsonl(dataset_root / "metadata.jsonl")
    if not rows:
        return False
    for row in rows:
        path = resolve_dataset_path(dataset_root, row.get(field))
        if path is None or not path.exists():
            return False
    return True


def bundle_sort_key(path: Path) -> tuple[int, float]:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        step = int(manifest.get("global_step") or 0)
    except Exception:
        step = 0
    return step, path.stat().st_mtime


def find_latest_bundle(root: Path) -> Path | None:
    candidates = [
        Path(path).parent
        for path in glob.glob(str(root / "**" / "manifest.json"), recursive=True)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=bundle_sort_key, reverse=True)[0]


def run_step(name: str, command: list[str], env: dict[str, str]) -> None:
    started = time.time()
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {name}", flush=True)
    print(" ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=True)
    elapsed = time.time() - started
    print(f"[{datetime.now().isoformat(timespec='seconds')}] DONE {name} ({elapsed:.1f}s)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--hf-dataset", default=str(DEFAULT_HF_DATASET))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--train-last-n-blocks", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--max-total-clips", type=int, default=360)
    parser.add_argument("--max-clips-per-source", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    hf_dataset = Path(args.hf_dataset).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(max(1, args.cpu_threads))
    env["MKL_NUM_THREADS"] = str(max(1, args.cpu_threads))
    env["OPENBLAS_NUM_THREADS"] = str(max(1, args.cpu_threads))
    env["NUMEXPR_NUM_THREADS"] = str(max(1, args.cpu_threads))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    reference_bundle = find_latest_bundle(DEFAULT_LOG_DIR)
    print(json.dumps({
        "dataset_root": str(dataset_root),
        "hf_dataset": str(hf_dataset),
        "reference_bundle_before_training": str(reference_bundle) if reference_bundle else "",
        "max_steps": args.max_steps,
        "sample_size": args.sample_size,
        "train_last_n_blocks": args.train_last_n_blocks,
        "cpu_threads": args.cpu_threads,
    }, indent=2, ensure_ascii=False), flush=True)

    if args.rebuild_dataset or not (dataset_root / "metadata.jsonl").exists():
        run_step(
            "build 15s section dataset",
            [
                sys.executable,
                "scripts/build_avicii_section15_dataset.py",
                "--output-root",
                str(dataset_root),
                "--max-total-clips",
                str(args.max_total_clips),
                "--max-clips-per-source",
                str(args.max_clips_per_source),
                "--force",
            ],
            env,
        )
    else:
        print("SKIP build 15s section dataset: metadata.jsonl already exists", flush=True)

    if not field_files_complete(dataset_root, "latent_path") or not field_files_complete(dataset_root, "text_token_path"):
        run_step(
            "cache ACE latents and text tokens",
            [
                sys.executable,
                "scripts/cache_ace_assets.py",
                "--dataset-root",
                str(dataset_root),
                "--device",
                "cpu",
                "--batch-size",
                "1",
            ],
            env,
        )
    else:
        print("SKIP cache ACE assets: latent_path and text_token_path are complete", flush=True)

    if not field_files_complete(dataset_root, "control_path"):
        run_step(
            "build latent-aligned control assets",
            [
                sys.executable,
                "scripts/build_edm_control_assets.py",
                "--dataset-root",
                str(dataset_root),
            ],
            env,
        )
    else:
        print("SKIP build controls: control_path is complete", flush=True)

    run_step(
        "prepare ACE HF training dataset",
        [
            sys.executable,
            "scripts/prepare_ace_control_dataset.py",
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(hf_dataset),
            "--path-mode",
            "project-relative",
            "--min-quality",
            "4",
        ],
        env,
    )

    run_step(
        "train Avicii section15 v2 LoRA",
        [
            sys.executable,
            "scripts/train_avicii_local_lora.py",
            "--dataset-path",
            str(hf_dataset),
            "--exp-name",
            "avicii_section15_v2_cpu",
            "--max-steps",
            str(args.max_steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--every-n-train-steps",
            "40",
            "--sample-size",
            str(args.sample_size),
            "--cpu-threads",
            str(args.cpu_threads),
            "--train-last-n-blocks",
            str(args.train_last_n_blocks),
            "--learning-rate",
            str(args.learning_rate),
            "--auto-init-latest-lora",
        ],
        env,
    )

    candidate_bundle = find_latest_bundle(DEFAULT_LOG_DIR)
    if args.skip_generate:
        print("SKIP generation: --skip-generate set", flush=True)
        return
    if reference_bundle is None or candidate_bundle is None or candidate_bundle == reference_bundle:
        print("SKIP generation: reference/candidate bundle pair is unavailable", flush=True)
        return

    run_step(
        "generate same-seed triple comparison",
        [
            sys.executable,
            "scripts/generate_avicii_triple_compare.py",
            "--reference-adapter-bundle",
            str(reference_bundle),
            "--candidate-adapter-bundle",
            str(candidate_bundle),
            "--sections",
            "intro,breakdown,drop",
            "--duration",
            "15",
            "--seed",
            "42",
            "--infer-step",
            "30",
            "--lora-weight",
            "1.6",
        ],
        env,
    )

    print(json.dumps({
        "status": "complete",
        "reference_bundle": str(reference_bundle),
        "candidate_bundle": str(candidate_bundle),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
