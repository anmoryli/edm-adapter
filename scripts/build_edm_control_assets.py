"""Create latent-aligned EDM control curves and update dataset metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.edm_control.control_curves import (  # noqa: E402
    ControlCurveConfig,
    build_control_curve,
    infer_frame_count,
    save_schema,
)
from src.edm_control.metadata_utils import read_jsonl, to_posix, write_csv, write_jsonl  # noqa: E402


def build_assets(
    dataset_root: Path,
    metadata_name: str = "metadata.jsonl",
    overwrite: bool = False,
) -> dict:
    metadata_path = dataset_root / metadata_name
    rows = read_jsonl(metadata_path)
    controls_dir = dataset_root / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)

    base_config = ControlCurveConfig()
    save_schema(dataset_root / "controls" / "schema.json", base_config)

    updated_rows: list[dict] = []
    created = 0
    reused = 0
    for row in rows:
        clip_id = row["clip_id"]
        frame_count = infer_frame_count(row, base_config.frame_count)
        config = ControlCurveConfig(frame_count=frame_count)
        control_path = controls_dir / f"{clip_id}.pt"
        if overwrite or not control_path.exists():
            curve, meta = build_control_curve(row, config)
            torch.save(
                {
                    "clip_id": clip_id,
                    "control_type": "edm_latent_aligned_control_curve_v1",
                    "control": curve,
                    "metadata": meta,
                },
                control_path,
            )
            created += 1
        else:
            reused += 1

        rel = control_path.relative_to(dataset_root)
        row["control_path"] = to_posix(rel)
        row["control_config"] = {
            "type": "edm_latent_aligned_control_curve_v1",
            "feature_dim": config.feature_dim,
            "frame_count": frame_count,
            "schema_path": "controls/schema.json",
        }
        updated_rows.append(row)

    write_jsonl(metadata_path, updated_rows)
    write_csv(dataset_root / "metadata.csv", updated_rows)

    by_id = {row["clip_id"]: row for row in updated_rows}
    split_counts: dict[str, int] = {}
    for split_name in ["train", "val", "test"]:
        split_path = dataset_root / "splits" / f"{split_name}.jsonl"
        if not split_path.exists():
            continue
        split_rows = read_jsonl(split_path)
        new_split_rows = [by_id.get(row.get("clip_id"), row) for row in split_rows]
        write_jsonl(split_path, new_split_rows)
        split_counts[split_name] = len(new_split_rows)

    report = {
        "metadata_rows": len(updated_rows),
        "created_controls": created,
        "reused_controls": reused,
        "feature_dim": base_config.feature_dim,
        "schema_path": "dataset/controls/schema.json",
        "split_counts": split_counts,
    }
    report_path = dataset_root / "reports" / "control_assets_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# EDM Control Assets Report\n\n"
        f"- Metadata rows: {report['metadata_rows']}\n"
        f"- Created controls: {report['created_controls']}\n"
        f"- Reused controls: {report['reused_controls']}\n"
        f"- Feature dimension: {report['feature_dim']}\n"
        f"- Schema: `{report['schema_path']}`\n"
        f"- Split counts: `{json.dumps(split_counts, ensure_ascii=False)}`\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--metadata-name", default="metadata.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = build_assets(Path(args.dataset_root), args.metadata_name, args.overwrite)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
