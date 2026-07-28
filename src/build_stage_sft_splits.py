#!/usr/bin/env python3
"""Export bar_charts_1000 SFT splits for stage-specific finetuning experiments.

Fixed chart-level split: 900 train charts / 100 val charts (seed 42).
Five setups filter question stages:

  1. perception            — Perception only
  2. grounding             — Grounding only
  3. reasoning             — Reasoning only
  4. perception_grounding  — Perception + Grounding
  5. all                   — Perception + Grounding + Reasoning

Usage:
  python src/build_stage_sft_splits.py
  python src/build_stage_sft_splits.py --setup perception
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_sft_jsonl import (
    load_questions,
    normalize_stage,
    stage_counts,
    write_jsonl,
)

REPO = SCRIPT_DIR.parent

STAGE_SETUPS: dict[str, list[str]] = {
    "perception": ["Perception"],
    "grounding": ["Grounding"],
    "reasoning": ["Reasoning"],
    "perception_grounding": ["Perception", "Grounding"],
    "all": ["Perception", "Grounding", "Reasoning"],
}


def split_by_chart_fixed(
    rows: list[dict],
    *,
    n_val_charts: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    chart_ids = sorted({r["chart_id"] for r in rows})
    if n_val_charts >= len(chart_ids):
        raise ValueError(f"n_val_charts={n_val_charts} must be < {len(chart_ids)} charts")
    rng = random.Random(seed)
    shuffled = chart_ids.copy()
    rng.shuffle(shuffled)
    val_charts = set(shuffled[:n_val_charts])
    train_charts = set(shuffled[n_val_charts:])
    train = [r for r in rows if r["chart_id"] in train_charts]
    val = [r for r in rows if r["chart_id"] in val_charts]
    split_info = {
        "seed": seed,
        "n_train_charts": len(train_charts),
        "n_val_charts": len(val_charts),
        "train_chart_ids": sorted(train_charts),
        "val_chart_ids": sorted(val_charts),
    }
    return train, val, split_info


def export_setup(
    *,
    rows: list[dict],
    setup: str,
    stages: list[str],
    out_root: Path,
    images_dir: Path,
    n_val_charts: int,
    seed: int,
    split_info: dict | None = None,
) -> dict:
    stage_set = {normalize_stage(s) for s in stages}
    filtered = [r for r in rows if normalize_stage(r.get("stage")) in stage_set]
    if not filtered:
        raise ValueError(f"No rows for setup={setup} stages={sorted(stage_set)}")

    if split_info is None:
        train, val, split_info = split_by_chart_fixed(
            filtered, n_val_charts=n_val_charts, seed=seed
        )
    else:
        train_charts = set(split_info["train_chart_ids"])
        val_charts = set(split_info["val_chart_ids"])
        train = [r for r in filtered if r["chart_id"] in train_charts]
        val = [r for r in filtered if r["chart_id"] in val_charts]

    out_dir = out_root / setup
    write_jsonl(out_dir / "train.jsonl", train, images_dir=images_dir)
    write_jsonl(out_dir / "val.jsonl", val, images_dir=images_dir)

    meta = {
        "setup": setup,
        "train_stages": sorted(stage_set),
        "split_by": "chart",
        "n_train_charts": split_info["n_train_charts"],
        "n_val_charts": split_info["n_val_charts"],
        "seed": seed,
        "train_rows": len(train),
        "val_rows": len(val),
        "train_stage_counts": stage_counts(train),
        "val_stage_counts": stage_counts(val),
        "source": "bar_charts_1000",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO / "data" / "bar_charts_1000",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO / "data_other" / "chart_qa_sft" / "stage_experiments",
    )
    ap.add_argument("--n-val-charts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--setup",
        choices=sorted(STAGE_SETUPS),
        default=None,
        help="Export one setup only (default: all five)",
    )
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    rows = load_questions(dataset_dir / "dataset" / "questions.json")
    images_dir = dataset_dir / "images"

    setups = [args.setup] if args.setup else sorted(STAGE_SETUPS)
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Use full-row split once so all setups share the same chart IDs.
    _, _, shared_split = split_by_chart_fixed(
        rows, n_val_charts=args.n_val_charts, seed=args.seed
    )
    (out_root / "split_manifest.json").write_text(
        json.dumps(shared_split, indent=2), encoding="utf-8"
    )

    summaries = []
    for name in setups:
        meta = export_setup(
            rows=rows,
            setup=name,
            stages=STAGE_SETUPS[name],
            out_root=out_root,
            images_dir=images_dir,
            n_val_charts=args.n_val_charts,
            seed=args.seed,
            split_info=shared_split,
        )
        summaries.append(meta)
        print(json.dumps(meta, indent=2))

    manifest = {
        "setups": setups,
        "n_val_charts": args.n_val_charts,
        "seed": args.seed,
        "summaries": summaries,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
