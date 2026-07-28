#!/usr/bin/env python3
"""Export volume-matched ``all`` SFT: 2700 train / 300 val (900 per stage).

Uses the same chart-level 900/100 split as ``build_stage_sft_splits.py`` (seed 42).
Train: 300 of the 900 train charts × 3 questions per stage = 900 × 3 stages = 2700.
Val: 100 questions per stage sampled from the 100 val charts = 300 total.

This matches single-stage setup row counts (2700/300) so ``all`` is confound-free vs
``perception`` / ``grounding`` / ``reasoning``.

Usage:
  python src/build_stage_sft_volume_matched_all.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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

STAGES = ("Perception", "Grounding", "Reasoning")
SETUP = "all_volume_matched"
TRAIN_CHARTS = 300
VAL_PER_STAGE = 100


def subsample_volume_matched(
    rows: list[dict],
    *,
    train_chart_ids: list[str],
    val_chart_ids: list[str],
    train_charts_n: int,
    val_per_stage: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    train_charts = set(rng.sample(sorted(train_chart_ids), train_charts_n))
    train_pool = [r for r in rows if r["chart_id"] in train_charts]
    train_by_stage: dict[str, list[dict]] = {s: [] for s in STAGES}
    for r in train_pool:
        stage = normalize_stage(r.get("stage"))
        if stage in train_by_stage:
            train_by_stage[stage].append(r)

    train: list[dict] = []
    for stage in STAGES:
        stage_rows = train_by_stage[stage]
        if len(stage_rows) != train_charts_n * 3:
            raise ValueError(
                f"Expected {train_charts_n * 3} {stage} train rows, got {len(stage_rows)}"
            )
        train.extend(stage_rows)

    val_pool = [r for r in rows if r["chart_id"] in set(val_chart_ids)]
    val_by_stage: dict[str, list[dict]] = {s: [] for s in STAGES}
    for r in val_pool:
        stage = normalize_stage(r.get("stage"))
        if stage in val_by_stage:
            val_by_stage[stage].append(r)

    val: list[dict] = []
    for stage in STAGES:
        stage_rows = val_by_stage[stage]
        if len(stage_rows) < val_per_stage:
            raise ValueError(
                f"Need {val_per_stage} {stage} val rows, only {len(stage_rows)} available"
            )
        val.extend(rng.sample(stage_rows, val_per_stage))

    rng.shuffle(train)
    rng.shuffle(val)

    info = {
        "setup": SETUP,
        "volume_matched": True,
        "train_stages": list(STAGES),
        "split_by": "chart",
        "n_train_charts": len(train_charts),
        "n_val_charts": len(val_chart_ids),
        "train_charts_sampled_from": len(train_chart_ids),
        "val_per_stage": val_per_stage,
        "seed": seed,
        "train_rows": len(train),
        "val_rows": len(val),
        "train_stage_counts": stage_counts(train),
        "val_stage_counts": stage_counts(val),
        "train_chart_ids": sorted(train_charts),
        "val_chart_ids": sorted(val_chart_ids),
        "source": "bar_charts_1000",
    }
    return train, val, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO / "data" / "bar_charts_1000",
    )
    ap.add_argument(
        "--split-manifest",
        type=Path,
        default=REPO / "data_other" / "chart_qa_sft" / "stage_experiments" / "split_manifest.json",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO / "data_other" / "chart_qa_sft" / "stage_experiments",
    )
    ap.add_argument("--train-charts", type=int, default=TRAIN_CHARTS)
    ap.add_argument("--val-per-stage", type=int, default=VAL_PER_STAGE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    rows = load_questions(dataset_dir / "dataset" / "questions.json")
    images_dir = dataset_dir / "images"

    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_chart_ids = split_manifest["train_chart_ids"]
    val_chart_ids = split_manifest["val_chart_ids"]

    train, val, info = subsample_volume_matched(
        rows,
        train_chart_ids=train_chart_ids,
        val_chart_ids=val_chart_ids,
        train_charts_n=args.train_charts,
        val_per_stage=args.val_per_stage,
        seed=args.seed,
    )

    out_dir = args.out_root.resolve() / SETUP
    write_jsonl(out_dir / "train.jsonl", train, images_dir=images_dir)
    write_jsonl(out_dir / "val.jsonl", val, images_dir=images_dir)
    (out_dir / "meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
