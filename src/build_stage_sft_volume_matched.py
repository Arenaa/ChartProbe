#!/usr/bin/env python3
"""Export volume-matched SFT splits at 2700 train / 300 val for multi-stage setups.

Single-stage setups (perception / grounding / reasoning) are already 2700/300.
This script builds confound-free counterparts for:

  - perception_grounding_volume_matched
      450 of 900 train charts × (Perc+Ground=6Q) = 2700 train
      50 of 100 val charts × 6Q = 300 val

  - all_volume_matched  (delegates to existing logic / also re-exported here)
      300 of 900 train charts × 9Q = 2700 train
      100 Q per stage from val charts = 300 val

Usage:
  python src/build_stage_sft_volume_matched.py
  python src/build_stage_sft_volume_matched.py --setup perception_grounding_volume_matched
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

STAGES_ALL = ("Perception", "Grounding", "Reasoning")
STAGES_PG = ("Perception", "Grounding")


def export_pg_volume_matched(
    rows: list[dict],
    *,
    train_chart_ids: list[str],
    val_chart_ids: list[str],
    images_dir: Path,
    out_dir: Path,
    seed: int,
    train_charts_n: int = 450,
    val_charts_n: int = 50,
) -> dict:
    rng = random.Random(seed)
    train_charts = set(rng.sample(sorted(train_chart_ids), train_charts_n))
    val_charts = set(rng.sample(sorted(val_chart_ids), val_charts_n))
    stage_set = set(STAGES_PG)

    train = [
        r
        for r in rows
        if r["chart_id"] in train_charts and normalize_stage(r.get("stage")) in stage_set
    ]
    val = [
        r
        for r in rows
        if r["chart_id"] in val_charts and normalize_stage(r.get("stage")) in stage_set
    ]
    expected_train = train_charts_n * len(STAGES_PG) * 3
    expected_val = val_charts_n * len(STAGES_PG) * 3
    if len(train) != expected_train:
        raise ValueError(f"Expected {expected_train} train rows, got {len(train)}")
    if len(val) != expected_val:
        raise ValueError(f"Expected {expected_val} val rows, got {len(val)}")

    rng.shuffle(train)
    rng.shuffle(val)
    info = {
        "setup": "perception_grounding_volume_matched",
        "volume_matched": True,
        "train_stages": list(STAGES_PG),
        "split_by": "chart",
        "n_train_charts": len(train_charts),
        "n_val_charts": len(val_charts),
        "seed": seed,
        "train_rows": len(train),
        "val_rows": len(val),
        "train_stage_counts": stage_counts(train),
        "val_stage_counts": stage_counts(val),
        "train_chart_ids": sorted(train_charts),
        "val_chart_ids": sorted(val_charts),
        "source": "bar_charts_1000",
        "note": "2700/300 to match single-stage and all_volume_matched",
    }
    write_jsonl(out_dir / "train.jsonl", train, images_dir=images_dir)
    write_jsonl(out_dir / "val.jsonl", val, images_dir=images_dir)
    (out_dir / "meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--setup",
        choices=(
            "perception_grounding_volume_matched",
            "all_volume_matched",
            "all",
        ),
        default="all",
        help="'all' exports both volume-matched multi-stage setups",
    )
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO / "data" / "bar_charts_1000",
    )
    ap.add_argument(
        "--split-manifest",
        type=Path,
        default=REPO
        / "data_other"
        / "chart_qa_sft"
        / "stage_experiments"
        / "split_manifest.json",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO / "data_other" / "chart_qa_sft" / "stage_experiments",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    rows = load_questions(dataset_dir / "dataset" / "questions.json")
    images_dir = dataset_dir / "images"
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_chart_ids = split_manifest["train_chart_ids"]
    val_chart_ids = split_manifest["val_chart_ids"]
    out_root = args.out_root.resolve()

    do_pg = args.setup in ("perception_grounding_volume_matched", "all")
    do_all = args.setup in ("all_volume_matched", "all")

    if do_pg:
        info = export_pg_volume_matched(
            rows,
            train_chart_ids=train_chart_ids,
            val_chart_ids=val_chart_ids,
            images_dir=images_dir,
            out_dir=out_root / "perception_grounding_volume_matched",
            seed=args.seed,
        )
        print(json.dumps(info, indent=2))

    if do_all:
        # Reuse existing exporter for all_volume_matched
        from build_stage_sft_volume_matched_all import subsample_volume_matched

        train, val, info = subsample_volume_matched(
            rows,
            train_chart_ids=train_chart_ids,
            val_chart_ids=val_chart_ids,
            train_charts_n=300,
            val_per_stage=100,
            seed=args.seed,
        )
        out_dir = out_root / "all_volume_matched"
        write_jsonl(out_dir / "train.jsonl", train, images_dir=images_dir)
        write_jsonl(out_dir / "val.jsonl", val, images_dir=images_dir)
        (out_dir / "meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(json.dumps(info, indent=2))

    # Sanity print for the already-matched single-stage setups
    print("\nVolume-matched SFT inventory (target 2700/300):")
    for setup in (
        "perception",
        "grounding",
        "reasoning",
        "perception_grounding_volume_matched",
        "all_volume_matched",
    ):
        d = out_root / setup
        tr = d / "train.jsonl"
        va = d / "val.jsonl"
        if tr.exists() and va.exists():
            nt = sum(1 for _ in tr.open())
            nv = sum(1 for _ in va.open())
            ok = "OK" if (nt, nv) == (2700, 300) else "MISMATCH"
            print(f"  {setup:40s} train={nt:4d} val={nv:3d}  {ok}")
        else:
            print(f"  {setup:40s} MISSING")


if __name__ == "__main__":
    main()
