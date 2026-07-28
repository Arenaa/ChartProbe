#!/usr/bin/env python3
"""Export bar-chart QA rows to multimodal SFT JSONL.

Split modes:
  chart  — train/val by chart_id (default; avoids chart leakage)
  stage  — train/val by question stage (e.g. Perception+Grounding train, Reasoning val)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_stage(stage: str | None) -> str:
    return (stage or "").strip()


def split_by_chart(rows: list[dict], *, val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    chart_ids = sorted({r["chart_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(chart_ids)
    n_val = max(1, int(round(len(chart_ids) * val_ratio)))
    val_charts = set(chart_ids[:n_val])
    train = [r for r in rows if r["chart_id"] not in val_charts]
    val = [r for r in rows if r["chart_id"] in val_charts]
    return train, val


def split_by_stage(
    rows: list[dict],
    *,
    train_stages: set[str],
    val_stages: set[str],
) -> tuple[list[dict], list[dict]]:
    overlap = train_stages & val_stages
    if overlap:
        raise ValueError(f"train and val stages overlap: {sorted(overlap)}")
    train = [r for r in rows if normalize_stage(r.get("stage")) in train_stages]
    val = [r for r in rows if normalize_stage(r.get("stage")) in val_stages]
    if not train:
        raise ValueError(f"No train rows for stages {sorted(train_stages)}")
    if not val:
        raise ValueError(f"No val rows for stages {sorted(val_stages)}")
    return train, val


def write_jsonl(path: Path, rows: list[dict], *, images_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for i, q in enumerate(rows):
            chart_id = q["chart_id"]
            image_path = images_dir / f"{chart_id}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image for chart {chart_id}: {image_path}")
            row = {
                "index": i,
                "id": f"{chart_id}__{q.get('template_id', i)}",
                "chart_id": chart_id,
                "image": str(image_path.resolve()),
                "question": q["question"],
                "answer": q["answer"],
                "stage": q.get("stage"),
                "template_id": q.get("template_id"),
                "answer_type": q.get("answer_type"),
                "complexity_bin": q.get("complexity_bin"),
                "bar_count": q.get("bar_count"),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def stage_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(normalize_stage(r.get("stage")) for r in rows))


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=repo / "bar_charts_1000",
        help="Dataset root with dataset/questions.json and images/",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "data_other" / "chart_qa_sft" / "bar_charts_1000",
    )
    ap.add_argument(
        "--split-by",
        choices=("chart", "stage"),
        default="chart",
        help="chart = split by chart_id; stage = split by question stage",
    )
    ap.add_argument(
        "--train-stages",
        nargs="+",
        default=["Perception", "Grounding"],
        help="With --split-by stage: stages for train.jsonl",
    )
    ap.add_argument(
        "--val-stages",
        nargs="+",
        default=["Reasoning"],
        help="With --split-by stage: stages for val.jsonl",
    )
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Only for --split-by chart")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    questions_path = dataset_dir / "dataset" / "questions.json"
    images_dir = dataset_dir / "images"
    rows = load_questions(questions_path)

    if args.split_by == "chart":
        train, val = split_by_chart(rows, val_ratio=args.val_ratio, seed=args.seed)
        split_meta = {"split_by": "chart", "val_ratio": args.val_ratio, "seed": args.seed}
    else:
        train_stages = {normalize_stage(s) for s in args.train_stages}
        val_stages = {normalize_stage(s) for s in args.val_stages}
        train, val = split_by_stage(rows, train_stages=train_stages, val_stages=val_stages)
        split_meta = {
            "split_by": "stage",
            "train_stages": sorted(train_stages),
            "val_stages": sorted(val_stages),
        }

    out = args.out_dir.resolve()
    write_jsonl(out / "train.jsonl", train, images_dir=images_dir)
    write_jsonl(out / "val.jsonl", val, images_dir=images_dir)
    meta = {
        "source": str(dataset_dir),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_charts": len({r["chart_id"] for r in train}),
        "val_charts": len({r["chart_id"] for r in val}),
        "train_stage_counts": stage_counts(train),
        "val_stage_counts": stage_counts(val),
        **split_meta,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
