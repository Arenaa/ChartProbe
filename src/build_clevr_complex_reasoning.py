#!/usr/bin/env python3
"""
Export CLEVR complex-reasoning eval bundle from original val questions.

One original CLEVR val question per scene (the bbox-linked val question).
Output: {dataset}/complex_reasoning/{samples.jsonl, manifest.json}
Images are referenced via ../images/ (no copy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=REPO / "data" / "clevr_100")
    ap.add_argument(
        "--val-questions",
        type=Path,
        default=REPO / "CLEVR_val_questions_new.json",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    out_dir = (args.out_dir or dataset_dir / "complex_reasoning").resolve()
    val_questions_path = args.val_questions.resolve()
    questions_json = dataset_dir / "questions.json"

    val_data = json.loads(val_questions_path.read_text(encoding="utf-8"))
    by_qidx = {q["question_index"]: q for q in val_data["questions"]}

    bbox_rows = [
        q
        for q in json.loads(questions_json.read_text(encoding="utf-8"))
        if q.get("template_id") == "G-BBOX-ELEMENT" and q.get("val_question_index") is not None
    ]
    bbox_rows.sort(key=lambda r: r["scene_id"])
    if not bbox_rows:
        raise SystemExit("No bbox-linked scenes found in questions.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, row in enumerate(bbox_rows):
        qidx = int(row["val_question_index"])
        orig = by_qidx.get(qidx)
        if orig is None:
            raise SystemExit(f"Missing val question index {qidx} for {row['scene_id']}")
        gold = str(orig["answer"]).strip()
        question = str(orig["question"]).strip()
        full_reasoning = json.dumps(
            {
                "think": "",
                "answer": gold,
                "raw": f"<answer>\n{gold}\n</answer>",
            },
            ensure_ascii=False,
        )
        rel_image = f"../images/{row['image_filename']}"
        img_path = (out_dir / rel_image).resolve()
        if not img_path.is_file():
            raise SystemExit(f"Missing image: {img_path}")
        rows.append(
            {
                "run_index": i,
                "id": f"{row['scene_id']}__q{qidx}",
                "scene_id": row["scene_id"],
                "complex_question": question,
                "full_reasoning": full_reasoning,
                "image_file": rel_image,
                "source_question_index": qidx,
                "chart_type": "clevr",
                "task": "complex_reasoning_qa",
            }
        )

    jsonl_path = out_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for rec in rows:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": f"clevr_complex_reasoning_{len(rows)}",
        "source": "CLEVR_val_questions_new.json (bbox-linked question per scene)",
        "num_samples": len(rows),
        "bundle_dir": str(out_dir.relative_to(REPO)),
        "samples_jsonl": str(jsonl_path.relative_to(REPO)),
        "images": "data/clevr_100/images",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
