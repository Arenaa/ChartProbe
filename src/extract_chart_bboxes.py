#!/usr/bin/env python3
"""Extract bar/wedge and value-label bounding boxes for bbox chart datasets.

Writes per-chart annotations and 2 Grounding questions per chart:
  - G-BBOX-ELEMENT  — geometry of the bar or pie segment
  - G-BBOX-VALUE    — geometry of the CSV value text on that element

Re-renders charts the same way as render_charts_with_value_labels.py so boxes
align with the saved PNGs.

Usage:
  python src/extract_chart_bboxes.py
  python src/extract_chart_bboxes.py --dataset bar_charts_100
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent


def _load_render_utils():
    path = SCRIPT_DIR / "render_charts_with_value_labels.py"
    spec = importlib.util.spec_from_file_location("chart_render_utils", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


RU = _load_render_utils()


def bbox_norm(bb, img_w: int, img_h: int) -> list[int]:
    if hasattr(bb, "x0"):
        x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
    else:
        x0, y0, x1, y1 = bb
    return [
        max(0, min(1000, round(1000 * x0 / img_w))),
        max(0, min(1000, round(1000 * (1 - y1 / img_h)))),
        max(0, min(1000, round(1000 * x1 / img_w))),
        max(0, min(1000, round(1000 * (1 - y0 / img_h)))),
    ]


def format_bbox(bbox: list[int]) -> str:
    return json.dumps(bbox)


def add_bar_labels_with_artists(ax, pairs):
    bars = [p for p in ax.patches if RU._is_bar(p)]
    if not bars:
        return []
    ordered = RU.order_bar_pairs(bars, pairs, ax)
    out = []
    n = min(len(bars), len(ordered))
    for patch, (label, display, num) in zip(bars[:n], ordered[:n]):
        x, y = patch.get_xy()
        w, h = patch.get_width(), patch.get_height()
        if abs(w) >= abs(h):
            text = ax.text(
                x + w + 0.01 * max(abs(ax.get_xlim()[1]), 1),
                y + h / 2,
                display,
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="black",
                clip_on=False,
            )
        else:
            text = ax.text(
                x + w / 2,
                y + h,
                display,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="black",
                clip_on=False,
            )
        out.append(
            {
                "label": label,
                "value": display,
                "value_numeric": num,
                "patch": patch,
                "text": text,
            }
        )
    return out


def add_pie_labels_with_artists(ax, pairs):
    wedges = [p for p in ax.patches if RU._is_wedge(p)]
    if not wedges:
        return []
    ordered = RU.order_pie_pairs(wedges, pairs)
    out = []
    n = min(len(wedges), len(ordered))
    for wedge, (label, display, num) in zip(wedges[:n], ordered[:n]):
        ang = (float(wedge.theta2) + float(wedge.theta1)) / 2.0
        rad = math.radians(ang)
        r = 0.65
        text = ax.text(
            r * math.cos(rad),
            r * math.sin(rad),
            display,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="black",
                alpha=0.45,
                edgecolor="none",
            ),
        )
        out.append(
            {
                "label": label,
                "value": display,
                "value_numeric": num,
                "patch": wedge,
                "text": text,
            }
        )
    return out


def extract_chart_annotation(
    *,
    chart_id: str,
    code: str,
    csv_text: str,
    chart_type: str,
    out_image: Path | None = None,
) -> dict:
    pairs = RU.load_csv_pairs(csv_text)
    if not pairs:
        raise ValueError("empty csv")

    fig, ax = RU.exec_chart_code(code)
    is_pie = chart_type.lower().startswith("pie")
    elements = (
        add_pie_labels_with_artists(ax, pairs)
        if is_pie
        else add_bar_labels_with_artists(ax, pairs)
    )
    if not elements:
        raise ValueError("no chart elements found")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf)
    img_w, img_h = img.size

    if out_image is not None:
        out_image.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_image)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_w, canvas_h = fig.canvas.get_width_height()
    sx = img_w / canvas_w
    sy = img_h / canvas_h

    def to_norm(bb):
        return bbox_norm(
            (bb.x0 * sx, bb.y0 * sy, bb.x1 * sx, bb.y1 * sy),
            img_w,
            img_h,
        )

    element_rows = []
    for el in elements:
        patch_bb = to_norm(el["patch"].get_window_extent(renderer))
        text_bb = to_norm(el["text"].get_window_extent(renderer))
        element_rows.append(
            {
                "label": el["label"],
                "value": el["value"],
                "value_numeric": el["value_numeric"],
                "element_bbox": patch_bb,
                "value_bbox": text_bb,
            }
        )

    plt.close(fig)
    element_kind = "wedge" if is_pie else "bar"
    return {
        "chart_id": chart_id,
        "chart_type": chart_type,
        "image_width": img_w,
        "image_height": img_h,
        "bbox_format": "xyxy_1000",
        "element_kind": element_kind,
        "elements": element_rows,
    }


def pick_largest(elements: list[dict]) -> dict:
    return max(elements, key=lambda e: float(e["value_numeric"]))


def build_questions(
    annotation: dict,
    *,
    complexity_bin: str | None,
    count_key: str,
    count_val: int | None,
) -> list[dict]:
    chart_id = annotation["chart_id"]
    kind = annotation["element_kind"]
    target = pick_largest(annotation["elements"])
    label = target["label"]
    element_word = "pie segment" if kind == "wedge" else "bar"

    base = {
        "chart_id": chart_id,
        "complexity_bin": complexity_bin,
        count_key: count_val,
        "stage": "Grounding",
        "answer_type": "bbox",
        "bbox_format": "xyxy_1000",
        "target_label": label,
        "verified": "auto",
        "ambiguity_flag": False,
    }

    return [
        {
            **base,
            "template_id": "G-BBOX-ELEMENT",
            "question": f"What is the bounding box of the {element_word} for {label}?",
            "answer": format_bbox(target["element_bbox"]),
            "bbox": target["element_bbox"],
            "bbox_target": "element",
        },
        {
            **base,
            "template_id": "G-BBOX-VALUE",
            "question": f"What is the bounding box of the value label '{target['value']}' for {label}?",
            "answer": format_bbox(target["value_bbox"]),
            "bbox": target["value_bbox"],
            "bbox_target": "value",
        },
    ]


def export_eval_bundle(
    questions: list[dict], *, dataset_dir: Path, images_dir: Path
) -> None:
    out_dir = dataset_dir / "eval_bundle_bbox"
    out_dir.mkdir(parents=True, exist_ok=True)
    link = out_dir / "images"
    if link.is_symlink():
        link.unlink()
    elif link.is_dir() and not any(link.iterdir()):
        link.rmdir()
    if not link.exists():
        link.symlink_to(images_dir.resolve())

    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as fout:
        for i, q in enumerate(questions):
            chart_id = q["chart_id"]
            row = {
                "index": i,
                "id": f"{chart_id}__{q['template_id']}",
                "chart_id": chart_id,
                "image_file": f"images/{chart_id}.png",
                "question": q["question"],
                "answer": q["answer"],
                "stage": q.get("stage"),
                "template_id": q.get("template_id"),
                "answer_type": q.get("answer_type"),
                "bbox_format": q.get("bbox_format"),
                "target_label": q.get("target_label"),
                "bbox_target": q.get("bbox_target"),
            }
            if "bar_count" in q:
                row["bar_count"] = q["bar_count"]
            if "wedge_count" in q:
                row["wedge_count"] = q["wedge_count"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_dataset(dataset_dir: Path) -> tuple[int, int]:
    meta_path = dataset_dir / "metadata.json"
    rows = json.loads(meta_path.read_text(encoding="utf-8"))
    chart_type = "Pie Chart" if "pie" in dataset_dir.name else "Bar Chart"
    count_key = "wedge_count" if chart_type == "Pie Chart" else "bar_count"

    ann_dir = dataset_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    all_questions: list[dict] = []
    ok = 0
    errors: list[str] = []

    for row in rows:
        chart_id = row["id"]
        out_image = dataset_dir / "images" / f"{chart_id}.png"
        out_bundle = dataset_dir / "eval_bundle" / "images" / f"{chart_id}.png"
        try:
            ann = extract_chart_annotation(
                chart_id=chart_id,
                code=row.get("code", ""),
                csv_text=row.get("csv", ""),
                chart_type=chart_type,
                out_image=out_image,
            )
            (ann_dir / f"{chart_id}.json").write_text(
                json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            out_bundle.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_image, out_bundle)

            row["bboxes_extracted"] = True
            row["annotation_path"] = f"{dataset_dir.name}/annotations/{chart_id}.json"
            row["num_bbox_elements"] = len(ann["elements"])

            qs = build_questions(
                ann,
                complexity_bin=row.get("complexity_bin"),
                count_key=count_key,
                count_val=row.get(count_key),
            )
            all_questions.extend(qs)
            ok += 1
        except Exception as exc:
            errors.append(f"{chart_id}: {exc}")

    meta_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    qdir = dataset_dir / "dataset"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "bbox_questions.json").write_text(
        json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    export_eval_bundle(all_questions, dataset_dir=dataset_dir, images_dir=dataset_dir / "images")

    meta_path = dataset_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta.update(
        {
            "bboxes_extracted": True,
            "bbox_questions_per_chart": 2,
            "total_bbox_questions": len(all_questions),
            "annotations_dir": f"{dataset_dir.name}/annotations",
            "bbox_questions_json": f"{dataset_dir.name}/dataset/bbox_questions.json",
            "eval_bundle_bbox": f"{dataset_dir.name}/eval_bundle_bbox/samples.jsonl",
        }
    )
    (dataset_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if errors:
        err_path = dataset_dir / "extract_bbox_errors.txt"
        err_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"  errors: {len(errors)} (see {err_path})")
    return ok, len(errors)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", action="append", default=[])
    args = ap.parse_args()

    datasets = args.dataset or [
        str(REPO / "data" / "bar_charts_100"),
        str(REPO / "data" / "pie_charts_100"),
    ]

    for ds in datasets:
        path = Path(ds).resolve()
        print(f"Extracting bboxes → {path}")
        ok, err = process_dataset(path)
        print(f"  done: {ok} charts, {err} errors, {ok * 2} questions")


if __name__ == "__main__":
    main()
