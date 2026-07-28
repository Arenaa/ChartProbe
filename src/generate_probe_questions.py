#!/usr/bin/env python3
"""Rebuild test-set questions with a fixed question structure per sample.

Bar / pie charts (9 per sample):
  - Perception: 3 (count, height, trend — 1 template each)
  - Grounding:  3 (bbox, value_to_label, value_text — 1 each)
  - Reasoning:  3 (rank, arith, multi — 1 each)

CLEVR (9 per sample):
  - Perception: 3, Grounding: 3, Reasoning: 3

Datasets:
  - data/bar_charts_100
  - data/bar_charts_1000
  - data/pie_charts_100
  - data/clevr_100

Usage:
  python src/generate_probe_questions.py --dataset bar_charts_500
  python src/generate_probe_questions.py --dataset bar_charts_1000
  python src/generate_probe_questions.py --dataset pie_charts_500 --templates my_templates.json

Question text comes from src/question_templates.json (override with --templates).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clevr_bbox_utils import (
    build_val_bbox_targets_by_image,
    erobic_xywh_to_xyxy_1000,
    load_val_bbox_resources,
)
from templates import format_question, get_template, load_templates

REPO = SCRIPT_DIR.parent

VALUE_COL_HINTS = (
    "percentage", "share", "value", "amount", "total", "count", "balance",
    "production", "emission", "disbursement", "expenditure", "ratio", "rate",
    "avg", "mean", "sum", "oda", "earnings", "expenditure",
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def parse_value(raw: str) -> float:
    s = (raw or "").strip().replace(",", "")
    s = s.rstrip("%").strip()
    return float(s)


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    rounded_1 = round(value, 1)
    if abs(value - rounded_1) < 1e-9:
        return f"{rounded_1:.1f}"
    return f"{round(value, 2):.2f}"


def plural_label(name: str) -> str:
    name = (name or "Category").strip()
    return name if name.endswith("s") else f"{name}s"


def _column_tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t}


def detect_columns(fieldnames: list[str], sample_row: dict[str, str]) -> tuple[str, str]:
    value_col = None
    for f in fieldnames:
        if _column_tokens(f) & set(VALUE_COL_HINTS):
            value_col = f
            break
    if value_col is None:
        parseable = []
        for f in fieldnames:
            try:
                parse_value(str(sample_row.get(f, "")))
                parseable.append(f)
            except ValueError:
                continue
        if len(parseable) == 1:
            value_col = parseable[0]
        elif len(parseable) > 1:
            value_col = parseable[-1]
        elif len(fieldnames) > 1:
            value_col = fieldnames[-1]
    if value_col is None:
        raise ValueError(f"Cannot detect value column in {fieldnames}")
    label_col = next((f for f in fieldnames if f != value_col), fieldnames[0])
    return label_col, value_col


def load_chart_table(csv_text: str) -> tuple[str, str, list[tuple[str, float]]]:
    rows = list(csv.DictReader(io.StringIO((csv_text or "").strip())))
    if not rows:
        raise ValueError("empty csv")
    fieldnames = list(rows[0].keys())
    label_col, value_col = detect_columns(fieldnames, rows[0])
    pairs: list[tuple[str, float]] = []
    for row in rows:
        label = str(row.get(label_col, "")).strip()
        if not label:
            continue
        pairs.append((label, parse_value(str(row.get(value_col, "")))))
    if not pairs:
        raise ValueError("no data rows")
    return label_col, value_col, pairs


def bar_complexity_bin(n: int) -> str:
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def pie_complexity_bin(n: int | None) -> str:
    if n is None or n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def clevr_complexity_bin(n: int) -> str:
    if n <= 5:
        return "3-5"
    if n <= 8:
        return "6-8"
    return "9+"


def format_bbox(bbox: list[int]) -> str:
    return json.dumps(bbox)


def normalize_chart_meta(meta: dict) -> dict:
    out = dict(meta)
    chart_id = meta.get("chart_id") or meta.get("id")
    if not chart_id:
        raise ValueError("chart meta missing id/chart_id")
    out["chart_id"] = chart_id
    return out


def build_csv_fallback_annotation(
    chart_id: str,
    pairs: list[tuple[str, float]],
) -> dict:
    """CSV-only annotation for charts without matplotlib bbox extraction (e.g. plotly)."""
    return {
        "chart_id": chart_id,
        "bbox_format": "xyxy_1000",
        "element_kind": "bar",
        "bbox_extractable": False,
        "elements": [
            {
                "label": lbl,
                "value": format_number(v),
                "value_numeric": v,
                "element_bbox": [0, 0, 0, 0],
            }
            for lbl, v in pairs
        ],
    }


def ordered_chart_elements(annotation: dict) -> list[dict]:
    elements = list(annotation.get("elements") or [])
    if not elements:
        return elements
    b = elements[0]["element_bbox"]
    w = b[2] - b[0]
    h = b[3] - b[1]
    vertical_bars = w < h
    if vertical_bars:
        return sorted(elements, key=lambda e: (e["element_bbox"][0] + e["element_bbox"][2]) / 2)
    return sorted(elements, key=lambda e: (e["element_bbox"][1] + e["element_bbox"][3]) / 2)


def ordered_wedge_elements(annotation: dict) -> list[dict]:
    elements = list(annotation.get("elements") or [])
    if len(elements) <= 1:
        return elements
    centers = []
    for e in elements:
        b = e["element_bbox"]
        centers.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)

    def angle_key(e: dict) -> float:
        b = e["element_bbox"]
        x = (b[0] + b[2]) / 2 - cx
        y = (b[1] + b[3]) / 2 - cy
        return math.atan2(y, x)

    return sorted(elements, key=angle_key)


def trend_label(values: list[float]) -> str:
    if len(values) < 2:
        return "Neither consistently increasing nor decreasing"
    inc = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    dec = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    if inc and not dec:
        return "Consistently increasing"
    if dec and not inc:
        return "Consistently decreasing"
    return "Neither consistently increasing nor decreasing"


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


# ---------------------------------------------------------------------------
# Chart (bar / pie) question builders
# ---------------------------------------------------------------------------


def _chart_base_meta(
    chart_id: str,
    *,
    chart_type: str,
    count_key: str,
    count_val: int,
    complexity_bin: str,
) -> dict:
    return {
        "chart_id": chart_id,
        "complexity_bin": complexity_bin,
        count_key: count_val,
        "verified": "auto",
        "ambiguity_flag": False,
    }


def build_chart_perception(
    *,
    chart_type: str,
    label_col: str,
    pairs: list[tuple[str, float]],
    ordered: list[dict],
    base: dict,
    specs: dict,
) -> list[dict]:
    values = [v for _, v in pairs]
    mean = sum(values) / len(values)
    largest_elem = max(ordered, key=lambda e: (e["value_numeric"], e["label"]))
    min_val = min(values)
    max_val = largest_elem["value_numeric"]
    ordered_vals = [e["value_numeric"] for e in ordered]
    family = "pie" if chart_type == "pie" else "bar"

    out = []
    for tid, answer, answer_type in (
        ("P-COUNT-ABOVE-AVG", str(sum(1 for v in values if v > mean)), "count"),
        ("P-HEIGHT-2X", yes_no(max_val > 2 * min_val), "binary"),
        ("P-TREND-PATTERN", trend_label(ordered_vals), "open"),
    ):
        tmpl = get_template(specs, family, tid)
        out.append(
            {
                **base,
                "stage": tmpl["stage"],
                "template_id": tid,
                "question": format_question(tmpl),
                "answer": answer,
                "answer_type": tmpl.get("answer_type", answer_type),
            }
        )
    return out


def build_chart_grounding(
    *,
    chart_type: str,
    label_col: str,
    value_col: str,
    pairs: list[tuple[str, float]],
    ordered: list[dict],
    annotation: dict,
    base: dict,
    specs: dict,
) -> list[dict]:
    family = "pie" if chart_type == "pie" else "bar"
    left = ordered[0]
    largest = max(ordered, key=lambda e: (e["value_numeric"], e["label"]))
    second = sorted(ordered, key=lambda e: (-e["value_numeric"], e["label"]))[1]

    bbox_tmpl = get_template(specs, family, "G-BBOX-ELEMENT")
    bbox_q = {
        **base,
        "stage": bbox_tmpl["stage"],
        "template_id": "G-BBOX-ELEMENT",
        "question": format_question(bbox_tmpl, label=largest["label"]),
        "answer": format_bbox(largest["element_bbox"]),
        "answer_type": bbox_tmpl.get("answer_type", "bbox"),
        "bbox_format": "xyxy_1000",
        "bbox": largest["element_bbox"],
        "bbox_target": "element",
        "target_label": largest["label"],
    }
    if annotation.get("bbox_extractable") is False:
        bbox_q["bbox_extractable"] = False

    vt_id = "G-VT-LEFTMOST" if chart_type == "bar" else "G-VT-LARGEST"
    vt_tmpl = get_template(specs, family, vt_id)
    vt_q = {
        **base,
        "stage": vt_tmpl["stage"],
        "template_id": vt_id,
        "question": format_question(vt_tmpl, label_col=label_col),
        "answer": left["label"] if chart_type == "bar" else largest["label"],
        "answer_type": vt_tmpl.get("answer_type", "label"),
    }

    tv_tmpl = get_template(specs, family, "G-TV-VALUE-B")
    tv_q = {
        **base,
        "stage": tv_tmpl["stage"],
        "template_id": "G-TV-VALUE-B",
        "question": format_question(tv_tmpl, value_col=value_col, label=second["label"]),
        "answer": second["value"],
        "answer_type": tv_tmpl.get("answer_type", "value"),
    }
    return [bbox_q, vt_q, tv_q]


def build_chart_reasoning(
    *,
    chart_type: str,
    label_col: str,
    value_col: str,
    pairs: list[tuple[str, float]],
    base: dict,
    specs: dict,
) -> list[dict]:
    family = "pie" if chart_type == "pie" else "bar"
    values = [v for _, v in pairs]
    mean = sum(values) / len(values)
    ranked = sorted(pairs, key=lambda x: (-x[1], x[0]))
    top3 = [lbl for lbl, _ in ranked[: min(3, len(ranked))]]
    top_label, top_val = ranked[0]
    label_col_plural = plural_label(label_col)

    out = []
    for tid, answer, slots in (
        ("R-RANK-TOP3", ", ".join(top3), {"label_col_plural": label_col_plural, "value_col": value_col}),
        ("R-ARITH-TOTAL", format_number(sum(values)), {"value_col": value_col}),
        ("R-MULTI-DIFF-MEAN", format_number(top_val - mean), {"top_label": top_label, "value_col": value_col}),
    ):
        tmpl = get_template(specs, family, tid)
        out.append(
            {
                **base,
                "stage": tmpl["stage"],
                "template_id": tid,
                "question": format_question(tmpl, **slots),
                "answer": answer,
                "answer_type": tmpl.get("answer_type", "open"),
            }
        )
    return out


def build_chart_questions(
    meta: dict, annotation: dict, *, chart_type: str, specs: dict
) -> list[dict]:
    chart_id = meta["chart_id"]
    label_col, value_col, pairs = load_chart_table(meta.get("csv", ""))
    count_key = "wedge_count" if chart_type == "pie" else "bar_count"
    count_val = meta.get(count_key) or len(pairs)
    cbin = pie_complexity_bin(count_val) if chart_type == "pie" else bar_complexity_bin(count_val)
    base = _chart_base_meta(
        chart_id,
        chart_type=chart_type,
        count_key=count_key,
        count_val=count_val,
        complexity_bin=cbin,
    )
    ordered = ordered_wedge_elements(annotation) if chart_type == "pie" else ordered_chart_elements(annotation)
    if not ordered:
        ordered = [
            {"label": lbl, "value": format_number(v), "value_numeric": v, "element_bbox": [0, 0, 0, 0]}
            for lbl, v in pairs
        ]
    perception = build_chart_perception(
        chart_type=chart_type,
        label_col=label_col,
        pairs=pairs,
        ordered=ordered,
        base=base,
        specs=specs,
    )
    grounding = build_chart_grounding(
        chart_type=chart_type,
        label_col=label_col,
        value_col=value_col,
        pairs=pairs,
        ordered=ordered,
        annotation=annotation,
        base=base,
        specs=specs,
    )
    reasoning = build_chart_reasoning(
        chart_type=chart_type,
        label_col=label_col,
        value_col=value_col,
        pairs=pairs,
        base=base,
        specs=specs,
    )
    return perception + grounding + reasoning


# ---------------------------------------------------------------------------
# CLEVR question builders
# ---------------------------------------------------------------------------


def clevr_object_phrase(obj: dict) -> str:
    return f"{obj['color']} {obj['shape']}"


def pick_clevr_anchor(objects: list[dict]) -> dict:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for o in objects:
        counts[(o["color"], o["shape"])] += 1
    for o in sorted(objects, key=lambda x: x["pixel_coords"][0]):
        if counts[(o["color"], o["shape"])] == 1:
            return o
    return sorted(objects, key=lambda x: x["pixel_coords"][0])[0]


def pick_clevr_second_anchor(objects: list[dict], anchor: dict) -> dict:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for o in objects:
        counts[(o["color"], o["shape"])] += 1
    for o in sorted(objects, key=lambda x: x["pixel_coords"][0]):
        if o is anchor:
            continue
        if counts[(o["color"], o["shape"])] == 1:
            return o
    for o in sorted(objects, key=lambda x: x["pixel_coords"][0]):
        if o is not anchor:
            return o
    return anchor


def clevr_left_right_count(objects: list[dict], anchor: dict, side: str) -> int:
    ax = anchor["pixel_coords"][0]
    if side == "left":
        return sum(1 for o in objects if o["pixel_coords"][0] < ax and o is not anchor)
    return sum(1 for o in objects if o["pixel_coords"][0] > ax and o is not anchor)


def clevr_chain_material(objects: list[dict], anchor: dict, side: str) -> str:
    ax = anchor["pixel_coords"][0]
    candidates = [
        o for o in objects
        if o is not anchor and (o["pixel_coords"][0] < ax if side == "left" else o["pixel_coords"][0] > ax)
    ]
    if not candidates:
        return "none"
    if side == "left":
        target = max(candidates, key=lambda o: o["pixel_coords"][0])
    else:
        target = min(candidates, key=lambda o: o["pixel_coords"][0])
    return target["material"]


def clevr_binary_yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def clevr_material_count_answer(metal_n: int, rubber_n: int) -> str:
    if metal_n > rubber_n:
        return "metal"
    if rubber_n > metal_n:
        return "rubber"
    return "equal"


def clevr_spatial_half_answer(left_n: int, right_n: int) -> str:
    if left_n > right_n:
        return "left"
    if right_n > left_n:
        return "right"
    return "equal"


def build_clevr_color_binary_questions(
    objects: list[dict],
    *,
    anchor: dict,
    shape_pick: str,
    base: dict,
) -> list[dict]:
    """P-COLOR-BINARY variants from the updated CLEVR template spec."""
    color_any = anchor["color"]
    shape_any = anchor["shape"]
    any_exists = any(
        o["color"] == color_any and o["shape"] == shape_any for o in objects
    )

    # Pick a shape/color pair for the "Is the {shape} {color}?" template.
    shape_is = shape_pick
    shape_objs = [o for o in objects if o["shape"] == shape_is]
    test_color = shape_objs[0]["color"] if shape_objs else anchor["color"]
    is_match = any(o["shape"] == shape_is and o["color"] == test_color for o in objects)

    return [
        {
            **base,
            "stage": "Perception",
            "template_id": "P-COLOR-BINARY-IS",
            "question": f"Is the {shape_is} {test_color}?",
            "answer": clevr_binary_yes_no(is_match),
            "answer_type": "binary",
        },
        {
            **base,
            "stage": "Perception",
            "template_id": "P-COLOR-BINARY-ANY",
            "question": f"Is there any {color_any} {shape_any}?",
            "answer": clevr_binary_yes_no(any_exists),
            "answer_type": "binary",
        },
    ]


def build_clevr_questions(
    scene: dict,
    *,
    val_bbox_target: dict,
    specs: dict,
) -> list[dict]:
    image_filename = scene["image_filename"]
    scene_id = image_filename.replace(".png", "")
    objects = list(scene["objects"])
    n = len(objects)
    cbin = clevr_complexity_bin(n)
    base = {
        "scene_id": scene_id,
        "image_filename": image_filename,
        "complexity_bin": cbin,
        "n_objects": n,
        "verified": "auto",
        "ambiguity_flag": False,
    }
    anchor = pick_clevr_anchor(objects)
    anchor_b = pick_clevr_second_anchor(objects, anchor)
    anchor_phrase = clevr_object_phrase(anchor)
    anchor_b_phrase = clevr_object_phrase(anchor_b)

    bbox_label = val_bbox_target["target_label"]
    bbox_idx = val_bbox_target["object_index"]
    bbox = erobic_xywh_to_xyxy_1000(*val_bbox_target["xywh"])

    large_n = sum(1 for o in objects if o["size"] == "large")
    small_n = sum(1 for o in objects if o["size"] == "small")
    metal_n = sum(1 for o in objects if o["material"] == "metal")
    rubber_n = sum(1 for o in objects if o["material"] == "rubber")

    shape_counts = {s: sum(1 for o in objects if o["shape"] == s) for s in ("cube", "sphere", "cylinder")}
    shape_pick = max(shape_counts, key=lambda s: (shape_counts[s], s))
    shape_tid = f"P-COUNT-{shape_pick.upper()}"

    mat_answer = clevr_material_count_answer(metal_n, rubber_n)

    def _q(tid: str, answer: str, **slots: object) -> dict:
        tmpl = get_template(specs, "clevr", tid)
        return {
            **base,
            "stage": tmpl["stage"],
            "template_id": tid,
            "question": format_question(tmpl, **slots),
            "answer": answer,
            "answer_type": tmpl.get("answer_type", "open"),
        }

    perception = [
        _q("P-COUNT-TOTAL", str(n)),
        _q("P-COUNT-LARGE", str(large_n)),
        _q(shape_tid, str(shape_counts[shape_pick]), shape=shape_pick),
    ]

    bbox_tmpl = get_template(specs, "clevr", "G-BBOX-ELEMENT")
    grounding = [
        {
            **_q("G-BBOX-ELEMENT", format_bbox(bbox), target=bbox_label),
            "answer_type": bbox_tmpl.get("answer_type", "bbox"),
            "bbox_format": "xyxy_1000",
            "bbox": bbox,
            "bbox_source": val_bbox_target["bbox_source"],
            "bbox_target": "object",
            "target_label": bbox_label,
            "object_index": bbox_idx,
            "val_question_index": val_bbox_target["val_question_index"],
            "val_question": val_bbox_target["val_question"],
            "val_n_grounded_objects": val_bbox_target["n_grounded_objects"],
        },
        _q("G-ATTR-COLOR", anchor["color"], anchor=anchor_phrase),
        _q(
            "G-SPATIAL-LR",
            "left" if anchor_b["pixel_coords"][0] < anchor["pixel_coords"][0] else "right",
            obj_a=anchor_b_phrase,
            obj_b=anchor_phrase,
        ),
    ]

    reasoning = [
        _q("R-COMPARE-LARGE-SMALL", clevr_binary_yes_no(large_n > small_n)),
        _q("R-MATERIAL-COUNT", mat_answer),
        _q(
            "R-SAME-MATERIAL",
            clevr_binary_yes_no(anchor["material"] == anchor_b["material"]),
            anchor_a=anchor_phrase,
            anchor_b=anchor_b_phrase,
        ),
    ]
    return perception + grounding + reasoning


# ---------------------------------------------------------------------------
# Dataset processors + export
# ---------------------------------------------------------------------------


def export_eval_bundle(
    questions: list[dict],
    *,
    dataset_dir: Path,
    images_dir: Path,
    out_dir: Path | None = None,
    id_key: str = "chart_id",
) -> int:
    bundle_dir = out_dir or (dataset_dir / "eval_bundle")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    link = bundle_dir / "images"
    if link.is_symlink():
        link.unlink()
    elif link.is_dir() and not any(link.iterdir()):
        link.rmdir()
    if not link.exists():
        link.symlink_to(images_dir.resolve())

    out_jsonl = bundle_dir / "samples.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for i, q in enumerate(questions):
            sample_id = q.get(id_key) or q.get("scene_id")
            template_id = q.get("template_id", "")
            image_name = q.get("image_filename") or f"{sample_id}.png"
            row = {
                "index": i,
                "id": f"{sample_id}__{template_id}" if template_id else sample_id,
                "chart_id": sample_id,
                "scene_id": q.get("scene_id", sample_id),
                "image_file": f"images/{image_name}",
                "question": q["question"],
                "answer": q["answer"],
                "stage": q.get("stage"),
                "template_id": template_id,
                "answer_type": q.get("answer_type"),
                "complexity_bin": q.get("complexity_bin"),
            }
            for extra in (
                "bar_count", "wedge_count", "n_objects",
                "bbox_format", "bbox_target", "target_label",
                "val_question_index", "val_question", "val_n_grounded_objects",
            ):
                if extra in q:
                    row[extra] = q[extra]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(questions)


def process_chart_dataset(
    dataset_dir: Path,
    *,
    chart_type: str,
    specs: dict,
    allow_fallback_annotations: bool = False,
    fail_on_error: bool = True,
) -> dict:
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    all_questions: list[dict] = []
    errors: list[str] = []
    with_bbox = 0
    with_fallback = 0

    for raw_meta in metadata:
        meta = normalize_chart_meta(raw_meta)
        chart_id = meta["chart_id"]
        ann_path = dataset_dir / "annotations" / f"{chart_id}.json"
        try:
            if ann_path.is_file():
                annotation = json.loads(ann_path.read_text(encoding="utf-8"))
                with_bbox += 1
            elif allow_fallback_annotations:
                _, _, pairs = load_chart_table(meta.get("csv", ""))
                annotation = build_csv_fallback_annotation(chart_id, pairs)
                with_fallback += 1
            else:
                raise FileNotFoundError(f"missing annotation {ann_path}")
            all_questions.extend(
                build_chart_questions(meta, annotation, chart_type=chart_type, specs=specs)
            )
        except Exception as exc:
            errors.append(f"{chart_id}: {exc}")

    if errors and fail_on_error:
        raise RuntimeError("Chart question generation failed:\n" + "\n".join(errors))
    if errors:
        err_path = dataset_dir / "generate_questions_errors.txt"
        err_path.write_text("\n".join(errors), encoding="utf-8")

    out_json = dataset_dir / "dataset" / "questions.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8")

    images_dir = dataset_dir / "images"
    n = export_eval_bundle(all_questions, dataset_dir=dataset_dir, images_dir=images_dir)
    samples = len({q["chart_id"] for q in all_questions}) if all_questions else 0
    return {
        "dataset": dataset_dir.name,
        "samples": samples,
        "metadata_rows": len(metadata),
        "questions": len(all_questions),
        "questions_per_sample": len(all_questions) // samples if samples else 0,
        "charts_with_bbox_annotations": with_bbox,
        "charts_with_csv_fallback": with_fallback,
        "errors": len(errors),
        "eval_bundle": str(dataset_dir / "eval_bundle" / "samples.jsonl"),
        "exported_rows": n,
    }


def process_clevr_dataset(
    dataset_dir: Path,
    scenes_path: Path,
    *,
    specs: dict,
    val_questions_path: Path | None = None,
    val_question2bboxes_path: Path | None = None,
) -> dict:
    val_questions_path = (val_questions_path or REPO / "CLEVR_val_questions_new.json").resolve()
    val_question2bboxes_path = (
        val_question2bboxes_path or REPO / "CLEVR_val_question2bboxes.json"
    ).resolve()
    if not val_questions_path.is_file():
        raise FileNotFoundError(f"Val questions JSON not found: {val_questions_path}")
    if not val_question2bboxes_path.is_file():
        raise FileNotFoundError(f"Val question2bboxes JSON not found: {val_question2bboxes_path}")

    val_questions, question2bboxes, _image_bboxes = load_val_bbox_resources(
        val_questions_path,
        val_question2bboxes_path,
    )
    scenes_data = json.loads(scenes_path.read_text(encoding="utf-8"))
    scene_by_image = {s["image_filename"]: s for s in scenes_data["scenes"]}

    # scenes present in our 100-image test set
    images_dir = dataset_dir / "images"
    image_names = sorted(p.name for p in images_dir.glob("*.png"))
    val_bbox_targets = build_val_bbox_targets_by_image(image_names, val_questions, question2bboxes)
    all_questions: list[dict] = []
    missing: list[str] = []

    for image_name in image_names:
        scene = scene_by_image.get(image_name)
        if scene is None:
            missing.append(image_name)
            continue
        all_questions.extend(
            build_clevr_questions(
                scene, val_bbox_target=val_bbox_targets[image_name], specs=specs
            )
        )

    if missing:
        raise RuntimeError(f"Missing scenes for {len(missing)} images, e.g. {missing[:3]}")

    out_json = dataset_dir / "questions.json"
    out_json.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8")

    n = export_eval_bundle(
        all_questions,
        dataset_dir=dataset_dir,
        images_dir=images_dir,
        id_key="scene_id",
    )
    return {
        "dataset": dataset_dir.name,
        "samples": len(image_names),
        "questions": len(all_questions),
        "questions_per_sample": len(all_questions) // len(image_names),
        "eval_bundle": str(dataset_dir / "eval_bundle" / "samples.jsonl"),
        "exported_rows": n,
    }


def validate_questions(
    questions: list[dict],
    samples: int,
    *,
    dataset_name: str,
    perception_count: int = 3,
    total_per_sample: int = 9,
) -> None:
    by_sample: dict[str, list[dict]] = defaultdict(list)
    id_key = "scene_id" if questions and "scene_id" in questions[0] else "chart_id"
    for q in questions:
        by_sample[q[id_key]].append(q)

    if len(by_sample) != samples:
        raise RuntimeError(f"{dataset_name}: expected {samples} samples, got {len(by_sample)}")

    def p_cat(tid: str) -> str:
        if "COUNT" in tid:
            return "count"
        if "HEIGHT" in tid:
            return "height"
        if "TREND" in tid:
            return "trend"
        if "POS" in tid:
            return "position"
        if "COLOR" in tid:
            return "color_binary"
        return "other"

    def g_cat(tid: str) -> str:
        if "BBOX" in tid:
            return "bbox"
        if tid.startswith("G-VT"):
            return "value_to_label"
        if tid.startswith("G-TV"):
            return "value_text"
        if "ATTR" in tid:
            return "attribute"
        if "SPATIAL" in tid:
            return "spatial"
        return "other"

    def r_cat(tid: str) -> str:
        if "RANK" in tid:
            return "rank"
        if "ARITH" in tid:
            return "arith"
        if "MULTI" in tid or "CLOSEST" in tid:
            return "multi"
        if "COMPARE" in tid:
            return "compare"
        if tid in ("R-MATERIAL-COUNT", "R-SPATIAL-HALF"):
            return "open_count"
        if "COUNT" in tid:
            return "count"
        if "CHAIN" in tid or "SAME" in tid:
            return "relation"
        return "other"

    for sid, rows in by_sample.items():
        if len(rows) != total_per_sample:
            raise RuntimeError(
                f"{dataset_name} {sid}: expected {total_per_sample} questions, got {len(rows)}"
            )
        stages = [r["stage"] for r in rows]
        if (
            stages.count("Perception") != perception_count
            or stages.count("Grounding") != 3
            or stages.count("Reasoning") != 3
        ):
            raise RuntimeError(f"{dataset_name} {sid}: bad stage mix {stages}")

        if id_key == "chart_id":
            pc = {p_cat(r["template_id"]) for r in rows if r["stage"] == "Perception"}
            gc = {g_cat(r["template_id"]) for r in rows if r["stage"] == "Grounding"}
            rc = {r_cat(r["template_id"]) for r in rows if r["stage"] == "Reasoning"}
            if pc != {"count", "height", "trend"}:
                raise RuntimeError(f"{dataset_name} {sid}: bad perception categories {pc}")
            if gc != {"bbox", "value_to_label", "value_text"}:
                raise RuntimeError(f"{dataset_name} {sid}: bad grounding categories {gc}")
            if rc != {"rank", "arith", "multi"}:
                raise RuntimeError(f"{dataset_name} {sid}: bad reasoning categories {rc}")
        else:

            def clevr_p_cat(tid: str) -> str:
                if tid == "P-COUNT-TOTAL":
                    return "count_total"
                if "LARGE" in tid or "SMALL" in tid:
                    return "count_size"
                if "COUNT" in tid:
                    return "count_shape"
                return "other"

            pc = {clevr_p_cat(r["template_id"]) for r in rows if r["stage"] == "Perception"}
            gc = {g_cat(r["template_id"]) for r in rows if r["stage"] == "Grounding"}
            rc = {r_cat(r["template_id"]) for r in rows if r["stage"] == "Reasoning"}
            if pc != {"count_total", "count_size", "count_shape"}:
                raise RuntimeError(f"{dataset_name} {sid}: bad perception categories {pc}")
            if gc != {"bbox", "attribute", "spatial"}:
                raise RuntimeError(f"{dataset_name} {sid}: bad grounding categories {gc}")
            if rc != {"compare", "open_count", "relation"}:
                raise RuntimeError(f"{dataset_name} {sid}: bad reasoning categories {rc}")


def dataset_dir_for(name: str) -> Path:
    if name in ("bar_charts_100", "bar_charts_1000", "pie_charts_100", "clevr_100"):
        return REPO / "data" / name
    if name in ("bar_charts_500", "pie_charts_500", "clevr_500"):
        return REPO / "data" / name
    raise ValueError(f"Unknown dataset: {name}")


def process_named_dataset(name: str, args: argparse.Namespace) -> dict:
    specs = args.templates_specs
    if name in ("bar_charts_100", "bar_charts_500"):
        dataset_dir = dataset_dir_for(name)
        summary = process_chart_dataset(dataset_dir, chart_type="bar", specs=specs)
        questions = json.loads((dataset_dir / "dataset" / "questions.json").read_text())
        validate_questions(questions, summary["samples"], dataset_name=name)
        return summary
    if name == "bar_charts_1000":
        dataset_dir = dataset_dir_for(name)
        summary = process_chart_dataset(
            dataset_dir,
            chart_type="bar",
            specs=specs,
            allow_fallback_annotations=True,
            fail_on_error=False,
        )
        questions = json.loads((dataset_dir / "dataset" / "questions.json").read_text())
        validate_questions(questions, summary["samples"], dataset_name=name)
        meta_path = dataset_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        meta.update(
            {
                "question_structure": "9_per_chart",
                "questions_per_chart": 9,
                "total_questions": summary["questions"],
                "charts_with_bbox_annotations": summary["charts_with_bbox_annotations"],
                "charts_with_csv_fallback": summary["charts_with_csv_fallback"],
                "questions_json": f"{dataset_dir.name}/dataset/questions.json",
                "eval_bundle": f"{dataset_dir.name}/eval_bundle/samples.jsonl",
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return summary
    if name in ("pie_charts_100", "pie_charts_500"):
        dataset_dir = dataset_dir_for(name)
        summary = process_chart_dataset(dataset_dir, chart_type="pie", specs=specs)
        questions = json.loads((dataset_dir / "dataset" / "questions.json").read_text())
        validate_questions(questions, summary["samples"], dataset_name=name)
        return summary
    dataset_dir = dataset_dir_for(name)
    scenes_path = args.clevr_scenes.resolve()
    if not scenes_path.is_file():
        raise SystemExit(f"CLEVR scenes JSON not found: {scenes_path}")
    summary = process_clevr_dataset(
        dataset_dir,
        scenes_path,
        specs=specs,
        val_questions_path=args.clevr_val_questions,
        val_question2bboxes_path=args.clevr_val_question2bboxes,
    )
    questions = json.loads((dataset_dir / "questions.json").read_text())
    validate_questions(questions, summary["samples"], dataset_name=name)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        choices=[
            "bar_charts_100",
            "bar_charts_500",
            "bar_charts_1000",
            "pie_charts_100",
            "pie_charts_500",
            "clevr_100",
            "clevr_500",
            "all",
        ],
        default="all",
    )
    ap.add_argument(
        "--clevr-scenes",
        type=Path,
        default=REPO / "data" / "clevr_100" / "scenes" / "CLEVR_val_scenes.json",
    )
    ap.add_argument(
        "--templates",
        type=Path,
        default=None,
        help="Path to question_templates.json (default: src/question_templates.json)",
    )
    ap.add_argument(
        "--clevr-val-questions",
        type=Path,
        default=REPO / "CLEVR_val_questions_new.json",
    )
    ap.add_argument(
        "--clevr-val-question2bboxes",
        type=Path,
        default=REPO / "CLEVR_val_question2bboxes.json",
    )
    args = ap.parse_args()
    args.templates_specs = load_templates(args.templates)

    targets = (
        ["bar_charts_100", "pie_charts_100", "clevr_100"]
        if args.dataset == "all"
        else [args.dataset]
    )
    summaries = []

    for name in targets:
        summaries.append(process_named_dataset(name, args))
        print(json.dumps(summaries[-1], indent=2))


if __name__ == "__main__":
    main()
