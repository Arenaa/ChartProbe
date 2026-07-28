#!/usr/bin/env python3
"""Build a summary catalog from ``src/question_templates.json``.

Output: src/question_templates_catalog.json

Usage:
  python src/build_question_templates_catalog.py
  python src/build_question_templates_catalog.py --templates path/to/custom.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from templates import (  # noqa: E402
    clevr_test_set_selection,
    clear_templates_cache,
    family_spec,
    iter_templates,
    load_templates,
)

OUT = SCRIPT_DIR / "question_templates_catalog.json"

COMPLEX_REASONING_DATASETS = [
    {
        "dataset": "clevr_complex_reasoning_500",
        "path": "data/clevr_complex_reasoning_500",
        "parent_9q": "clevr_500",
        "questions_per_sample": 1,
        "template_mode": "free_form",
        "description": "One original CLEVR val question per scene (bbox-linked); no fixed template ID",
    },
    {
        "dataset": "complex_reasoning_bar_500",
        "path": "data/complex_reasoning_bar_500",
        "questions_per_sample": 1,
        "template_mode": "free_form",
        "description": "ChartNet reasoning split; complex_question field per chart",
    },
    {
        "dataset": "complex_reasoning_pie_500",
        "path": "data/complex_reasoning_pie_500",
        "questions_per_sample": 1,
        "template_mode": "free_form",
        "description": "ChartNet reasoning split; complex_question field per chart",
    },
]


def dataset_entry(
    name: str,
    *,
    path: str,
    role: str,
    samples: int,
    questions_per_sample: int,
    chart_type: str,
    template_family: str,
    active_template_ids: list[str],
    notes: str = "",
) -> dict:
    return {
        "dataset": name,
        "path": path,
        "role": role,
        "samples": samples,
        "questions_per_sample": questions_per_sample,
        "total_questions": samples * questions_per_sample,
        "chart_type": chart_type,
        "template_family": template_family,
        "active_template_ids": active_template_ids,
        "notes": notes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--templates",
        type=Path,
        default=None,
        help="Custom templates JSON (default: src/question_templates.json)",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    clear_templates_cache()
    specs = load_templates(args.templates)

    bar = iter_templates(specs, "bar")
    pie = iter_templates(specs, "pie")
    clevr = iter_templates(specs, "clevr")
    clevr_sel = clevr_test_set_selection(specs)

    clevr_active = []
    for stage, cats in clevr_sel.items():
        for cat, tid in cats.items():
            resolved = tid.replace("{SHAPE}", "CUBE|SPHERE|CYLINDER (most frequent shape)")
            clevr_active.append(
                {
                    "stage": stage,
                    "category": cat,
                    "template_id": tid,
                    "resolved_template_id": resolved,
                }
            )

    bar_ids = [t["template_id"] for t in bar]
    pie_ids = [t["template_id"] for t in pie]
    clevr_9q_ids = [a["template_id"] for a in clevr_active]

    datasets = [
        dataset_entry(
            "bar_charts_1000",
            path="data/bar_charts_1000",
            role="train",
            samples=1000,
            questions_per_sample=9,
            chart_type="bar",
            template_family="bar",
            active_template_ids=bar_ids,
            notes="Stage LoRA / SFT training source",
        ),
        dataset_entry(
            "bar_charts_500",
            path="data/bar_charts_500",
            role="test",
            samples=500,
            questions_per_sample=9,
            chart_type="bar",
            template_family="bar",
            active_template_ids=bar_ids,
        ),
        dataset_entry(
            "pie_charts_500",
            path="data/pie_charts_500",
            role="test",
            samples=500,
            questions_per_sample=9,
            chart_type="pie",
            template_family="pie",
            active_template_ids=pie_ids,
        ),
        dataset_entry(
            "clevr_500",
            path="data/clevr_500",
            role="test",
            samples=500,
            questions_per_sample=9,
            chart_type="clevr",
            template_family="clevr",
            active_template_ids=clevr_9q_ids,
            notes="Active set from families.clevr.test_set_selection",
        ),
    ]

    src_templates = Path(args.templates) if args.templates else (SCRIPT_DIR / "question_templates.json")
    try:
        source_templates = src_templates.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        source_templates = "src/question_templates.json"

    catalog = {
        "version": 2,
        "description": "Summary catalog derived from question_templates.json",
        "source_templates": source_templates,
        "generator": "src/build_question_templates_catalog.py",
        "selection_rule": (
            "Each 9-Q sample uses exactly one fixed template per stage category "
            "(3 Perception + 3 Grounding + 3 Reasoning)."
        ),
        "template_families": {
            "bar": {
                "questions_per_sample": family_spec(specs, "bar").get("questions_per_sample", 9),
                "templates": bar,
            },
            "pie": {
                "questions_per_sample": family_spec(specs, "pie").get("questions_per_sample", 9),
                "templates": pie,
            },
            "clevr": {
                "questions_per_sample": family_spec(specs, "clevr").get("questions_per_sample", 9),
                "test_set_selection": clevr_sel,
                "active_templates": clevr_active,
                "templates": clevr,
            },
        },
        "datasets_9q": datasets,
        "datasets_complex_reasoning": COMPLEX_REASONING_DATASETS,
        "stage_category_matrix": {
            "bar_charts": "Perception: count|height|trend — Grounding: bbox|value_to_label|value_lookup — Reasoning: rank|arithmetic|multi_step",
            "pie_charts": "Perception: count|height|trend — Grounding: bbox|value_to_label|value_lookup — Reasoning: rank|arithmetic|multi_step",
            "clevr": "Perception: count_total|count_size|count_shape — Grounding: bbox|attribute|spatial — Reasoning: compare|open_count|relation",
        },
        "customize": (
            "Edit src/question_templates.json (or pass --templates) then re-run "
            "generate_probe_questions.py / build_question_templates_catalog.py."
        ),
    }

    args.out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"  families: {sorted(specs['families'])}")
    print(f"  bar templates: {len(bar)}")
    print(f"  pie templates: {len(pie)}")
    print(f"  clevr templates: {len(clevr)}")


if __name__ == "__main__":
    main()
