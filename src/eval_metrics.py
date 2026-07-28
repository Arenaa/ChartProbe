"""Primary eval metrics: type-aware for P/G/R test sets, fuzzy for complex reasoning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

COMPLEX_REASONING_DATASETS = frozenset(
    {
        "complex_reasoning_bar_100",
        "complex_reasoning_pie_100",
        "clevr_complex_reasoning_100",
        "complex_reasoning_bar_500",
        "complex_reasoning_pie_500",
        "clevr_complex_reasoning_500",
        "complex_reasoning_bar_500_extra400",
        "complex_reasoning_pie_500_extra400",
        "clevr_complex_reasoning_500_extra400",
    }
)

STAGE_TEST_DATASETS = frozenset(
    {
        "bar_charts_1000",
        "bar_charts_100",
        "pie_charts_100",
        "clevr_100",
        "bar_charts_500",
        "pie_charts_500",
        "clevr_500",
        "bar_charts_500_extra400",
        "pie_charts_500_extra400",
        "clevr_500_extra400",
    }
)

# Complex-reasoning QA: rapidfuzz ratio (not type-aware / exact).
COMPLEX_REASONING_METRIC = "avg_fuzzy_accuracy_percent"
STAGE_TEST_METRIC = "avg_type_aware_correct_percent"


def is_complex_reasoning_dataset(dataset: str) -> bool:
    return dataset in COMPLEX_REASONING_DATASETS


def primary_metric_key(dataset: str) -> str:
    if is_complex_reasoning_dataset(dataset):
        return COMPLEX_REASONING_METRIC
    return STAGE_TEST_METRIC


def primary_metric_value(metrics: dict[str, Any], dataset: str) -> float:
    key = primary_metric_key(dataset)
    return float(metrics.get(key, 0))


ALL_KNOWN_DATASETS = COMPLEX_REASONING_DATASETS | STAGE_TEST_DATASETS
_SETUPS = ("perception_grounding", "perception", "grounding", "reasoning", "all")


def _parse_run_tag_dataset(run_tag: str) -> str | None:
    if not run_tag:
        return None
    m = re.search(r"_seed(\d+)$", run_tag)
    if not m:
        return None
    body = run_tag[: m.start()]
    for ds in sorted(ALL_KNOWN_DATASETS, key=len, reverse=True):
        if body.endswith(f"_{ds}"):
            return ds
    for setup in _SETUPS:
        suffix = f"_{setup}_"
        idx = body.rfind(suffix)
        if idx >= 0:
            return body[idx + len(suffix) :]
    return None


def infer_dataset(
    *,
    run_tag: str | None = None,
    output_dir: Path | None = None,
    metrics: dict[str, Any] | None = None,
    metrics_path: Path | None = None,
) -> str | None:
    if metrics and metrics.get("dataset"):
        return str(metrics["dataset"])
    if run_tag:
        parsed = _parse_run_tag_dataset(run_tag)
        if parsed:
            return parsed
    if metrics_path is not None:
        for part in reversed(metrics_path.parts):
            if part in COMPLEX_REASONING_DATASETS or part in STAGE_TEST_DATASETS:
                return part
    if output_dir is not None:
        name = output_dir.name
        if name.startswith("rep"):
            return output_dir.parent.name if output_dir.parent else None
        if name in COMPLEX_REASONING_DATASETS or name in STAGE_TEST_DATASETS:
            return name
    return None


def attach_primary_metrics(
    metrics: dict[str, Any],
    *,
    dataset: str | None = None,
    run_tag: str | None = None,
    output_dir: Path | None = None,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    ds = dataset or infer_dataset(
        run_tag=run_tag or str(metrics.get("run_tag") or ""),
        output_dir=output_dir,
        metrics=metrics,
        metrics_path=metrics_path,
    )
    if not ds:
        return metrics
    key = primary_metric_key(ds)
    metrics = dict(metrics)
    metrics["dataset"] = ds
    metrics["primary_metric_key"] = key
    metrics["primary_metric_percent"] = round(primary_metric_value(metrics, ds), 4)
    metrics["eval_mode"] = "fuzzy" if is_complex_reasoning_dataset(ds) else "type_aware"
    return metrics


def expected_num_samples(dataset: str) -> int:
    if dataset in {
        "complex_reasoning_bar_100",
        "complex_reasoning_pie_100",
        "clevr_complex_reasoning_100",
    }:
        return 100
    if dataset in {
        "complex_reasoning_bar_500_extra400",
        "complex_reasoning_pie_500_extra400",
        "clevr_complex_reasoning_500_extra400",
    }:
        return 400
    if dataset in {
        "complex_reasoning_bar_500",
        "complex_reasoning_pie_500",
        "clevr_complex_reasoning_500",
    }:
        return 500
    if dataset in {"bar_charts_500_extra400", "pie_charts_500_extra400", "clevr_500_extra400"}:
        return 3600
    if dataset in {"bar_charts_500", "pie_charts_500", "clevr_500"}:
        return 4500
    return 900


def complex_reasoning_eval_complete(metrics: dict[str, Any], *, dataset: str) -> bool:
    """True when a complex-reasoning run has full fuzzy-primary metadata."""
    if not is_complex_reasoning_dataset(dataset):
        return False
    if int(metrics.get("num_samples") or 0) != expected_num_samples(dataset):
        return False
    if metrics.get("avg_fuzzy_accuracy_percent") is None:
        return False
    return (
        metrics.get("eval_mode") == "fuzzy"
        and metrics.get("primary_metric_key") == COMPLEX_REASONING_METRIC
    )


def primary_wandb_summary(metrics: dict[str, Any], dataset: str) -> dict[str, float | int | str]:
    """Top-level W&B scalars for the run's official score."""
    out: dict[str, float | int | str] = {
        "primary/metric_key": primary_metric_key(dataset),
        "primary/score_percent": primary_metric_value(metrics, dataset),
        "primary/eval_mode": "fuzzy" if is_complex_reasoning_dataset(dataset) else "type_aware",
    }
    if is_complex_reasoning_dataset(dataset):
        out["complex_reasoning/fuzzy_accuracy_percent"] = float(
            metrics.get("avg_fuzzy_accuracy_percent", 0)
        )
        return out

    out["stage_test/type_aware_correct_percent"] = float(
        metrics.get("avg_type_aware_correct_percent", 0)
    )
    for stage in ("Perception", "Grounding", "Reasoning"):
        block = (metrics.get("by_stage") or {}).get(stage) or {}
        if "avg_type_aware_correct_percent" in block:
            out[f"stage/{stage}/type_aware_correct_percent"] = float(
                block["avg_type_aware_correct_percent"]
            )
    return out
