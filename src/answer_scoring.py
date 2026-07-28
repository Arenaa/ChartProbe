"""Answer scoring helpers for ChartNet benchmarks (beyond fuzzy string match)."""

from __future__ import annotations

import re
from typing import Any

_fuzz_mod = None


def _fuzz():
    global _fuzz_mod
    if _fuzz_mod is None:
        try:
            from rapidfuzz import fuzz as _f
        except ImportError as e:
            raise ImportError("Install RapidFuzz: pip install rapidfuzz") from e
        _fuzz_mod = _f
    return _fuzz_mod


def normalize_text(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_label(text: str) -> str:
    s = normalize_text(text)
    s = re.sub(r"[^\w\s&%-]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_number(text: str) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    s = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def fuzzy_ratio(pred: str, gold: str) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    return float(_fuzz().ratio(pred, gold)) / 100.0


def fuzzy_token_sort_ratio(pred: str, gold: str) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    return float(_fuzz().token_sort_ratio(pred, gold)) / 100.0


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def exact_label_match(pred: str, gold: str) -> bool:
    return normalize_label(pred) == normalize_label(gold)


def fuzzy_threshold(pred: str, gold: str, threshold: float) -> bool:
    return fuzzy_ratio(pred, gold) >= threshold


_BINARY_YES = frozenset({"yes", "true", "y"})
_BINARY_NO = frozenset({"no", "false", "n"})


def _normalize_binary(text: str) -> str | None:
    s = normalize_text(text)
    if s in _BINARY_YES:
        return "yes"
    if s in _BINARY_NO:
        return "no"
    return None


def binary_match(pred: str, gold: str) -> bool | None:
    g = _normalize_binary(gold)
    p = _normalize_binary(pred)
    if g is None:
        return None
    if p is None:
        return False
    return p == g


def numeric_match(
    pred: str,
    gold: str,
    *,
    rel_tol: float = 0.05,
    abs_tol: float = 0.1,
) -> bool | None:
    g = parse_number(gold)
    p = parse_number(pred)
    if g is None:
        return None
    if p is None:
        return False
    if rel_tol <= 0 and abs_tol <= 0:
        return p == g
    diff = abs(p - g)
    if diff <= abs_tol:
        return True
    denom = max(abs(g), 1e-9)
    return diff / denom <= rel_tol


def type_aware_correct(pred: str, gold: str, answer_type: str | None) -> bool:
    at = (answer_type or "").strip().lower()
    if at == "binary":
        matched = binary_match(pred, gold)
        return bool(matched) if matched is not None else fuzzy_threshold(pred, gold, 0.8)
    if at == "count":
        matched = numeric_match(pred, gold, rel_tol=0.0, abs_tol=0.0)
        if matched is not None:
            return matched
        return exact_match(pred, gold)
    if at == "value":
        matched = numeric_match(pred, gold, rel_tol=0.05, abs_tol=0.1)
        if matched is not None:
            return matched
        return fuzzy_threshold(pred, gold, 0.8)
    if at == "label":
        if exact_label_match(pred, gold):
            return True
        return fuzzy_threshold(pred, gold, 0.9)
    if at == "open":
        return fuzzy_threshold(pred, gold, 0.8)
    return fuzzy_threshold(pred, gold, 0.8)


def hard_correct(pred: str, gold: str, answer_type: str | None = None) -> bool:
    """Strict correctness: exact match, with ±5% numeric tolerance when applicable.

    Used alongside RapidFuzz to show gains are not an artifact of fuzzy token overlap.
    When ``answer_type`` is known, uses the same numeric/binary/label rules as type-aware
    scoring but never falls back to a fuzzy threshold. When unknown (e.g. CR-500),
    accepts numeric ±5% if gold parses as a number, else normalized exact match.
    """
    at = (answer_type or "").strip().lower()
    if at == "binary":
        matched = binary_match(pred, gold)
        return bool(matched) if matched is not None else exact_match(pred, gold)
    if at == "count":
        matched = numeric_match(pred, gold, rel_tol=0.0, abs_tol=0.0)
        if matched is not None:
            return matched
        return exact_match(pred, gold)
    if at == "value":
        matched = numeric_match(pred, gold, rel_tol=0.05, abs_tol=0.1)
        if matched is not None:
            return matched
        return exact_match(pred, gold)
    if at == "label":
        return exact_label_match(pred, gold)
    if at in {"open", "bbox"}:
        return exact_match(pred, gold)

    # Untyped (complex-reasoning): prefer numeric ±5% when gold is numeric.
    n = numeric_match(pred, gold, rel_tol=0.05, abs_tol=0.1)
    if n is not None and parse_number(gold) is not None:
        return bool(n)
    if exact_label_match(pred, gold):
        return True
    return exact_match(pred, gold)


def score_answer(
    pred: str,
    gold: str,
    *,
    answer_type: str | None = None,
) -> dict[str, Any]:
    fuzzy = fuzzy_ratio(pred, gold)
    token_sort = fuzzy_token_sort_ratio(pred, gold)
    row: dict[str, Any] = {
        "fuzzy_accuracy": round(fuzzy, 6),
        "fuzzy_token_sort_ratio": round(token_sort, 6),
        "exact_match": int(exact_match(pred, gold)),
        "exact_label_match": int(exact_label_match(pred, gold)),
        "fuzzy_ge_80": int(fuzzy >= 0.80),
        "fuzzy_ge_90": int(fuzzy >= 0.90),
        "type_aware_correct": int(type_aware_correct(pred, gold, answer_type)),
        "hard_correct": int(hard_correct(pred, gold, answer_type)),
    }
    b = binary_match(pred, gold)
    if b is not None:
        row["binary_match"] = int(b)
    n_count = numeric_match(pred, gold, rel_tol=0.0, abs_tol=0.0)
    if n_count is not None:
        row["numeric_exact_match"] = int(n_count)
    n_value = numeric_match(pred, gold, rel_tol=0.05, abs_tol=0.1)
    if n_value is not None:
        row["numeric_tol_match"] = int(n_value)
    if answer_type:
        row["answer_type"] = answer_type
    return row


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    def _mean_pct(key: str) -> float:
        return round(_mean(key) * 100.0, 4)

    metrics = {
        "avg_fuzzy_accuracy": _mean("fuzzy_accuracy"),
        "avg_fuzzy_accuracy_percent": _mean_pct("fuzzy_accuracy"),
        "avg_fuzzy_token_sort_ratio": _mean("fuzzy_token_sort_ratio"),
        "avg_fuzzy_token_sort_ratio_percent": _mean_pct("fuzzy_token_sort_ratio"),
        "avg_exact_match": _mean("exact_match"),
        "avg_exact_match_percent": _mean_pct("exact_match"),
        "avg_exact_label_match": _mean("exact_label_match"),
        "avg_exact_label_match_percent": _mean_pct("exact_label_match"),
        "avg_fuzzy_ge_80": _mean("fuzzy_ge_80"),
        "avg_fuzzy_ge_80_percent": _mean_pct("fuzzy_ge_80"),
        "avg_fuzzy_ge_90": _mean("fuzzy_ge_90"),
        "avg_fuzzy_ge_90_percent": _mean_pct("fuzzy_ge_90"),
        "avg_type_aware_correct": _mean("type_aware_correct"),
        "avg_type_aware_correct_percent": _mean_pct("type_aware_correct"),
        "avg_hard_correct": _mean("hard_correct"),
        "avg_hard_correct_percent": _mean_pct("hard_correct"),
    }

    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        at = r.get("answer_type")
        if at:
            by_type.setdefault(str(at), []).append(r)

    if by_type:
        metrics["by_answer_type"] = {}
        for at, typed_rows in sorted(by_type.items()):
            def _typed_mean(key: str) -> float:
                vals = [float(r[key]) for r in typed_rows if r.get(key) is not None]
                return round(sum(vals) / len(vals), 6) if vals else 0.0

            metrics["by_answer_type"][at] = {
                "n": len(typed_rows),
                "avg_fuzzy_accuracy_percent": round(_typed_mean("fuzzy_accuracy") * 100.0, 4),
                "avg_exact_match_percent": round(_typed_mean("exact_match") * 100.0, 4),
                "avg_type_aware_correct_percent": round(
                    _typed_mean("type_aware_correct") * 100.0, 4
                ),
                "avg_hard_correct_percent": round(_typed_mean("hard_correct") * 100.0, 4),
            }

    if any(r.get("binary_match") is not None for r in rows):
        binary_rows = [r for r in rows if r.get("binary_match") is not None]
        metrics["avg_binary_match"] = round(
            sum(float(r["binary_match"]) for r in binary_rows) / len(binary_rows), 6
        )
        metrics["avg_binary_match_percent"] = round(metrics["avg_binary_match"] * 100.0, 4)

    if any(r.get("numeric_exact_match") is not None for r in rows):
        num_rows = [r for r in rows if r.get("numeric_exact_match") is not None]
        metrics["avg_numeric_exact_match"] = round(
            sum(float(r["numeric_exact_match"]) for r in num_rows) / len(num_rows), 6
        )
        metrics["avg_numeric_exact_match_percent"] = round(
            metrics["avg_numeric_exact_match"] * 100.0, 4
        )

    if any(r.get("numeric_tol_match") is not None for r in rows):
        tol_rows = [r for r in rows if r.get("numeric_tol_match") is not None]
        metrics["avg_numeric_tol_match"] = round(
            sum(float(r["numeric_tol_match"]) for r in tol_rows) / len(tol_rows), 6
        )
        metrics["avg_numeric_tol_match_percent"] = round(
            metrics["avg_numeric_tol_match"] * 100.0, 4
        )

    def _group_metrics(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not group_rows:
            return {"n": 0}
        return {
            "n": len(group_rows),
            "avg_fuzzy_accuracy_percent": round(_mean_key(group_rows, "fuzzy_accuracy") * 100.0, 4),
            "avg_exact_match_percent": round(_mean_key(group_rows, "exact_match") * 100.0, 4),
            "avg_type_aware_correct_percent": round(
                _mean_key(group_rows, "type_aware_correct") * 100.0, 4
            ),
            "avg_hard_correct_percent": round(_mean_key(group_rows, "hard_correct") * 100.0, 4),
        }

    def _mean_key(group_rows: list[dict[str, Any]], key: str) -> float:
        vals = [float(r[key]) for r in group_rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    by_stage: dict[str, list[dict[str, Any]]] = {}
    by_template: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        stage = r.get("stage")
        if stage:
            by_stage.setdefault(str(stage), []).append(r)
        template = r.get("template_id")
        if template:
            by_template.setdefault(str(template), []).append(r)

    if by_stage:
        metrics["by_stage"] = {k: _group_metrics(v) for k, v in sorted(by_stage.items())}
    if by_template:
        metrics["by_template_id"] = {k: _group_metrics(v) for k, v in sorted(by_template.items())}

    return metrics
