"""
Shared helpers for ChartNet runners: question parsing, reference extraction, simple metrics.

Used by run_benchmark.py, chartnet_qwen3vl_experiment.py, etc.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

CHART_INSTRUCTION = (
    "You are given a chart image and a question about it.\n\n"
    "Answer the question using the chart. Be concise and accurate."
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Minimal config placeholder for scripts that still import this type."""

    output_dir: Path = Path("outputs/chartnet")


def parse_chartnet_question(complex_question: str | None) -> str:
    """
    Normalize `complex_question` from bundles: either a plain string or a JSON list of one string.
    """
    if complex_question is None:
        return ""
    s = str(complex_question).strip()
    if not s:
        return ""
    if s.startswith("["):
        try:
            data = json.loads(s)
            if isinstance(data, list) and len(data) > 0:
                return str(data[0]).strip()
        except json.JSONDecodeError:
            pass
    return s


def parse_reference_fields(full_reasoning: str | None) -> tuple[str, str]:
    """
    Return (gold_answer, reference_raw_string).

    Expects JSON with an ``answer`` field (bar-chart / reasoning export). Falls back to ``<answer>...</answer>``
    inside ``raw`` or the full string.
    """
    if full_reasoning is None:
        return "", ""
    raw = str(full_reasoning).strip()
    if not raw:
        return "", ""

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            ans = str(obj.get("answer", "")).strip()
            if ans:
                return ans, raw
            nested = obj.get("raw")
            if isinstance(nested, str):
                m = re.search(r"<answer>\s*(.*?)\s*</answer>", nested, flags=re.DOTALL | re.IGNORECASE)
                if m:
                    return m.group(1).strip(), raw
    except json.JSONDecodeError:
        pass

    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), raw

    return "", raw


def parse_gold_thinking(full_reasoning: str | None) -> str:
    """Return gold chain-of-thought text from ``full_reasoning`` JSON or XML raw field."""
    if full_reasoning is None:
        return ""
    raw = str(full_reasoning).strip()
    if not raw:
        return ""

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            think = str(obj.get("think", "")).strip()
            if think:
                return think
            nested = obj.get("raw")
            if isinstance(nested, str):
                m = re.search(
                    r"<think>\s*(.*?)\s*</think>",
                    nested,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if m:
                    return m.group(1).strip()
    except json.JSONDecodeError:
        pass

    m = re.search(
        r"<think>\s*(.*?)\s*</think>",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def token_f1(prediction: str, reference: str) -> float:
    """Unigram F1 over whitespace tokens (case-insensitive)."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = len(pred_tokens & ref_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
