"""
Load official ChartNet HuggingFace configs (see ibm-granite/ChartNet dataset card).

Splits are disjoint:
- ``core`` + ``reasoning`` overlap on ``id`` (join for full tuples with QA).
- ``human_verified`` train/test are completely disjoint from ``core`` and ``reasoning``.

Example (human-verified held-out test, 2000 charts)::

    from chartnet_datasets import load_human_verified
    human_verified_test = load_human_verified("test")

Complex-reasoning QA (``complex_question``, ``full_reasoning`` with CoT gold) lives in
the ``reasoning`` config, not in ``human_verified``.
"""

from __future__ import annotations

from typing import Any, Literal

CHARTNET_DATASET = "ibm-granite/ChartNet"

HumanVerifiedSplit = Literal["train", "test"]
ReasoningSplit = Literal["train"]


def _load_dataset(name: str, split: str) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from e
    return load_dataset(CHARTNET_DATASET, name=name, split=split)


def load_human_verified(split: HumanVerifiedSplit = "test") -> Any:
    """
    Human-reviewed charts: ``id``, ``image``, ``code``, ``csv``, ``summary``.

    - ``train``: 92,603 rows
    - ``test``: 2,000-row held-out eval set (disjoint from core/reasoning)
    """
    if split not in ("train", "test"):
        raise ValueError(f"human_verified split must be 'train' or 'test', got {split!r}")
    return _load_dataset("human_verified", split)


def load_reasoning(split: ReasoningSplit | str = "train") -> Any:
    """
    Synthetic complex-reasoning QA: ``id``, ``image``, ``complex_question``, ``full_reasoning``.

    Gold answers for RapidFuzz eval are parsed from ``full_reasoning`` (JSON ``answer`` field).
    """
    return _load_dataset("reasoning", split)


def load_core(split: str = "train") -> Any:
    """Synthetic core tuples: ``id``, ``image``, ``code``, ``csv``, ``summary``."""
    return _load_dataset("core", split)
