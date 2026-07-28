"""
Load portable chart QA bundles (samples.jsonl + images/, or ProbeQ *.jsonl).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def image_to_pil(value: object) -> PILImage.Image | None:
    """Normalize HF dataset image values (PIL, dict, bytes, path, ndarray) to RGB PIL."""
    if value is None:
        return None
    if isinstance(value, PILImage.Image):
        return value

    if isinstance(value, dict):
        b = value.get("bytes")
        if b is not None and len(b) > 0:
            return PILImage.open(io.BytesIO(b if isinstance(b, (bytes, bytearray)) else bytes(b)))
        p = value.get("path")
        if p is not None and str(p).strip():
            path = Path(str(p)).expanduser()
            try:
                if path.is_file():
                    return PILImage.open(path)
                return PILImage.open(str(p))
            except Exception:
                return None
        return None

    if isinstance(value, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(value))

    if isinstance(value, str) and Path(value).expanduser().is_file():
        return PILImage.open(value)

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            arr = np.asarray(value)
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating) and float(arr.max()) <= 1.0:
                    arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(
                arr.shape[1], arr.shape[2]
            ):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 2:
                return PILImage.fromarray(np.ascontiguousarray(arr), mode="L")
            if arr.ndim == 3:
                return PILImage.fromarray(np.ascontiguousarray(arr))
    except ImportError:
        pass

    return None


def materialize_rows_with_pil_images(ds: Any) -> list[dict]:
    """Build one dict per row with image decoded to PIL."""
    image_feature = ds.features.get("image") if ds.features else None
    rows: list[dict] = []
    for i in range(len(ds)):
        row = {k: ds[i][k] for k in ds.column_names}
        raw = row.get("image")
        pil: PILImage.Image | None = None
        if isinstance(raw, PILImage.Image):
            pil = raw
        elif image_feature is not None and hasattr(image_feature, "decode_example"):
            try:
                pil = image_feature.decode_example(raw)
            except Exception:
                pil = None
        if pil is None:
            pil = image_to_pil(raw)
        if pil is not None:
            row["image"] = pil
        else:
            hint = ""
            if isinstance(raw, dict):
                hint = f" dict keys={list(raw.keys())}"
            logger.warning(
                "Row %d: could not decode image (type=%s%s)",
                i,
                type(raw).__name__,
                hint,
            )
        rows.append(row)
    return rows


def _resolve_test_bundle_paths(path: Path) -> tuple[Path, Path]:
    """
    Normalize bundle path: either a directory containing samples.jsonl, or a path to
    samples.jsonl itself. Returns (bundle_dir, jsonl_path).
    """
    p = path.expanduser().resolve()
    if p.is_file():
        return p.parent, p
    if p.suffix.lower() == ".jsonl" or p.name == "samples.jsonl":
        return p.parent, p
    return p, p / "samples.jsonl"


def test_bundle_output_root(bundle_dir_or_jsonl: Path) -> Path:
    """Directory for run outputs: bundle folder, or parent of samples.jsonl if a file was passed."""
    p = bundle_dir_or_jsonl.expanduser().resolve()
    return p if p.is_dir() else p.parent


def default_test_bundle_dir(script_dir: Path) -> Path:
    """
    Default ``--test-bundle`` directory: first folder under ``script_dir`` that contains
    ``samples.jsonl``. Prefers ``bar_chart_bundle/``, then ``bar_chart_data/``, then optional
    ``test_bundle/``; otherwise any sibling dir with ``samples.jsonl``; falls back to
    ``bar_chart_bundle/``.
    """
    root = script_dir.expanduser().resolve()
    for name in (
        "bar_chart_bundle",
        "bar_chart_data",
        "test_bundle",
    ):
        d = root / name
        if (d / "samples.jsonl").is_file():
            return d
    try:
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "samples.jsonl").is_file():
                return child
    except OSError:
        pass
    return root / "bar_chart_bundle"


def _normalize_test_bundle_row(rec: dict, line_index: int) -> dict:
    """
    ChartNet-style rows already have complex_question + full_reasoning.

    Bar-chart / ProbeQ rows use ``question``, optional ``think``, ``answer`` (string or number — including 0),
    ``image_file`` (relative to the JSONL parent dir, e.g. ``images/0001.png``), and ``sample_id`` / ``id``.
    """
    cq = rec.get("complex_question")
    if isinstance(cq, str) and cq.strip():
        out = dict(rec)
        if "run_index" not in out:
            out["run_index"] = out.get("index", line_index)
        for extra in ("code", "csv", "summary", "hf_config", "hf_split"):
            if extra in rec:
                out[extra] = rec[extra]
        return out

    q_text = rec.get("question")
    if q_text is None or (isinstance(q_text, str) and not q_text.strip()):
        raise ValueError(
            "Each JSONL line needs ChartNet fields (complex_question, full_reasoning) "
            "or bar-chart fields (question, think, answer, image_file). "
            f"Keys seen: {list(rec.keys())}"
        )

    think = rec.get("think")
    think = "" if think is None else str(think)
    ans_raw = rec.get("answer", None)
    if ans_raw is None:
        answer = ""
    elif isinstance(ans_raw, bool):
        answer = str(ans_raw)
    else:
        answer = str(ans_raw)
    complex_question = json.dumps([str(q_text)])
    full_reasoning = json.dumps(
        {
            "think": think,
            "answer": answer,
            "raw": (
                f"<think>\n{think}\n</think>\n\n"
                f"<answer>\n{answer}\n</answer>"
            ),
        }
    )
    rel_img = rec.get("image_file", "")
    if not rel_img:
        raise ValueError("Bar-chart row missing image_file")
    row_id = rec.get("id")
    if row_id is None or str(row_id).strip() == "":
        row_id = rec.get("sample_id", "")
    out = {
        "run_index": rec.get("index", rec.get("run_index", line_index)),
        "id": str(row_id),
        "complex_question": complex_question,
        "full_reasoning": full_reasoning,
        "image_file": rel_img,
    }
    for extra in (
        "answer_type",
        "stage",
        "template_id",
        "chart_id",
        "complexity_bin",
        "bar_count",
    ):
        if extra in rec:
            out[extra] = rec[extra]
    return out


def load_test_bundle(bundle_dir_or_jsonl: Path) -> list[dict]:
    """
    Load sample dicts: ``image`` (PIL), ``complex_question``, ``full_reasoning``, ``id``.

    ``bundle_dir_or_jsonl`` may be:
    - a directory that contains ``samples.jsonl`` and ``images/``, or
    - a direct path to a JSONL file (e.g. ``ProbeQ/probes_P4.jsonl``).

    For a file path, images are resolved as ``<parent of jsonl> / <image_file>``.
    """
    bundle_dir, jsonl_path = _resolve_test_bundle_paths(bundle_dir_or_jsonl)
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"Missing {jsonl_path}. "
            "Pass --test-bundle DIR (with DIR/samples.jsonl) or --test-bundle PATH/samples.jsonl. "
            "Expected layouts: bar_chart_bundle/, ProbeQ/*.jsonl with sibling images/, etc."
        )
    samples: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            raw = json.loads(line)
            rec = _normalize_test_bundle_row(raw, i)
            rel = rec.pop("image_file", "")
            path = bundle_dir / rel
            if path.is_file():
                rec["image"] = PILImage.open(path).convert("RGB")
            else:
                raise FileNotFoundError(f"Missing image: {path}")
            samples.append(rec)
    logger.debug("Loaded %d samples from %s", len(samples), jsonl_path)
    return samples


def load_chartnet_samples(
    *,
    hf_config: str,
    hf_split: str = "test",
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load eval samples directly from HuggingFace (no local JSONL export required).

    - ``human_verified`` / ``test``: official 2,000-chart held-out set (no QA fields).
    - ``reasoning`` / ``train``: complex-reasoning QA; optionally subsample with ``max_samples`` + ``seed``.
    """
    import random

    if hf_config == "human_verified":
        from chartnet_datasets import load_human_verified

        ds = load_human_verified(hf_split)  # type: ignore[arg-type]
        rows = materialize_rows_with_pil_images(ds)
        samples: list[dict] = []
        for i, row in enumerate(rows):
            samples.append(
                {
                    "run_index": i,
                    "id": row.get("id", ""),
                    "image": row.get("image"),
                    "code": row.get("code", ""),
                    "csv": row.get("csv", ""),
                    "summary": row.get("summary", ""),
                    "hf_config": hf_config,
                    "hf_split": hf_split,
                }
            )
    elif hf_config == "reasoning":
        from chartnet_datasets import load_reasoning

        ds = load_reasoning(hf_split)

        def _keep(batch: dict) -> list[bool]:
            return [bool(str(x or "").strip()) for x in batch["full_reasoning"]]

        before = len(ds)
        ds = ds.filter(_keep, batched=True, batch_size=1024)
        after = len(ds)
        logger.info("reasoning rows with full_reasoning: %d / %d", after, before)
        if after == 0:
            raise RuntimeError("No reasoning rows with non-empty full_reasoning.")
        if max_samples is not None and max_samples < after:
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(after), max_samples))
            ds = ds.select(indices)
        rows = materialize_rows_with_pil_images(ds)
        samples = []
        for i, row in enumerate(rows):
            samples.append(
                {
                    "run_index": i,
                    "id": row.get("id", ""),
                    "image": row.get("image"),
                    "complex_question": row.get("complex_question", ""),
                    "full_reasoning": str(row.get("full_reasoning") or ""),
                    "hf_config": hf_config,
                    "hf_split": hf_split,
                }
            )
    else:
        raise ValueError(
            f"Unsupported hf_config {hf_config!r}. Use 'human_verified' or 'reasoning'."
        )

    for s in samples:
        img = s.get("image")
        if img is not None and not isinstance(img, PILImage.Image):
            s["image"] = image_to_pil(img)
    return samples


def export_test_bundle(rows: list[dict], bundle_dir: Path) -> None:
    """Write PNGs + samples.jsonl for a portable ChartNet reasoning / CR bundle."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    img_dir = bundle_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = bundle_dir / "samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            fname = f"{i:04d}.png"
            rel_image = f"images/{fname}"
            img = row.get("image")
            if not isinstance(img, PILImage.Image):
                img = image_to_pil(img)
            if isinstance(img, PILImage.Image):
                img.save(str(img_dir / fname), format="PNG")
            else:
                logger.warning("Row %d: no PIL image after decode; skipping PNG", i)
            rec = {
                "run_index": i,
                "id": row.get("id", ""),
                "complex_question": row.get("complex_question", ""),
                "full_reasoning": str(row.get("full_reasoning") or ""),
                "image_file": rel_image,
            }
            for extra in ("code", "csv", "summary"):
                if extra in row and row[extra]:
                    rec[extra] = row[extra]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote test bundle → %s (images/ + samples.jsonl)", bundle_dir)
