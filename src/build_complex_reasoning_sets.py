#!/usr/bin/env python3
"""
Export N random samples from ibm-granite/ChartNet ``reasoning`` (train) filtered by chart type.

Uses the local HuggingFace snapshot (parquet) when available — much faster than streaming.

Output layout (default parent: data/complex_reasoning_100/)::

    bar_charts/   samples.jsonl + images/
    pie_charts/   samples.jsonl + images/
    manifest.json

Usage::

    python src/build_complex_reasoning_sets.py
    python src/build_complex_reasoning_sets.py --chart-type bar --num-samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from PIL import Image as PILImage

from test_bundle import export_test_bundle, image_to_pil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHARTNET_DATASET = "ibm-granite/ChartNet"
CHART_TYPE_RE = re.compile(r"ChartType=([^,\n]+)", re.I)

ChartTypeFilter = Literal["bar", "pie", "both"]


def chart_type_from_code(code: str) -> str | None:
    m = CHART_TYPE_RE.search(code or "")
    if not m:
        return None
    ct = m.group(1).strip().lower()
    if ct in {"bar", "bar chart"}:
        return "bar"
    if ct in {"pie", "pie chart"}:
        return "pie"
    return None


def find_local_snapshot(hf_home: Path | None = None) -> Path | None:
    """Return ChartNet snapshot dir with core/ + reasoning/ parquet if cached locally."""
    roots = []
    if hf_home is not None:
        roots.append(hf_home / "datasets--ibm-granite--ChartNet" / "snapshots")
    env = os.environ.get("HF_HOME")
    if env:
        roots.append(Path(env) / "datasets--ibm-granite--ChartNet" / "snapshots")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub" / "datasets--ibm-granite--ChartNet" / "snapshots")

    seen: set[Path] = set()
    for base in roots:
        if not base.is_dir():
            continue
        for snap in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not snap.is_dir() or snap in seen:
                continue
            seen.add(snap)
            if (snap / "core").is_dir() and (snap / "reasoning").is_dir():
                if any((snap / "core").glob("core_*.parquet")) and any(
                    (snap / "reasoning").glob("reasoning_*.parquet")
                ):
                    return snap
    return None


def parquet_paths(snapshot: Path, subdir: str, prefix: str) -> list[Path]:
    return sorted((snapshot / subdir).glob(f"{prefix}_*.parquet"))


def has_reasoning_fields(row: dict) -> bool:
    fr = row.get("full_reasoning")
    if isinstance(fr, str) and fr.strip():
        return True
    think = str(row.get("think") or "").strip()
    answer = row.get("answer")
    if answer is None:
        ans = ""
    elif isinstance(answer, bool):
        ans = str(answer)
    else:
        ans = str(answer).strip()
    return bool(think) and bool(ans)


def full_reasoning_json(row: dict) -> str:
    fr = row.get("full_reasoning")
    if isinstance(fr, str) and fr.strip():
        return fr
    think = str(row.get("think") or "")
    ans_raw = row.get("answer")
    if ans_raw is None:
        answer = ""
    elif isinstance(ans_raw, bool):
        answer = str(ans_raw)
    else:
        answer = str(ans_raw)
    return json.dumps(
        {
            "think": think,
            "answer": answer,
            "raw": (
                f"<think>\n{think}\n</think>\n\n"
                f"<answer>\n{answer}\n</answer>"
            ),
        },
        ensure_ascii=False,
    )


def build_chart_type_index_local(snapshot: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    index: dict[str, str] = {}
    files = parquet_paths(snapshot, "core", "core")
    logger.info("Local core index from %d parquet files under %s", len(files), snapshot)
    n = 0
    for path in files:
        table = pq.read_table(path, columns=["id", "code"])
        for cid, code in zip(table["id"].to_pylist(), table["code"].to_pylist()):
            cid = str(cid or "").strip()
            if not cid:
                continue
            ct = chart_type_from_code(code or "")
            if ct:
                index[cid] = ct
            n += 1
        if n % 200_000 == 0:
            logger.info("  core rows scanned: %d (typed ids: %d)", n, len(index))
    bar_n = sum(1 for v in index.values() if v == "bar")
    pie_n = sum(1 for v in index.values() if v == "pie")
    logger.info("Core index done: %d rows, %d bar/pie ids (bar=%d pie=%d)", n, len(index), bar_n, pie_n)
    return index


def collect_reasoning_id_pools_local(
    snapshot: Path,
    chart_type_index: dict[str, str],
    *,
    want_types: set[str],
) -> dict[str, list[str]]:
    import pyarrow.parquet as pq

    pools: dict[str, list[str]] = {t: [] for t in want_types}
    files = parquet_paths(snapshot, "reasoning", "reasoning")
    logger.info("Local reasoning scan: %d parquet files", len(files))
    n = 0
    for path in files:
        table = pq.read_table(path, columns=["id", "full_reasoning"])
        for cid, fr in zip(table["id"].to_pylist(), table["full_reasoning"].to_pylist()):
            n += 1
            if not (isinstance(fr, str) and fr.strip()):
                continue
            cid = str(cid or "").strip()
            ct = chart_type_index.get(cid)
            if ct in want_types:
                pools[ct].append(cid)
        if n % 200_000 == 0:
            sizes = ", ".join(f"{k}={len(v)}" for k, v in sorted(pools.items()))
            logger.info("  reasoning rows scanned: %d (%s)", n, sizes)
    for t, ids in pools.items():
        logger.info("Reasoning pool %s: %d rows", t, len(ids))
        if not ids:
            raise RuntimeError(f"No reasoning rows found for chart type {t!r}")
    return pools


def sample_id_lists(
    pools: dict[str, list[str]],
    *,
    num_samples: int,
    seed: int,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for t, ids in pools.items():
        unique = list(dict.fromkeys(ids))
        k = min(num_samples, len(unique))
        if k < num_samples:
            logger.warning("Only %d %s rows available; sampling %d", len(unique), t, k)
        rng = random.Random(seed if t == "bar" else seed + 1)
        out[t] = rng.sample(unique, k)
    return out


def fetch_reasoning_rows_by_ids(snapshot: Path, wanted_ids: set[str]) -> dict[str, dict]:
    import pyarrow.parquet as pq

    logger.info("Fetching %d reasoning rows from local parquet…", len(wanted_ids))
    found: dict[str, dict] = {}
    for path in parquet_paths(snapshot, "reasoning", "reasoning"):
        if len(found) == len(wanted_ids):
            break
        table = pq.read_table(path)
        cols = table.column_names
        for i in range(len(table)):
            row = {c: table[c][i].as_py() for c in cols}
            cid = str(row.get("id", "")).strip()
            if cid in wanted_ids and cid not in found:
                found[cid] = row
    if len(found) != len(wanted_ids):
        raise RuntimeError(f"Missing {len(wanted_ids) - len(found)} reasoning rows in local cache")
    return found


def fetch_core_fields_by_ids(snapshot: Path, wanted_ids: set[str]) -> dict[str, dict]:
    import pyarrow.parquet as pq

    logger.info("Fetching core fields for %d ids from local parquet…", len(wanted_ids))
    found: dict[str, dict] = {}
    for path in parquet_paths(snapshot, "core", "core"):
        if len(found) == len(wanted_ids):
            break
        table = pq.read_table(path, columns=["id", "code", "csv", "summary"])
        for i in range(len(table)):
            cid = str(table["id"][i].as_py() or "").strip()
            if cid in wanted_ids and cid not in found:
                found[cid] = {
                    "code": table["code"][i].as_py() or "",
                    "csv": table["csv"][i].as_py() or "",
                    "summary": table["summary"][i].as_py() or "",
                }
    missing = len(wanted_ids) - len(found)
    if missing:
        logger.warning("%d ids missing from local core (code/csv/summary left empty)", missing)
    return found


def rows_for_export_from_ids(
    ids: list[str],
    reasoning_by_id: dict[str, dict],
    core_by_id: dict[str, dict],
    *,
    chart_type_label: str,
) -> list[dict]:
    rows: list[dict] = []
    for run_i, cid in enumerate(ids):
        raw = dict(reasoning_by_id[cid])
        img = raw.get("image")
        if not isinstance(img, PILImage.Image):
            pil = image_to_pil(img)
            if pil is not None:
                raw["image"] = pil
        core = core_by_id.get(cid, {})
        row = {
            "run_index": run_i,
            "id": cid,
            "complex_question": raw.get("complex_question", ""),
            "full_reasoning": full_reasoning_json(raw),
            "think": str(raw.get("think") or ""),
            "answer": raw.get("answer", ""),
            "code": core.get("code", ""),
            "csv": core.get("csv", ""),
            "summary": core.get("summary", ""),
            "chart_type": chart_type_label,
            "hf_config": "reasoning",
            "hf_split": "train",
            "image": raw.get("image"),
        }
        rows.append(row)
    return rows


def export_type_bundle_from_ids(
    *,
    chart_type: str,
    ids: list[str],
    reasoning_by_id: dict[str, dict],
    core_by_id: dict[str, dict],
    out_dir: Path,
) -> Path:
    label = "Bar Chart" if chart_type == "bar" else "Pie Chart"
    rows = rows_for_export_from_ids(ids, reasoning_by_id, core_by_id, chart_type_label=label)
    export_test_bundle(rows, out_dir)
    meta = {
        "source": CHARTNET_DATASET,
        "hf_config": "reasoning",
        "hf_split": "train",
        "chart_type": label,
        "num_samples": len(rows),
        "seed_note": "bar uses --seed, pie uses --seed+1",
        "fields": [
            "id",
            "complex_question",
            "think",
            "answer",
            "full_reasoning",
            "code",
            "csv",
            "summary",
            "chart_type",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d samples)", out_dir, len(rows))
    return out_dir


def export_reasoning_by_type_local(
    snapshot: Path,
    *,
    output_parent: Path,
    num_samples: int = 100,
    seed: int = 42,
    chart_type: ChartTypeFilter = "both",
) -> Path:
    output_parent.mkdir(parents=True, exist_ok=True)
    want = {"bar", "pie"} if chart_type == "both" else {chart_type}

    chart_type_index = build_chart_type_index_local(snapshot)
    pools = collect_reasoning_id_pools_local(snapshot, chart_type_index, want_types=want)
    picked = sample_id_lists(pools, num_samples=num_samples, seed=seed)

    wanted_ids: set[str] = set()
    for ids in picked.values():
        wanted_ids.update(ids)
    reasoning_by_id = fetch_reasoning_rows_by_ids(snapshot, wanted_ids)
    core_by_id = fetch_core_fields_by_ids(snapshot, wanted_ids)

    bundles: dict[str, str] = {}
    for ct in sorted(picked):
        sub = "bar_charts" if ct == "bar" else "pie_charts"
        out = output_parent / sub
        export_type_bundle_from_ids(
            chart_type=ct,
            ids=picked[ct],
            reasoning_by_id=reasoning_by_id,
            core_by_id=core_by_id,
            out_dir=out,
        )
        bundles[ct] = str(out)

    manifest = {
        "description": "Random samples from ChartNet HF reasoning/train filtered by chart type",
        "source": CHARTNET_DATASET,
        "hf_config": "reasoning",
        "hf_split": "train",
        "local_snapshot": str(snapshot),
        "num_samples_per_type": num_samples,
        "seed_bar": seed,
        "seed_pie": seed + 1,
        "bundles": bundles,
    }
    manifest_path = output_parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Done → %s", manifest_path)
    return output_parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export ChartNet HF reasoning samples filtered to bar and/or pie charts."
    )
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-parent",
        type=Path,
        default=Path("data/complex_reasoning_100"),
    )
    p.add_argument(
        "--chart-type",
        choices=("bar", "pie", "both"),
        default="both",
        help="Export one chart type or both (default: both).",
    )
    p.add_argument(
        "--local-snapshot",
        type=Path,
        default=None,
        help="Path to HF ChartNet snapshot (auto-detected from HF_HOME if omitted).",
    )
    p.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = args.local_snapshot
    if snapshot is None:
        snapshot = find_local_snapshot(args.hf_home)
    if snapshot is None:
        raise SystemExit(
            "ChartNet not found in local HF cache. Download first:\n"
            "  huggingface-cli download ibm-granite/ChartNet --repo-type dataset "
            "--include 'core/*' --include 'reasoning/*'"
        )
    logger.info("Using local snapshot: %s", snapshot)
    export_reasoning_by_type_local(
        snapshot,
        output_parent=args.output_parent,
        num_samples=args.num_samples,
        seed=args.seed,
        chart_type=args.chart_type,
    )


if __name__ == "__main__":
    main()
