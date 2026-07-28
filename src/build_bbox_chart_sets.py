#!/usr/bin/env python3
"""Export 100 bar + 100 pie charts using only matplotlib/seaborn (bbox-friendly).

Bar charts: sampled from local bar_charts_1000 (mpl/seaborn only).
Pie charts: human_verified matplotlib pies, topped up from HF ChartNet core if needed.

Usage:
  python src/build_bbox_chart_sets.py
  python src/build_bbox_chart_sets.py --seed 42 --skip-hf
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

CHART_TYPE_RE = re.compile(r"ChartType=([^,\n]+)", re.I)
LIBRARY_RE = re.compile(r"Library=([^,\n]+)", re.I)
BBOX_LIBRARIES = frozenset({"matplotlib", "seaborn"})


def chart_library(code: str) -> str | None:
    code_l = (code or "").lower()
    if "plotly" in code_l:
        return "plotly"
    m = LIBRARY_RE.search(code or "")
    if m:
        return m.group(1).strip().lower()
    if "seaborn" in code_l:
        return "seaborn"
    if "matplotlib" in code_l:
        return "matplotlib"
    return None


def is_bbox_friendly(code: str) -> bool:
    return chart_library(code) in BBOX_LIBRARIES


def parse_chart_meta(code: str) -> dict[str, str | None]:
    chart_type = None
    library = None
    m = CHART_TYPE_RE.search(code or "")
    if m:
        chart_type = m.group(1).strip()
    m = LIBRARY_RE.search(code or "")
    if m:
        library = m.group(1).strip()
    else:
        library = chart_library(code)
    return {"chart_type": chart_type, "library": library}


def wedge_count_from_csv(csv_text: str) -> int | None:
    if not csv_text:
        return None
    lines = [ln.strip() for ln in csv_text.strip().splitlines() if ln.strip()]
    return max(0, len(lines) - 1) if len(lines) >= 2 else None


def bar_count_from_csv(csv_text: str) -> int | None:
    return wedge_count_from_csv(csv_text)


def write_bundle(
    *,
    out_dir: Path,
    entries: list[dict],
    rel_image_prefix: str,
    chart_type: str,
    meta_extra: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "images"
    bundle_out = out_dir / "eval_bundle"
    bundle_img = bundle_out / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    bundle_img.mkdir(parents=True, exist_ok=True)

    metadata: list[dict] = []
    samples: list[dict] = []
    bundle_rows: list[dict] = []

    for i, row in enumerate(entries):
        chart_id = row["id"]
        src_img = Path(row["_image_src"])
        dst_name = f"{chart_id}.png"
        dst_img = img_out / dst_name
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_img, bundle_img / dst_name)

        meta = parse_chart_meta(row.get("code", ""))
        count_key = "wedge_count" if chart_type == "Pie Chart" else "bar_count"
        count_val = (
            wedge_count_from_csv(row.get("csv", ""))
            if chart_type == "Pie Chart"
            else bar_count_from_csv(row.get("csv", ""))
        )

        entry = {
            "id": chart_id,
            "chart_id": chart_id,
            "code": row.get("code"),
            "csv": row.get("csv"),
            "summary": row.get("summary"),
            "subtype": f"# Variation: ChartType={meta['chart_type']}, Library={meta['library']}",
            "chart_type": meta["chart_type"] or chart_type,
            "library": meta["library"],
            "image_path": f"{rel_image_prefix}/images/{dst_name}",
            count_key: count_val,
            "bbox_extractable": True,
            "hf_config": row.get("hf_config"),
            "hf_split": row.get("hf_split"),
        }
        metadata.append(entry)
        samples.append({"index": i, **entry})
        bundle_rows.append(
            {
                "index": i,
                "id": chart_id,
                "chart_id": chart_id,
                "image_file": f"images/{dst_name}",
                "question": f"Describe this {chart_type.lower()}.",
                "answer": (row.get("summary") or "")[:500],
                "chart_type": entry["chart_type"],
                "library": entry["library"],
                count_key: count_val,
            }
        )

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as fout:
        for s in samples:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")
    with (bundle_out / "samples.jsonl").open("w", encoding="utf-8") as fout:
        for s in bundle_rows:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

    meta = {
        "chart_type": chart_type,
        "num_charts": len(entries),
        "bbox_extractable": True,
        "libraries_allowed": sorted(BBOX_LIBRARIES),
        **meta_extra,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} {chart_type} charts → {out_dir}")


def is_bar_chart_code(code: str) -> bool:
    c = code or ""
    return "ChartType=Bar Chart" in c or "ChartType=Bar" in c or "bar chart" in c.lower()


def export_bar_charts(*, repo: Path, out_dir: Path, num: int, seed: int) -> None:
    meta_path = repo / "bar_charts_1000" / "metadata.json"
    img_dir = repo / "bar_charts_1000" / "images"
    all_rows = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(all_rows) != 1000:
        raise SystemExit(f"Expected 1000 rows in {meta_path}, found {len(all_rows)}")

    pool = [
        r
        for r in all_rows
        if is_bbox_friendly(r.get("code", ""))
        and (img_dir / f"{r['id']}.png").is_file()
    ]
    if len(pool) < num:
        raise SystemExit(
            f"Only {len(pool)} matplotlib/seaborn bar charts in bar_charts_1000; need {num}"
        )

    rng = random.Random(seed)
    picked = sorted(rng.sample(pool, num), key=lambda r: r["id"])
    for r in picked:
        r["_image_src"] = img_dir / f"{r['id']}.png"
        r["hf_config"] = "core"
        r["hf_split"] = "train"
        r["source_dataset"] = "bar_charts_1000"

    write_bundle(
        out_dir=out_dir,
        entries=picked,
        rel_image_prefix=out_dir.name,
        chart_type="Bar Chart",
        meta_extra={
            "seed": seed,
            "source": str(meta_path),
            "parent_dataset": "bar_charts_1000",
            "parent_num_charts": len(all_rows),
            "pool_size": len(pool),
            "pool_filter": "matplotlib_or_seaborn_only",
        },
    )


def load_hv_pie_pool(repo: Path) -> list[dict]:
    src = repo / "human_verified_test" / "samples.jsonl"
    img_dir = repo / "human_verified_test" / "images"
    rows: list[dict] = []
    with src.open(encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            code = row.get("code") or ""
            if "ChartType=Pie Chart" not in code or not is_bbox_friendly(code):
                continue
            img = img_dir / Path(row["image_file"]).name
            if not img.is_file():
                continue
            row["_image_src"] = img
            row["hf_config"] = row.get("hf_config", "human_verified")
            row["hf_split"] = row.get("hf_split", "test")
            rows.append(row)
    return rows


def load_hf_core_pie_pool(*, need: int, exclude_ids: set[str], seed: int) -> list[dict]:
    import sys

    from datasets import load_dataset

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from test_bundle import image_to_pil

    tmp_img = Path(__file__).resolve().parent.parent / "_tmp_bbox_hf_images"
    tmp_img.mkdir(parents=True, exist_ok=True)

    found: list[dict] = []
    for row in load_dataset("ibm-granite/ChartNet", "core", split="train", streaming=True):
        cid = str(row.get("id", ""))
        if not cid or cid in exclude_ids:
            continue
        code = row.get("code") or ""
        if "ChartType=Pie Chart" not in code or not is_bbox_friendly(code):
            continue
        pil = image_to_pil(row.get("image"))
        if pil is None:
            continue
        out_path = tmp_img / f"{cid}.png"
        pil.save(out_path)
        found.append(
            {
                "id": cid,
                "code": code,
                "csv": row.get("csv", ""),
                "summary": row.get("summary", ""),
                "_image_src": out_path,
                "hf_config": "core",
                "hf_split": "train",
            }
        )
        if len(found) >= need:
            break

    return found


def export_pie_charts(
    *, repo: Path, out_dir: Path, num: int, seed: int, skip_hf: bool
) -> None:
    pool = load_hv_pie_pool(repo)
    rng = random.Random(seed)

    if len(pool) >= num:
        picked = sorted(rng.sample(pool, num), key=lambda r: r["id"])
    else:
        if skip_hf:
            raise SystemExit(
                f"Only {len(pool)} matplotlib pie charts in human_verified; "
                f"need {num}. Re-run without --skip-hf to pull from HF core."
            )
        hv_take = sorted(pool, key=lambda r: r["id"])
        need = num - len(hv_take)
        exclude = {r["id"] for r in hv_take}
        print(f"human_verified matplotlib pies: {len(hv_take)}; fetching {need} from HF core…")
        hf_rows = load_hf_core_pie_pool(need=need, exclude_ids=exclude, seed=seed)
        if len(hf_rows) < need:
            raise SystemExit(f"HF core returned only {len(hf_rows)} matplotlib pies; need {need}")
        picked = sorted(hv_take + hf_rows, key=lambda r: r["id"])

    write_bundle(
        out_dir=out_dir,
        entries=picked,
        rel_image_prefix=out_dir.name,
        chart_type="Pie Chart",
        meta_extra={
            "seed": seed,
            "human_verified_pool": len(pool),
            "hf_core_supplement": max(0, num - len(pool)),
        },
    )


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--bar-out", type=Path, default=repo / "data" / "bar_charts_100")
    ap.add_argument("--pie-out", type=Path, default=repo / "pie_charts_100")
    ap.add_argument(
        "--skip-hf",
        action="store_true",
        help="Do not fetch extra pie charts from HF core (fails if human_verified < num).",
    )
    ap.add_argument("--bars-only", action="store_true")
    ap.add_argument("--pies-only", action="store_true")
    args = ap.parse_args()

    if not args.pies_only:
        export_bar_charts(
            repo=repo, out_dir=args.bar_out.resolve(), num=args.num, seed=args.seed
        )
    if not args.bars_only:
        export_pie_charts(
            repo=repo,
            out_dir=args.pie_out.resolve(),
            num=args.num,
            seed=args.seed,
            skip_hf=args.skip_hf,
        )


if __name__ == "__main__":
    main()
