#!/usr/bin/env python3
"""Extend 100-sample test sets to 500 (preserve original 100 + 400 stratified new).

Creates:
  data/bar_charts_500/
  data/pie_charts_500/
  data/clevr_500/
  data/complex_reasoning_500/   (bar + pie complex reasoning)
  data/clevr_500/complex_reasoning/

Each dataset keeps the original 100 samples unchanged and adds 400 new samples
using the same selection recipe (bbox-friendly charts / CLEVR val bbox scenes,
complexity-stratified sampling). Also writes eval_bundle_extra400/ (3600 or 400 rows)
for eval-only on the new samples.

Usage:
  python src/extend_probes_to_500.py --dataset bar
  python src/extend_probes_to_500.py --dataset all --skip-clevr-download
  python src/extend_probes_to_500.py --dataset clevr --clevr-images-dir /path/to/CLEVR/val
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clevr_bbox_utils import load_val_bbox_resources
from build_bbox_chart_sets import (
    bar_count_from_csv,
    is_bbox_friendly,
    load_hf_core_pie_pool,
    parse_chart_meta,
    wedge_count_from_csv,
)
from generate_probe_questions import (
    bar_complexity_bin,
    clevr_complexity_bin,
    export_eval_bundle,
    load_chart_table,
    pie_complexity_bin,
    process_chart_dataset,
    process_clevr_dataset,
    validate_questions,
)

CLEVR_VAL_IMAGE_URL = (
    "https://cs.stanford.edu/people/jcjohns/clevr/CLEVR_v1.0/images/val/{name}"
)
TARGET_TOTAL = 500
EXTRA_N = 400
EXTRA_SEED = 43
CHARTNET_PYTHON = os.environ.get(
    "CHARTNET_PYTHON",
    "python3",
)


def _load_export_reasoning():
    path = SCRIPT_DIR / "build_complex_reasoning_sets.py"
    spec = importlib.util.spec_from_file_location("export_reasoning", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def complexity_bin_chart(meta: dict, *, chart_type: str) -> str:
    count_key = "wedge_count" if chart_type == "pie" else "bar_count"
    n = meta.get(count_key)
    if n is None:
        csv_text = meta.get("csv") or ""
        n = wedge_count_from_csv(csv_text) if chart_type == "pie" else bar_count_from_csv(csv_text)
    if chart_type == "pie":
        return pie_complexity_bin(n)
    return bar_complexity_bin(n or 0)


def bin_counts(items: list, bin_fn) -> Counter:
    return Counter(bin_fn(x) for x in items)


def stratified_extra_targets(
    existing: list,
    *,
    extra_n: int,
    bin_fn,
) -> dict[str, int]:
    """Allocate extra_n slots proportional to existing complexity bins."""
    counts = bin_counts(existing, bin_fn)
    total = sum(counts.values()) or 1
    raw = {b: extra_n * counts[b] / total for b in counts}
    targets = {b: int(raw[b]) for b in counts}
    remainder = extra_n - sum(targets.values())
    order = sorted(counts.keys(), key=lambda b: (raw[b] - targets[b]), reverse=True)
    for b in order:
        if remainder <= 0:
            break
        targets[b] += 1
        remainder -= 1
    return targets


def stratified_sample(
    pool: list,
    *,
    targets: dict[str, int],
    bin_fn,
    seed: int,
) -> list:
    by_bin: dict[str, list] = defaultdict(list)
    for item in pool:
        by_bin[bin_fn(item)].append(item)
    rng = random.Random(seed)
    picked: list = []
    for b, k in sorted(targets.items()):
        candidates = by_bin.get(b, [])
        if len(candidates) < k:
            raise RuntimeError(f"Pool too small for bin {b}: need {k}, have {len(candidates)}")
        picked.extend(rng.sample(candidates, k))
    return sorted(picked, key=lambda x: x.get("id") or x.get("image_filename") or "")


def copy_tree_preserve(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def load_metadata(dataset_dir: Path) -> list[dict]:
    return json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))


def write_metadata(dataset_dir: Path, rows: list[dict]) -> None:
    (dataset_dir / "metadata.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def chart_row_from_pool(row: dict, *, chart_type: str, rel_prefix: str) -> dict:
    meta = parse_chart_meta(row.get("code", ""))
    count_key = "wedge_count" if chart_type == "pie" else "bar_count"
    count_val = (
        wedge_count_from_csv(row.get("csv", ""))
        if chart_type == "pie"
        else bar_count_from_csv(row.get("csv", ""))
    )
    chart_id = row["id"]
    return {
        "id": chart_id,
        "chart_id": chart_id,
        "code": row.get("code"),
        "csv": row.get("csv"),
        "summary": row.get("summary"),
        "subtype": f"# Variation: ChartType={meta['chart_type']}, Library={meta['library']}",
        "chart_type": meta["chart_type"] or ("Pie Chart" if chart_type == "pie" else "Bar Chart"),
        "library": meta["library"],
        "image_path": f"{rel_prefix}/images/{chart_id}.png",
        count_key: count_val,
        "bbox_extractable": True,
        "hf_config": row.get("hf_config"),
        "hf_split": row.get("hf_split"),
    }


def add_chart_images(rows: list[dict], *, src_fn) -> None:
    img_dir = None
    for row in rows:
        chart_id = row["id"]
        src = src_fn(row)
        if img_dir is None:
            img_dir = src.parent.parent / "images" if "images" not in str(src.parent) else src.parent
        dst = img_dir / f"{chart_id}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_extra_eval_bundle(dataset_dir: Path, *, id_key: str, preserved_ids: set[str]) -> int:
    src = dataset_dir / "eval_bundle" / "samples.jsonl"
    if not src.is_file():
        return 0
    out_dir = dataset_dir / "eval_bundle_extra400"
    out_dir.mkdir(parents=True, exist_ok=True)
    link = out_dir / "images"
    src_images = dataset_dir / "eval_bundle" / "images"
    if not link.exists() and src_images.is_dir():
        link.symlink_to(src_images.resolve())
    elif not link.exists():
        link.symlink_to((dataset_dir / "images").resolve())

    kept: list[str] = []
    with src.open(encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            sid = row.get(id_key) or row.get("chart_id") or row.get("scene_id")
            if sid not in preserved_ids:
                kept.append(line.rstrip("\n"))
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as fout:
        for i, line in enumerate(kept):
            row = json.loads(line)
            row["index"] = i
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(kept)


def extend_bar_charts(*, seed: int = EXTRA_SEED) -> dict:
    src_dir = REPO / "data" / "bar_charts_100"
    dst_dir = REPO / "data" / "bar_charts_500"
    copy_tree_preserve(src_dir, dst_dir)

    existing = load_metadata(dst_dir)
    preserved_ids = {m["id"] for m in existing}
    targets = stratified_extra_targets(existing, extra_n=EXTRA_N, bin_fn=lambda m: complexity_bin_chart(m, chart_type="bar"))

    meta_path = REPO / "data" / "bar_charts_1000" / "metadata.json"
    img_dir = REPO / "data" / "bar_charts_1000" / "images"
    pool = []
    for r in json.loads(meta_path.read_text(encoding="utf-8")):
        if r["id"] in preserved_ids:
            continue
        if not is_bbox_friendly(r.get("code", "")):
            continue
        if not (img_dir / f"{r['id']}.png").is_file():
            continue
        pool.append(r)

    picked = stratified_sample(pool, targets=targets, bin_fn=lambda r: bar_complexity_bin(bar_count_from_csv(r.get("csv", "")) or 0), seed=seed)
    new_rows = [chart_row_from_pool(r, chart_type="bar", rel_prefix=dst_dir.name) for r in picked]
    for r, src in zip(new_rows, picked):
        shutil.copy2(img_dir / f"{src['id']}.png", dst_dir / "images" / f"{r['id']}.png")

    merged = existing + new_rows
    write_metadata(dst_dir, merged)

    meta = json.loads((dst_dir / "meta.json").read_text(encoding="utf-8"))
    meta.update(
        {
            "num_charts": len(merged),
            "total_questions": len(merged) * 9,
            "extended_from": "bar_charts_100",
            "preserved_sample_ids": sorted(preserved_ids),
            "extra_sample_ids": [r["id"] for r in new_rows],
            "extra_seed": seed,
            "extra_stratified_targets": targets,
        }
    )
    (dst_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"dataset": dst_dir.name, "total": len(merged), "extra": len(new_rows), "preserved": len(preserved_ids)}


def extend_pie_charts(*, seed: int = EXTRA_SEED) -> dict:
    src_dir = REPO / "data" / "pie_charts_100"
    dst_dir = REPO / "data" / "pie_charts_500"
    copy_tree_preserve(src_dir, dst_dir)

    existing = load_metadata(dst_dir)
    preserved_ids = {m["id"] for m in existing}
    targets = stratified_extra_targets(existing, extra_n=EXTRA_N, bin_fn=lambda m: complexity_bin_chart(m, chart_type="pie"))

    exclude = set(preserved_ids)
    hf_rows = load_hf_core_pie_pool(need=800, exclude_ids=exclude, seed=seed)
    pool = [r for r in hf_rows if r["id"] not in exclude]
    picked = stratified_sample(
        pool,
        targets=targets,
        bin_fn=lambda r: pie_complexity_bin(wedge_count_from_csv(r.get("csv", ""))),
        seed=seed,
    )

    new_rows = []
    for r in picked:
        dst_img = dst_dir / "images" / f"{r['id']}.png"
        shutil.copy2(r["_image_src"], dst_img)
        new_rows.append(chart_row_from_pool(r, chart_type="pie", rel_prefix=dst_dir.name))

    merged = existing + new_rows
    write_metadata(dst_dir, merged)

    meta = json.loads((dst_dir / "meta.json").read_text(encoding="utf-8"))
    meta.update(
        {
            "num_charts": len(merged),
            "total_questions": len(merged) * 9,
            "extended_from": "pie_charts_100",
            "preserved_sample_ids": sorted(preserved_ids),
            "extra_sample_ids": [r["id"] for r in new_rows],
            "extra_seed": seed,
            "extra_stratified_targets": targets,
            "hf_core_supplement_extra": len(new_rows),
        }
    )
    (dst_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"dataset": dst_dir.name, "total": len(merged), "extra": len(new_rows), "preserved": len(preserved_ids)}


def fix_pie_bad_charts(dst_dir: Path, *, seed: int = EXTRA_SEED) -> list[str]:
    """Replace pie charts that fail bbox/question generation with new HF candidates."""
    rows = load_metadata(dst_dir)
    ann_dir = dst_dir / "annotations"
    img_dir = dst_dir / "images"
    exclude = {m["id"] for m in rows}
    hf_rows = load_hf_core_pie_pool(need=50, exclude_ids=exclude, seed=seed + 99)
    pool = [r for r in hf_rows if r["id"] not in exclude]

    replaced: list[str] = []
    for i, row in enumerate(list(rows)):
        cid = row["id"]
        ann_path = ann_dir / f"{cid}.json"
        needs_replace = not ann_path.is_file()
        if not needs_replace:
            try:
                load_chart_table(row.get("csv", ""))
            except Exception:
                needs_replace = True
        if not needs_replace:
            continue
        if not pool:
            raise RuntimeError(f"No HF pie replacement for {cid}")
        repl = pool.pop(0)
        new_row = chart_row_from_pool(repl, chart_type="pie", rel_prefix=dst_dir.name)
        rows[i] = new_row
        for p in (img_dir / f"{cid}.png", ann_path):
            if p.is_file():
                p.unlink()
        shutil.copy2(repl["_image_src"], img_dir / f"{repl['id']}.png")
        replaced.append(f"{cid}->{repl['id']}")
        print(f"  replaced bad pie {cid} -> {repl['id']}")

    if replaced:
        write_metadata(dst_dir, rows)
        meta = json.loads((dst_dir / "meta.json").read_text(encoding="utf-8"))
        preserved = set(meta.get("preserved_sample_ids") or [])
        meta["extra_sample_ids"] = [m["id"] for m in rows if m["id"] not in preserved]
        meta["num_charts"] = len(rows)
        meta["total_questions"] = len(rows) * 9
        (dst_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return replaced


def chart_eval_complete(dataset_name: str, *, expected_samples: int = TARGET_TOTAL) -> bool:
    p = REPO / "data" / dataset_name / "eval_bundle" / "samples.jsonl"
    return p.is_file() and sum(1 for _ in p.open()) >= expected_samples * 9


def download_clevr_image(name: str, dst: Path) -> bool:
    url = CLEVR_VAL_IMAGE_URL.format(name=name)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            dst.write_bytes(resp.read())
        return True
    except Exception:
        return False


def extend_clevr(
    *,
    seed: int = EXTRA_SEED,
    clevr_images_dir: Path | None = None,
    skip_download: bool = False,
) -> dict:
    src_dir = REPO / "data" / "clevr_100"
    dst_dir = REPO / "data" / "clevr_500"
    copy_tree_preserve(src_dir, dst_dir)

    existing_images = sorted(p.name for p in (dst_dir / "images").glob("*.png"))
    preserved_ids = {n.replace(".png", "") for n in existing_images}

    scenes_all = json.loads((REPO / "data" / "clevr_100" / "scenes" / "CLEVR_val_scenes.json").read_text())["scenes"]
    scene_by_img = {s["image_filename"]: s for s in scenes_all}
    val_q = REPO / "CLEVR_val_questions_new.json"
    val_q2b = REPO / "CLEVR_val_question2bboxes.json"
    _, _, by_image = load_val_bbox_resources(val_q, val_q2b)

    existing_meta = [{"image_filename": n, "scene_id": n.replace(".png", ""), "n_objects": len(scene_by_img[n]["objects"])} for n in existing_images if n in scene_by_img]
    targets = stratified_extra_targets(
        existing_meta,
        extra_n=EXTRA_N,
        bin_fn=lambda m: clevr_complexity_bin(m["n_objects"]),
    )

    pool = []
    for img_name, scene in scene_by_img.items():
        if img_name in existing_images:
            continue
        if img_name not in by_image:
            continue
        n = len(scene["objects"])
        if n < 3:
            continue
        pool.append({"image_filename": img_name, "scene_id": img_name.replace(".png", ""), "n_objects": n, "scene": scene})

    picked = stratified_sample(
        pool,
        targets=targets,
        bin_fn=lambda m: clevr_complexity_bin(m["n_objects"]),
        seed=seed,
    )

    img_out = dst_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for item in picked:
        name = item["image_filename"]
        dst = img_out / name
        if clevr_images_dir is not None:
            src = clevr_images_dir / name
            if src.is_file():
                shutil.copy2(src, dst)
                continue
            failed.append(name)
            continue
        if dst.is_file():
            continue
        if skip_download:
            failed.append(name)
            continue
        if not download_clevr_image(name, dst):
            failed.append(name)

    if failed:
        raise RuntimeError(
            f"Missing {len(failed)} CLEVR images (e.g. {failed[:3]}). "
            "Provide --clevr-images-dir or allow download (omit --skip-clevr-download)."
        )

    extra_ids = {item["scene_id"] for item in picked}
    scene_list = json.loads((src_dir / "scene_list.json").read_text(encoding="utf-8"))
    scene_list.extend(
        {
            "image_index": int(item["image_filename"].split("_")[-1].replace(".png", "")),
            "image_filename": item["image_filename"],
            "scene_id": item["scene_id"],
            "n_objects": item["n_objects"],
        }
        for item in picked
    )
    scene_list.sort(key=lambda x: x["image_filename"])
    (dst_dir / "scene_list.json").write_text(json.dumps(scene_list, indent=2), encoding="utf-8")

    meta = json.loads((dst_dir / "meta.json").read_text(encoding="utf-8"))
    meta.update(
        {
            "num_scenes": len(scene_list),
            "extended_from": "clevr_100",
            "preserved_sample_ids": sorted(preserved_ids),
            "extra_sample_ids": sorted(extra_ids),
            "extra_seed": seed,
            "extra_stratified_targets": targets,
        }
    )
    (dst_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"dataset": dst_dir.name, "total": len(scene_list), "extra": len(picked), "preserved": len(preserved_ids)}


def run_postprocess_chart(dataset_name: str) -> None:
    ds = REPO / "data" / dataset_name
    subprocess.run(
        [CHARTNET_PYTHON, str(SCRIPT_DIR / "extract_chart_bboxes.py"), "--dataset", str(ds)],
        check=True,
    )
    subprocess.run(
        [CHARTNET_PYTHON, str(SCRIPT_DIR / "generate_probe_questions.py"), "--dataset", dataset_name],
        check=True,
    )
    meta = json.loads((ds / "meta.json").read_text(encoding="utf-8"))
    preserved = set(meta.get("preserved_sample_ids") or [])
    n_extra = write_extra_eval_bundle(ds, id_key="chart_id", preserved_ids=preserved)
    print(f"  eval_bundle_extra400: {n_extra} rows")


def run_postprocess_clevr() -> None:
    ds = REPO / "data" / "clevr_500"
    subprocess.run(
        [CHARTNET_PYTHON, str(SCRIPT_DIR / "extract_clevr_bboxes.py"), "--dataset-dir", str(ds)],
        check=True,
    )
    scenes_subset = ds / "scenes" / f"CLEVR_val_{len(list((ds / 'images').glob('*.png')))}_scenes.json"
    subprocess.run(
        [
            CHARTNET_PYTHON,
            str(SCRIPT_DIR / "generate_probe_questions.py"),
            "--dataset",
            "clevr_500",
            "--clevr-scenes",
            str(scenes_subset if scenes_subset.is_file() else ds / "scenes" / "CLEVR_val_scenes.json"),
        ],
        check=True,
    )
    meta = json.loads((ds / "meta.json").read_text(encoding="utf-8"))
    preserved = set(meta.get("preserved_sample_ids") or [])
    if not preserved:
        preserved = {p.stem for p in (REPO / "data" / "clevr_100" / "images").glob("*.png")}
        meta["preserved_sample_ids"] = sorted(preserved)
        meta.setdefault("extended_from", "clevr_100")
        (ds / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    n_extra = write_extra_eval_bundle(ds, id_key="scene_id", preserved_ids=preserved)
    subprocess.run(
        [CHARTNET_PYTHON, str(SCRIPT_DIR / "build_clevr_complex_reasoning.py"), "--dataset-dir", str(ds), "--out-dir", str(ds / "complex_reasoning")],
        check=True,
    )
    cr_extra = ds / "complex_reasoning_extra400"
    src_cr = ds / "complex_reasoning" / "samples.jsonl"
    kept: list[dict] = []
    if src_cr.is_file():
        cr_extra.mkdir(parents=True, exist_ok=True)
        kept = []
        with src_cr.open(encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                if row.get("scene_id") not in preserved:
                    kept.append(row)
        with (cr_extra / "samples.jsonl").open("w", encoding="utf-8") as fout:
            for i, row in enumerate(kept):
                row["run_index"] = i
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  eval_bundle_extra400: {n_extra} rows; complex_reasoning_extra400: {len(kept) if src_cr.is_file() else 0}")


def extend_hf_reasoning(*, seed: int = 42) -> dict:
    er = _load_export_reasoning()
    snapshot = er.find_local_snapshot(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))))
    if snapshot is None:
        raise RuntimeError("ChartNet HF snapshot not found for complex reasoning export")

    src_parent = REPO / "data" / "complex_reasoning_100"
    dst_parent = REPO / "data" / "complex_reasoning_500"
    dst_parent.mkdir(parents=True, exist_ok=True)

    chart_type_index = er.build_chart_type_index_local(snapshot)
    pools = er.collect_reasoning_id_pools_local(snapshot, chart_type_index, want_types={"bar", "pie"})

    summaries = {}
    for ct, sub in (("bar", "bar_charts"), ("pie", "pie_charts")):
        src_jsonl = src_parent / sub / "samples.jsonl"
        preserved = []
        for line in src_jsonl.open(encoding="utf-8"):
            preserved.append(json.loads(line)["id"])
        preserved_set = set(preserved)
        targets = stratified_extra_targets(
            [{"id": x} for x in preserved],
            extra_n=EXTRA_N,
            bin_fn=lambda _: "all",
        )
        pool_ids = [i for i in pools[ct] if i not in preserved_set]
        rng = random.Random(seed if ct == "bar" else seed + 1)
        extra = rng.sample(pool_ids, targets["all"])
        all_ids = preserved + extra

        reasoning_by_id = er.fetch_reasoning_rows_by_ids(snapshot, set(all_ids))
        core_by_id = er.fetch_core_fields_by_ids(snapshot, set(all_ids))
        out = dst_parent / sub
        er.export_type_bundle_from_ids(
            chart_type=ct,
            ids=all_ids,
            reasoning_by_id=reasoning_by_id,
            core_by_id=core_by_id,
            out_dir=out,
        )
        extra_dir = dst_parent / f"{sub}_extra400"
        extra_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        with (out / "samples.jsonl").open(encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                if row["id"] not in preserved_set:
                    rows.append(row)
        with (extra_dir / "samples.jsonl").open("w", encoding="utf-8") as fout:
            for i, row in enumerate(rows):
                row["run_index"] = i
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries[sub] = {"total": len(all_ids), "extra": len(extra), "preserved": len(preserved)}

    manifest = {
        "description": "ChartNet HF complex reasoning — 500 per chart type (100 preserved + 400 new)",
        "extended_from": "complex_reasoning_100",
        "num_samples_per_type": TARGET_TOTAL,
        "seed_bar": seed,
        "seed_pie": seed + 1,
        "bundles": summaries,
    }
    (dst_parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("bar", "pie", "clevr", "reasoning", "all"), default="all")
    ap.add_argument("--seed", type=int, default=EXTRA_SEED)
    ap.add_argument("--skip-postprocess", action="store_true")
    ap.add_argument("--skip-clevr-download", action="store_true")
    ap.add_argument("--clevr-images-dir", type=Path, default=None)
    ap.add_argument("--skip-reasoning", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip datasets already at 500 samples with complete eval bundles; fix bad pies before postprocess.",
    )
    args = ap.parse_args()

    summaries = []
    if args.dataset in ("bar", "all"):
        if args.resume and chart_eval_complete("bar_charts_500"):
            print("bar_charts_500 already complete — skipping")
        else:
            print("Extending bar_charts_100 → bar_charts_500 …")
            summaries.append(extend_bar_charts(seed=args.seed))
            if not args.skip_postprocess:
                run_postprocess_chart("bar_charts_500")

    if args.dataset in ("pie", "all"):
        pie_dst = REPO / "data" / "pie_charts_500"
        if args.resume and chart_eval_complete("pie_charts_500"):
            print("pie_charts_500 already complete — skipping")
        elif args.resume and len(load_metadata(pie_dst)) >= TARGET_TOTAL:
            print("pie_charts_500 metadata ready — fixing bad charts + postprocess …")
            fix_pie_bad_charts(pie_dst, seed=args.seed)
            if not args.skip_postprocess:
                run_postprocess_chart("pie_charts_500")
        else:
            print("Extending pie_charts_100 → pie_charts_500 …")
            summaries.append(extend_pie_charts(seed=args.seed))
            fix_pie_bad_charts(pie_dst, seed=args.seed)
            if not args.skip_postprocess:
                run_postprocess_chart("pie_charts_500")

    if args.dataset in ("clevr", "all"):
        clevr_dst = REPO / "data" / "clevr_500"
        n_imgs = len(list((clevr_dst / "images").glob("*.png"))) if (clevr_dst / "images").is_dir() else 0
        if args.resume and n_imgs >= TARGET_TOTAL and chart_eval_complete("clevr_500"):
            print("clevr_500 already complete — skipping")
        elif args.resume and n_imgs >= TARGET_TOTAL:
            print("clevr_500 images ready — postprocess only …")
            if not args.skip_postprocess:
                run_postprocess_clevr()
        else:
            print("Extending clevr_100 → clevr_500 …")
            summaries.append(
                extend_clevr(
                    seed=args.seed,
                    clevr_images_dir=args.clevr_images_dir,
                    skip_download=args.skip_clevr_download,
                )
            )
            if not args.skip_postprocess:
                run_postprocess_clevr()

    if args.dataset in ("reasoning", "all") and not args.skip_reasoning:
        print("Extending complex_reasoning_100 → complex_reasoning_500 …")
        summaries.append(extend_hf_reasoning(seed=42))

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
