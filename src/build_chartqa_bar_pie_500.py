#!/usr/bin/env python3
"""Select 500 bar + 500 pie ChartQA samples (HuggingFaceM4/ChartQA) and export bundles.

ChartQA has no chart-type field, so we:
  1) high-precision keyword cues from the question, then
  2) CLIP zero-shot image classification for the rest.

Usage:
  CUDA_VISIBLE_DEVICES=1 python src/build_chartqa_bar_pie_500.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]

PIE_RE = re.compile(r"\b(pie|slice|slices|wedge|wedges)\b", re.I)
BAR_RE = re.compile(r"\b(bar|bars|column|columns)\b", re.I)
LINE_RE = re.compile(r"\b(line|lines|intercept)\b", re.I)

CLIP_LABELS = [
    "a bar chart",
    "a pie chart",
    "a line chart",
    "a scatter plot",
]


def keyword_type(query: str) -> str | None:
    q = query or ""
    pie = bool(PIE_RE.search(q))
    bar = bool(BAR_RE.search(q))
    line = bool(LINE_RE.search(q))
    if pie and not bar and not line:
        return "pie"
    if bar and not pie:
        return "bar"
    return None


def load_clip(device: str):
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    return processor, model


@torch.inference_mode()
def clip_classify_batch(processor, model, images: list[Image.Image], device: str) -> list[str]:
    inputs = processor(
        text=CLIP_LABELS,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    probs = out.logits_per_image.softmax(dim=-1)
    idxs = probs.argmax(dim=-1).tolist()
    confs = probs.max(dim=-1).values.tolist()
    mapping = {0: "bar", 1: "pie", 2: "line", 3: "other"}
    return [(mapping[i], float(c)) for i, c in zip(idxs, confs)]


def answer_text(label) -> str:
    if isinstance(label, list):
        return str(label[0]) if label else ""
    return str(label)


def export_bundle(rows: list[dict], out_dir: Path, chart_type: str) -> None:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    meta = {
        "source": "HuggingFaceM4/ChartQA",
        "chart_type": chart_type,
        "n_samples": len(rows),
        "selection": "keyword + CLIP zero-shot; one QA per unique image when possible",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    with samples_path.open("w", encoding="utf-8") as fout:
        for i, r in enumerate(rows):
            img_name = f"{i:04d}.png"
            img = r["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")
            img.save(img_dir / img_name)
            row = {
                "index": i,
                "id": f"chartqa_{chart_type}_{i:04d}",
                "chart_id": r.get("chart_key", f"{chart_type}_{i:04d}"),
                "image_file": f"images/{img_name}",
                "question": r["query"],
                "answer": answer_text(r["label"]),
                "human_or_machine": r.get("human_or_machine"),
                "split": r.get("split"),
                "chart_type": chart_type,
                "type_source": r.get("type_source"),
                "clip_confidence": r.get("clip_confidence"),
                "source_dataset": "HuggingFaceM4/ChartQA",
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--clip-min-conf", type=float, default=0.45)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO / "data" / "chartqa_bar_pie_500",
    )
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # Collect candidates: prefer unique images.
    by_type: dict[str, list[dict]] = {"bar": [], "pie": []}
    seen_img: dict[str, set[str]] = {"bar": set(), "pie": set()}

    # Pass 1: keyword matches (high precision)
    for split in args.splits:
        ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
        for idx, row in enumerate(tqdm(ds, desc=f"keyword:{split}")):
            ctype = keyword_type(row["query"])
            if ctype not in by_type:
                continue
            # approximate image identity via pixel hash of a tiny thumbnail
            img = row["image"].convert("RGB")
            thumb = img.resize((32, 32))
            key = thumb.tobytes().hex()
            if key in seen_img[ctype]:
                continue
            if len(by_type[ctype]) >= args.n_per_type:
                continue
            seen_img[ctype].add(key)
            by_type[ctype].append(
                {
                    "query": row["query"],
                    "label": row["label"],
                    "image": img,
                    "human_or_machine": row["human_or_machine"],
                    "split": split,
                    "hf_index": idx,
                    "chart_key": f"{split}_{idx}",
                    "type_source": "keyword",
                    "clip_confidence": None,
                    "img_key": key,
                }
            )
        print(
            f"after keywords {split}: bar={len(by_type['bar'])} pie={len(by_type['pie'])}"
        )

    need = {
        "bar": max(0, args.n_per_type - len(by_type["bar"])),
        "pie": max(0, args.n_per_type - len(by_type["pie"])),
    }
    print("still need:", need)

    # Pass 2: CLIP fill for remaining slots
    if need["bar"] or need["pie"]:
        processor, model = load_clip(device)
        clip_pool: dict[str, list[dict]] = {"bar": [], "pie": []}
        for split in args.splits:
            if not need["bar"] and not need["pie"] and all(
                len(clip_pool[t]) >= need[t] for t in need
            ):
                break
            ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
            batch_imgs: list[Image.Image] = []
            batch_meta: list[dict] = []
            for idx, row in enumerate(tqdm(ds, desc=f"clip:{split}")):
                # skip already keyword-labeled queries of opposite/ambiguous types? still classify unknowns
                if keyword_type(row["query"]) is not None:
                    continue
                img = row["image"].convert("RGB")
                thumb = img.resize((32, 32))
                key = thumb.tobytes().hex()
                batch_imgs.append(img)
                batch_meta.append(
                    {
                        "query": row["query"],
                        "label": row["label"],
                        "human_or_machine": row["human_or_machine"],
                        "split": split,
                        "hf_index": idx,
                        "chart_key": f"{split}_{idx}",
                        "img_key": key,
                        "image": img,
                    }
                )
                if len(batch_imgs) < args.batch_size:
                    continue
                preds = clip_classify_batch(processor, model, batch_imgs, device)
                for meta, (ctype, conf) in zip(batch_meta, preds):
                    if ctype not in ("bar", "pie"):
                        continue
                    if conf < args.clip_min_conf:
                        continue
                    if meta["img_key"] in seen_img[ctype]:
                        continue
                    if len(by_type[ctype]) + len(clip_pool[ctype]) >= args.n_per_type:
                        continue
                    seen_img[ctype].add(meta["img_key"])
                    meta = dict(meta)
                    meta["type_source"] = "clip"
                    meta["clip_confidence"] = round(conf, 4)
                    clip_pool[ctype].append(meta)
                batch_imgs, batch_meta = [], []
                if (
                    len(by_type["bar"]) + len(clip_pool["bar"]) >= args.n_per_type
                    and len(by_type["pie"]) + len(clip_pool["pie"]) >= args.n_per_type
                ):
                    break
            # flush
            if batch_imgs:
                preds = clip_classify_batch(processor, model, batch_imgs, device)
                for meta, (ctype, conf) in zip(batch_meta, preds):
                    if ctype not in ("bar", "pie"):
                        continue
                    if conf < args.clip_min_conf:
                        continue
                    if meta["img_key"] in seen_img[ctype]:
                        continue
                    if len(by_type[ctype]) + len(clip_pool[ctype]) >= args.n_per_type:
                        continue
                    seen_img[ctype].add(meta["img_key"])
                    meta = dict(meta)
                    meta["type_source"] = "clip"
                    meta["clip_confidence"] = round(conf, 4)
                    clip_pool[ctype].append(meta)

        for ctype in ("bar", "pie"):
            take = args.n_per_type - len(by_type[ctype])
            pool = clip_pool[ctype]
            rng.shuffle(pool)
            by_type[ctype].extend(pool[:take])

    # If still short, allow duplicate images with different questions from keyword set
    for ctype in ("bar", "pie"):
        if len(by_type[ctype]) >= args.n_per_type:
            by_type[ctype] = by_type[ctype][: args.n_per_type]
            continue
        print(f"WARNING: only {len(by_type[ctype])} unique {ctype}; topping up with extra Qs")
        extras = []
        for split in args.splits:
            ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
            for idx, row in enumerate(ds):
                if keyword_type(row["query"]) != ctype:
                    continue
                extras.append(
                    {
                        "query": row["query"],
                        "label": row["label"],
                        "image": row["image"].convert("RGB"),
                        "human_or_machine": row["human_or_machine"],
                        "split": split,
                        "hf_index": idx,
                        "chart_key": f"{split}_{idx}_extra",
                        "type_source": "keyword_extra",
                        "clip_confidence": None,
                    }
                )
                if len(by_type[ctype]) + len(extras) >= args.n_per_type:
                    break
            if len(by_type[ctype]) + len(extras) >= args.n_per_type:
                break
        need_n = args.n_per_type - len(by_type[ctype])
        by_type[ctype].extend(extras[:need_n])

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ctype in ("bar", "pie"):
        rows = by_type[ctype]
        rng.shuffle(rows)
        rows = rows[: args.n_per_type]
        out_dir = args.out_root / f"{ctype}_charts_500"
        export_bundle(rows, out_dir, ctype)
        src = Counter(r["type_source"] for r in rows)
        splits = Counter(r["split"] for r in rows)
        summary[ctype] = {
            "n": len(rows),
            "type_source": dict(src),
            "splits": dict(splits),
            "out": str(out_dir),
        }
        print(ctype, summary[ctype])

    (args.out_root / "selection_summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", args.out_root / "selection_summary.json")


if __name__ == "__main__":
    main()
