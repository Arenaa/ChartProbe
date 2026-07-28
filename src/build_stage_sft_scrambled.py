#!/usr/bin/env python3
"""Export scrambled-supervision SFT splits (format-control / skill ablation).

Holds probe questions, answer *format*, chart split, and row counts fixed while
destroying the image→answer skill signal by permuting answers across charts
within the same template_id (same stage + answer type + format).

Setups (mirrors gold counterparts):
  perception_scrambled              ← perception
  perception_grounding_scrambled    ← perception_grounding

Default: scramble train only (val stays gold). Training does not early-stop on
val (--eval-steps 999999), so val gold only affects optional monitoring.

Usage:
  python src/build_stage_sft_scrambled.py
  python src/build_stage_sft_scrambled.py --setup perception_scrambled
  python src/build_stage_sft_scrambled.py --scramble-val
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_sft_jsonl import stage_counts

SCRAMBLED_SETUPS: dict[str, str] = {
    "perception_scrambled": "perception",
    "perception_grounding_scrambled": "perception_grounding",
}

ANSWER_SWAP_KEYS = ("answer",)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def derangement_indices(n: int, rng: random.Random) -> list[int]:
    """Return a derangement of range(n). For n < 2, identity."""
    if n < 2:
        return list(range(n))
    idx = list(range(n))
    for _ in range(200):
        rng.shuffle(idx)
        if all(idx[i] != i for i in range(n)):
            return idx
    # Guaranteed derangement via cyclic shift when shuffle retries fail.
    return [(i + 1) % n for i in range(n)]


def scramble_answers(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    group_key: str = "template_id",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Permute answer fields within each group_key; keep image/question fixed."""
    rng = random.Random(seed)
    out = [dict(r) for r in rows]
    group_to_indices: dict[str, list[int]] = {}
    for i, r in enumerate(out):
        g = str(r.get(group_key) or r.get("answer_type") or "unknown")
        group_to_indices.setdefault(g, []).append(i)

    map_rows: list[dict[str, Any]] = []
    residual_same = 0
    for g, indices in sorted(group_to_indices.items()):
        donors = derangement_indices(len(indices), rng)
        for local_i, row_i in enumerate(indices):
            donor_i = indices[donors[local_i]]
            original = out[row_i].get("answer")
            donated = rows[donor_i].get("answer")
            for key in ANSWER_SWAP_KEYS:
                if key in rows[donor_i]:
                    out[row_i][key] = rows[donor_i][key]
            same = original == donated
            if same:
                residual_same += 1
            map_rows.append(
                {
                    "index": row_i,
                    "id": out[row_i].get("id"),
                    "chart_id": out[row_i].get("chart_id"),
                    "template_id": out[row_i].get("template_id"),
                    "answer_type": out[row_i].get("answer_type"),
                    "donor_chart_id": rows[donor_i].get("chart_id"),
                    "donor_id": rows[donor_i].get("id"),
                    "answer_original": original,
                    "answer_scrambled": donated,
                    "residual_same_as_original": bool(same),
                }
            )

    stats = {
        "n_rows": len(out),
        "n_groups": len(group_to_indices),
        "group_sizes": {g: len(ix) for g, ix in sorted(group_to_indices.items())},
        "residual_same_as_original": residual_same,
        "residual_same_rate": round(residual_same / max(len(out), 1), 6),
        "donor_always_different_row": True,
        "map_rows": map_rows,
    }
    return out, stats


def _answer_multiset(rows: list[dict[str, Any]]) -> Counter:
    return Counter(json.dumps(r.get("answer"), sort_keys=True, ensure_ascii=False) for r in rows)


def verify_scramble(
    gold: list[dict[str, Any]],
    scrambled: list[dict[str, Any]],
    *,
    group_key: str = "template_id",
) -> dict[str, Any]:
    if len(gold) != len(scrambled):
        raise ValueError(f"row count mismatch: gold={len(gold)} scrambled={len(scrambled)}")

    fixed_ok = True
    for g, s in zip(gold, scrambled):
        for k in ("chart_id", "image", "question", "template_id", "answer_type", "stage", "id"):
            if g.get(k) != s.get(k):
                fixed_ok = False
                break

    # Global answer multiset must match (permutation).
    global_multiset_ok = _answer_multiset(gold) == _answer_multiset(scrambled)

    # Per-template multiset must match.
    gold_by: dict[str, list] = {}
    scr_by: dict[str, list] = {}
    for r in gold:
        gold_by.setdefault(str(r.get(group_key)), []).append(r)
    for r in scrambled:
        scr_by.setdefault(str(r.get(group_key)), []).append(r)
    per_group_ok = True
    per_group: dict[str, bool] = {}
    for g in sorted(set(gold_by) | set(scr_by)):
        ok = _answer_multiset(gold_by.get(g, [])) == _answer_multiset(scr_by.get(g, []))
        per_group[g] = ok
        per_group_ok = per_group_ok and ok

    different_chart_donors = 0
    residual_same = 0
    for g, s in zip(gold, scrambled):
        if g.get("answer") == s.get("answer"):
            residual_same += 1
        # donor identity is not stored on the row; checked via map file separately

    return {
        "fixed_fields_preserved": fixed_ok,
        "global_answer_multiset_preserved": global_multiset_ok,
        "per_template_answer_multiset_preserved": per_group_ok,
        "per_template_ok": per_group,
        "residual_same_as_original": residual_same,
        "residual_same_rate": round(residual_same / max(len(gold), 1), 6),
        "n_rows": len(gold),
        "different_chart_donors_note": different_chart_donors,
    }


def export_scrambled_setup(
    *,
    setup: str,
    source_setup: str,
    sft_root: Path,
    scramble_seed: int,
    scramble_val: bool,
) -> dict[str, Any]:
    src_dir = sft_root / source_setup
    train_gold = load_jsonl(src_dir / "train.jsonl")
    val_gold = load_jsonl(src_dir / "val.jsonl")
    source_meta = json.loads((src_dir / "meta.json").read_text(encoding="utf-8"))

    train_scr, train_stats = scramble_answers(train_gold, seed=scramble_seed)
    if scramble_val:
        val_scr, val_stats = scramble_answers(val_gold, seed=scramble_seed + 1)
    else:
        val_scr = [dict(r) for r in val_gold]
        val_stats = {
            "n_rows": len(val_scr),
            "scrambled": False,
            "note": "val left gold (default)",
        }

    train_verify = verify_scramble(train_gold, train_scr)
    if scramble_val:
        val_verify = verify_scramble(val_gold, val_scr)
    else:
        val_verify = {"scrambled": False, "identical_to_gold": train_gold is not val_gold}

    if not train_verify["fixed_fields_preserved"]:
        raise RuntimeError(f"{setup}: fixed fields not preserved after scramble")
    if not train_verify["per_template_answer_multiset_preserved"]:
        raise RuntimeError(f"{setup}: per-template answer multiset broken")

    # Donor must always be a different chart within each map entry.
    bad_donors = [
        m
        for m in train_stats["map_rows"]
        if m["chart_id"] == m["donor_chart_id"]
    ]
    if bad_donors:
        raise RuntimeError(f"{setup}: {len(bad_donors)} self-donors in scramble map")

    out_dir = sft_root / setup
    # write_jsonl rebuilds image paths from chart_id; keep absolute paths from gold rows.
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as fout:
        for i, row in enumerate(train_scr):
            out = dict(row)
            out["index"] = i
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    with (out_dir / "val.jsonl").open("w", encoding="utf-8") as fout:
        for i, row in enumerate(val_scr):
            out = dict(row)
            out["index"] = i
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    map_path = out_dir / "scramble_map_train.jsonl"
    with map_path.open("w", encoding="utf-8") as fout:
        for m in train_stats["map_rows"]:
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")

    meta = {
        "setup": setup,
        "source_setup": source_setup,
        "control": "scrambled_supervision",
        "scramble_seed": scramble_seed,
        "scramble_val": scramble_val,
        "group_by": "template_id",
        "train_stages": source_meta.get("train_stages"),
        "split_by": source_meta.get("split_by", "chart"),
        "n_train_charts": source_meta.get("n_train_charts"),
        "n_val_charts": source_meta.get("n_val_charts"),
        "seed": source_meta.get("seed"),
        "train_rows": len(train_scr),
        "val_rows": len(val_scr),
        "train_stage_counts": stage_counts(train_scr),
        "val_stage_counts": stage_counts(val_scr),
        "source": source_meta.get("source", "bar_charts_1000"),
        "train_verify": {k: v for k, v in train_verify.items() if k != "per_template_ok"},
        "train_verify_per_template": train_verify.get("per_template_ok"),
        "train_residual_same_rate": train_stats["residual_same_rate"],
        "train_group_sizes": train_stats["group_sizes"],
        "val_stats": {k: v for k, v in val_stats.items() if k != "map_rows"},
        "note": (
            "Answers permuted within template_id across charts. "
            "Questions/images/format unchanged. Residual same-as-original rate "
            "is unavoidable when duplicate answer strings exist in a group "
            "(e.g. binary Yes/No)."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sft-root",
        type=Path,
        default=REPO / "data_other" / "chart_qa_sft" / "stage_experiments",
    )
    ap.add_argument(
        "--setup",
        choices=sorted(SCRAMBLED_SETUPS) + ["all"],
        default="all",
    )
    ap.add_argument(
        "--scramble-seed",
        type=int,
        default=42,
        help="Seed for answer derangement (independent of LoRA train seeds).",
    )
    ap.add_argument(
        "--scramble-val",
        action="store_true",
        help="Also scramble val answers (default: train only).",
    )
    args = ap.parse_args()

    sft_root = args.sft_root.resolve()
    setups = (
        sorted(SCRAMBLED_SETUPS)
        if args.setup == "all"
        else [args.setup]
    )
    summaries = []
    for setup in setups:
        source = SCRAMBLED_SETUPS[setup]
        src = sft_root / source
        if not (src / "train.jsonl").is_file():
            raise FileNotFoundError(
                f"Missing gold source {src}/train.jsonl — run build_stage_sft_splits.py first"
            )
        meta = export_scrambled_setup(
            setup=setup,
            source_setup=source,
            sft_root=sft_root,
            scramble_seed=args.scramble_seed,
            scramble_val=args.scramble_val,
        )
        summaries.append(meta)
        print(json.dumps({k: v for k, v in meta.items() if k != "train_verify_per_template"}, indent=2))

    manifest = {
        "control": "scrambled_supervision",
        "scramble_seed": args.scramble_seed,
        "scramble_val": args.scramble_val,
        "setups": setups,
        "summaries": [
            {
                "setup": s["setup"],
                "source_setup": s["source_setup"],
                "train_rows": s["train_rows"],
                "val_rows": s["val_rows"],
                "train_residual_same_rate": s["train_residual_same_rate"],
            }
            for s in summaries
        ],
    }
    (sft_root / "scrambled_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Wrote", sft_root / "scrambled_manifest.json")


if __name__ == "__main__":
    main()
