#!/usr/bin/env python3
"""Extract per-object bounding boxes for the 100-scene CLEVR test set.

Reads scene metadata from CLEVR_val_scenes.json and erobic bboxes from
CLEVR_val_question2bboxes.json (union per object index across val questions).
Writes one annotation file per scene under annotations/{scene_id}.json.

Usage:
  python src/extract_clevr_bboxes.py
  python src/extract_clevr_bboxes.py --dataset-dir data/clevr_100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clevr_bbox_utils import (
    CLEVR_IMG_H,
    CLEVR_IMG_W,
    erobic_bbox_xyxy_1000_for_object,
    erobic_xywh_to_xyxy_1000,
    load_val_bbox_resources,
)


def object_label(obj: dict) -> str:
    return f"{obj['color']} {obj['shape']}"


def build_scene_annotation(
    scene: dict,
    bbox_map: dict[int, list[float]] | None = None,
) -> dict:
    scene_id = scene["image_filename"].replace(".png", "")
    elements = []
    erobic_n = 0
    for i, obj in enumerate(scene["objects"]):
        if bbox_map and i in bbox_map:
            xywh = bbox_map[i]
            element_bbox = erobic_bbox_xyxy_1000_for_object(i, bbox_map)
            bbox_source = "erobic_question2bboxes"
            erobic_n += 1
        elif all(k in obj for k in ("x", "y", "width", "height")):
            element_bbox = erobic_xywh_to_xyxy_1000(
                float(obj["x"]),
                float(obj["y"]),
                float(obj["width"]),
                float(obj["height"]),
            )
            bbox_source = "erobic_xywh"
            erobic_n += 1
        else:
            element_bbox = None
            bbox_source = "unavailable"
        entry = {
            "object_index": i,
            "label": object_label(obj),
            "color": obj["color"],
            "shape": obj["shape"],
            "size": obj["size"],
            "material": obj["material"],
            "pixel_coords": obj["pixel_coords"],
            "coords_3d": obj.get("3d_coords"),
            "bbox_source": bbox_source,
        }
        if element_bbox is not None:
            entry["element_bbox"] = element_bbox
        elements.append(entry)

    n = len(elements)
    if n <= 5:
        cbin = "3-5"
    elif n <= 8:
        cbin = "6-8"
    else:
        cbin = "9+"

    return {
        "scene_id": scene_id,
        "image_filename": scene["image_filename"],
        "chart_type": "CLEVR",
        "image_width": CLEVR_IMG_W,
        "image_height": CLEVR_IMG_H,
        "bbox_format": "xyxy_1000",
        "element_kind": "object",
        "complexity_bin": cbin,
        "n_objects": n,
        "n_objects_with_erobic_bbox": erobic_n,
        "elements": elements,
    }


def extract_clevr_bboxes(
    *,
    dataset_dir: Path,
    scenes_path: Path,
    val_questions_path: Path | None = None,
    val_question2bboxes_path: Path | None = None,
    write_subset_scenes: bool = True,
) -> dict:
    repo = Path(__file__).resolve().parent.parent
    val_questions_path = (val_questions_path or repo / "CLEVR_val_questions_new.json").resolve()
    val_question2bboxes_path = (
        val_question2bboxes_path or repo / "CLEVR_val_question2bboxes.json"
    ).resolve()
    if not val_questions_path.is_file():
        raise FileNotFoundError(f"Val questions JSON not found: {val_questions_path}")
    if not val_question2bboxes_path.is_file():
        raise FileNotFoundError(f"Val question2bboxes JSON not found: {val_question2bboxes_path}")

    image_bboxes = load_val_bbox_resources(val_questions_path, val_question2bboxes_path)[2]
    images_dir = dataset_dir / "images"
    image_names = sorted(p.name for p in images_dir.glob("*.png"))
    if not image_names:
        raise FileNotFoundError(f"No images in {images_dir}")

    scenes_data = json.loads(scenes_path.read_text(encoding="utf-8"))
    scene_by_image = {s["image_filename"]: s for s in scenes_data["scenes"]}

    ann_dir = dataset_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    subset_scenes: list[dict] = []
    missing: list[str] = []

    for image_name in image_names:
        scene = scene_by_image.get(image_name)
        if scene is None:
            missing.append(image_name)
            continue
        ann = build_scene_annotation(scene, bbox_map=image_bboxes.get(image_name))
        scene_id = ann["scene_id"]
        (ann_dir / f"{scene_id}.json").write_text(
            json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        subset_scenes.append(scene)

    if missing:
        raise RuntimeError(f"Missing scene metadata for {len(missing)} images: {missing[:5]}")

    subset_path = None
    if write_subset_scenes:
        subset_dir = dataset_dir / "scenes"
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_path = subset_dir / f"CLEVR_val_{len(image_names)}_scenes.json"
        subset_path.write_text(
            json.dumps({"info": scenes_data.get("info", {}), "scenes": subset_scenes}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    total_objects = sum(
        len(json.loads((ann_dir / f"{n.replace('.png', '')}.json").read_text())["elements"])
        for n in image_names
    )
    total_erobic = sum(
        json.loads((ann_dir / f"{n.replace('.png', '')}.json").read_text()).get(
            "n_objects_with_erobic_bbox", 0
        )
        for n in image_names
    )

    meta = {
        "dataset": "clevr_100_bbox",
        "num_scenes": len(image_names),
        "bbox_extractable": True,
        "bboxes_extracted": True,
        "bbox_source": "erobic_question2bboxes",
        "val_questions_json": str(val_questions_path.relative_to(repo))
        if val_questions_path.is_relative_to(repo)
        else str(val_questions_path),
        "val_question2bboxes_json": str(val_question2bboxes_path.relative_to(repo))
        if val_question2bboxes_path.is_relative_to(repo)
        else str(val_question2bboxes_path),
        "annotations_dir": f"{dataset_dir.name}/annotations",
        "scenes_subset_json": f"{dataset_dir.name}/scenes/CLEVR_val_{len(image_names)}_scenes.json" if subset_path else None,
        "questions_json": f"{dataset_dir.name}/questions.json",
        "bbox_questions_json": f"{dataset_dir.name}/dataset/bbox_questions.json",
        "eval_bundle": f"{dataset_dir.name}/eval_bundle/samples.jsonl",
        "eval_bundle_bbox": f"{dataset_dir.name}/eval_bundle_bbox/samples.jsonl",
        "questions_per_scene": 9,
        "bbox_questions_per_scene": 1,
        "total_bbox_questions": len(image_names),
        "bbox_objects_total": total_objects,
        "bbox_objects_with_erobic_gt": total_erobic,
        "bbox_format": "xyxy_1000",
        "image_size": [CLEVR_IMG_W, CLEVR_IMG_H],
    }
    (dataset_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    bbox_questions = _export_bbox_questions(dataset_dir)
    _export_bbox_eval_bundle(dataset_dir, bbox_questions)

    return {
        "scenes": len(image_names),
        "annotations": len(image_names),
        "total_objects": total_objects,
        "total_erobic_objects": total_erobic,
        "bbox_questions": len(bbox_questions),
        "annotations_dir": str(ann_dir),
        "subset_scenes": str(subset_path) if subset_path else None,
        "meta_json": str(dataset_dir / "meta.json"),
    }


def _export_bbox_questions(dataset_dir: Path) -> list[dict]:
    questions_path = dataset_dir / "questions.json"
    if not questions_path.is_file():
        return []
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    bbox_q = [q for q in questions if q.get("template_id") == "G-BBOX-ELEMENT"]
    out_path = dataset_dir / "dataset" / "bbox_questions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bbox_q, ensure_ascii=False, indent=2), encoding="utf-8")
    return bbox_q


def _export_bbox_eval_bundle(dataset_dir: Path, bbox_questions: list[dict]) -> None:
    if not bbox_questions:
        return
    out_dir = dataset_dir / "eval_bundle_bbox"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"
    link = out_dir / "images"
    if link.is_symlink():
        link.unlink()
    elif link.is_dir() and not any(link.iterdir()):
        link.rmdir()
    if not link.exists():
        link.symlink_to(images_dir.resolve())

    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as fout:
        for i, q in enumerate(bbox_questions):
            scene_id = q["scene_id"]
            template_id = q.get("template_id", "")
            image_name = q.get("image_filename") or f"{scene_id}.png"
            row = {
                "index": i,
                "id": f"{scene_id}__{template_id}",
                "scene_id": scene_id,
                "chart_id": scene_id,
                "image_file": f"images/{image_name}",
                "question": q["question"],
                "answer": q["answer"],
                "stage": q.get("stage"),
                "template_id": template_id,
                "answer_type": q.get("answer_type"),
            }
            for extra in (
                "bbox_format",
                "bbox_target",
                "target_label",
                "complexity_bin",
                "n_objects",
                "object_index",
                "bbox_source",
                "val_question_index",
                "val_question",
                "val_n_grounded_objects",
            ):
                if extra in q:
                    row[extra] = q[extra]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=repo / "data" / "clevr_100",
    )
    ap.add_argument(
        "--scenes-json",
        type=Path,
        default=None,
        help="Default: <dataset-dir>/scenes/CLEVR_val_scenes.json",
    )
    ap.add_argument(
        "--val-questions",
        type=Path,
        default=None,
        help="Default: <repo>/CLEVR_val_questions_new.json",
    )
    ap.add_argument(
        "--val-question2bboxes",
        type=Path,
        default=None,
        help="Default: <repo>/CLEVR_val_question2bboxes.json",
    )
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    scenes_path = (args.scenes_json or dataset_dir / "scenes" / "CLEVR_val_scenes.json").resolve()
    if not scenes_path.is_file():
        raise SystemExit(f"Scenes JSON not found: {scenes_path}")

    summary = extract_clevr_bboxes(
        dataset_dir=dataset_dir,
        scenes_path=scenes_path,
        val_questions_path=args.val_questions,
        val_question2bboxes_path=args.val_question2bboxes,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
