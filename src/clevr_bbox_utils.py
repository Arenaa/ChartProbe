"""CLEVR 2D bounding boxes for scene JSON.

Official CLEVR val scenes do not include erobic's Blender-rendered ``x/y/width/height``
fields. For clevr_100 we load ground-truth boxes from ``CLEVR_val_question2bboxes.json``
(union per object index across all val questions for each image). When those fields are
present on scene objects (see erobic/clevr-dataset-gen render_images.py), they are
converted directly. Otherwise we estimate boxes with the larchen/clevr-vqa heuristic
(https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py).

References:
  - https://github.com/erobic/clevr-dataset-gen/.../render_images.py (Box + camera_view_bounds_2d)
  - pixel_coords = [px, py, depth]; the third value is NOT a pixel radius.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CLEVR_IMG_W = 480
CLEVR_IMG_H = 320


def xyxy_px_to_xyxy_1000(
    x0: float, y0: float, x1: float, y1: float, *, img_w: int = CLEVR_IMG_W, img_h: int = CLEVR_IMG_H
) -> list[int]:
    x0, x1 = sorted((max(0, min(img_w, x0)), max(0, min(img_w, x1))))
    y0, y1 = sorted((max(0, min(img_h, y0)), max(0, min(img_h, y1))))
    return [
        max(0, min(1000, round(1000 * x0 / img_w))),
        max(0, min(1000, round(1000 * y0 / img_h))),
        max(0, min(1000, round(1000 * x1 / img_w))),
        max(0, min(1000, round(1000 * y1 / img_h))),
    ]


def erobic_xywh_to_xyxy_1000(
    x: float, y: float, width: float, height: float, *, img_w: int = CLEVR_IMG_W, img_h: int = CLEVR_IMG_H
) -> list[int]:
    """Convert erobic render_images.py Box fields (top-left xywh) to xyxy_1000."""
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    return xyxy_px_to_xyxy_1000(x, y, x + width, y + height, img_w=img_w, img_h=img_h)


def larchen_bbox_xyxy_px(obj: dict, scene: dict) -> tuple[float, float, float, float]:
    """Estimate object bbox in pixel space (top-left origin) from scene metadata."""
    x, y, _depth = obj["pixel_coords"]
    x3, y3, z1 = obj["3d_coords"]
    cos_theta, sin_theta, _ = scene["directions"]["right"]

    x1 = x3 * cos_theta + y3 * sin_theta
    y1 = x3 * (-sin_theta) + y3 * cos_theta

    height_d = 6.9 * z1 * (15 - y1) / 2.0
    height_u = height_d
    width_l = height_d
    width_r = height_d

    if obj["shape"] == "cylinder":
        d = 9.4 + y1
        h = 6.4
        s = z1
        height_u *= (s * (h / d + 1)) / ((s * (h / d + 1)) - (s * (h - s) / d))
        height_d = height_u * (h - s + d) / (h + s + d)
        width_l *= 11 / (10 + y1)
        width_r = width_l

    if obj["shape"] == "cube":
        height_u *= 1.3 * 10 / (10 + y1)
        height_d = height_u
        width_l = height_u
        width_r = height_u

    return x - width_l, y - height_d, x + width_r, y + height_u


def clevr_element_bbox_xyxy_1000(
    obj: dict,
    scene: dict,
    *,
    img_w: int = CLEVR_IMG_W,
    img_h: int = CLEVR_IMG_H,
) -> list[int]:
    """Return xyxy_1000 bbox for one CLEVR object."""
    if all(k in obj for k in ("x", "y", "width", "height")):
        return erobic_xywh_to_xyxy_1000(
            float(obj["x"]),
            float(obj["y"]),
            float(obj["width"]),
            float(obj["height"]),
            img_w=img_w,
            img_h=img_h,
        )
    x0, y0, x1, y1 = larchen_bbox_xyxy_px(obj, scene)
    return xyxy_px_to_xyxy_1000(x0, y0, x1, y1, img_w=img_w, img_h=img_h)


def load_val_image_bboxes(
    questions_path: Path,
    question2bboxes_path: Path,
) -> dict[str, dict[int, list[float]]]:
    """Union erobic xywh bboxes per object index for each val image."""
    _, _, by_image = load_val_bbox_resources(questions_path, question2bboxes_path)
    return by_image


def load_val_bbox_resources(
    questions_path: Path,
    question2bboxes_path: Path,
) -> tuple[list[dict], dict[str, dict], dict[str, dict[int, list[float]]]]:
    """Load val questions, per-question bboxes, and per-image bbox unions."""
    question2bboxes = json.loads(question2bboxes_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    by_image: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for i, q in enumerate(questions):
        bboxes = question2bboxes.get(str(i), {})
        if not bboxes:
            continue
        img = q["image_filename"]
        for obj_idx, xywh in bboxes.items():
            by_image[img][int(obj_idx)] = xywh
    return questions, question2bboxes, dict(by_image)


def grounded_object_meta(question: dict, object_index: int) -> dict:
    go0 = question["grounded_objects"][0]
    return go0.get(str(object_index)) or go0[object_index]


def pick_val_bbox_target_for_image(
    image_filename: str,
    questions: list[dict],
    question2bboxes: dict[str, dict],
) -> dict:
    """Pick one val question/object pair with erobic bbox GT for G-BBOX-ELEMENT."""
    candidates: list[tuple[tuple[int, int, int], int, dict, dict]] = []
    for i, q in enumerate(questions):
        if q["image_filename"] != image_filename:
            continue
        bboxes = question2bboxes.get(str(i), {})
        if not bboxes:
            continue
        sort_key = (len(bboxes), q.get("question_index", i), i)
        candidates.append((sort_key, i, q, bboxes))
    if not candidates:
        raise ValueError(f"no val bbox question for {image_filename}")

    _, q_idx, question, bboxes = min(candidates, key=lambda item: item[0])
    obj_idx = min(int(k) for k in bboxes)
    obj_meta = grounded_object_meta(question, obj_idx)
    xywh = bboxes[str(obj_idx)]
    return {
        "val_question_index": q_idx,
        "val_question": question["question"],
        "object_index": obj_idx,
        "target_label": f"{obj_meta['color']} {obj_meta['shape']}",
        "xywh": xywh,
        "bbox_source": "erobic_question2bboxes",
        "n_grounded_objects": len(bboxes),
    }


def build_val_bbox_targets_by_image(
    image_filenames: list[str],
    questions: list[dict],
    question2bboxes: dict[str, dict],
) -> dict[str, dict]:
    return {
        image_filename: pick_val_bbox_target_for_image(image_filename, questions, question2bboxes)
        for image_filename in image_filenames
    }


def erobic_bbox_xyxy_1000_for_object(
    object_index: int,
    bbox_map: dict[int, list[float]],
    *,
    img_w: int = CLEVR_IMG_W,
    img_h: int = CLEVR_IMG_H,
) -> list[int]:
    xywh = bbox_map[object_index]
    return erobic_xywh_to_xyxy_1000(
        float(xywh[0]),
        float(xywh[1]),
        float(xywh[2]),
        float(xywh[3]),
        img_w=img_w,
        img_h=img_h,
    )
