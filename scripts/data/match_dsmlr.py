#!/usr/bin/env python3
"""
match_dsmlr.py
--------------
"Matcher" for the DSMLR Car-Parts-Segmentation dataset.

Reads COCO JSON annotations from datasets/external/dsmlr/annotations/
and converts each image's polygon segmentations to YOLO-format normalised
polygon .txt files, mapped to raw/'s 23-class taxonomy.

DSMLR category mapping to 23-class target taxonomy
----------------------------------------------------
DSMLR is LEFT/RIGHT-specific; it has no generic back_door / front_door /
back_light / front_light / object.  Mapping:

  DSMLR name          ->  Target class ID : target name
  ──────────────────────────────────────────────────────
  _background_        ->  SKIP   (not a foreground class)
  back_bumper         ->  0  : back_bumper
  back_glass          ->  2  : back_glass
  back_left_door      ->  3  : back_left_door
  back_left_light     ->  4  : back_left_light
  back_right_door     ->  6  : back_right_door
  back_right_light    ->  7  : back_right_light
  front_bumper        ->  8  : front_bumper
  front_glass         ->  10 : front_glass
  front_left_door     ->  11 : front_left_door
  front_left_light    ->  12 : front_left_light
  front_right_door    ->  14 : front_right_door
  front_right_light   ->  15 : front_right_light
  hood                ->  16 : hood
  left_mirror         ->  17 : left_mirror
  right_mirror        ->  19 : right_mirror
  tailgate            ->  20 : tailgate
  trunk               ->  21 : trunk
  wheel               ->  22 : wheel

Classes NOT in DSMLR (present in 23-class taxonomy):
  back_door (1), back_light (5), front_door (9), front_light (13), object (18)
  -> These simply won't appear in DSMLR's contribution. No action needed.

Output layout:
  datasets/matched/dsmlr/images/{train,val,test}/   (copied from external/)
  datasets/matched/dsmlr/labels/{train,val,test}/   (converted from COCO JSON)
  datasets/matched/dsmlr/data.yaml
"""

import json
import shutil
from pathlib import Path
from pycocotools import mask as mask_utils
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "datasets" / "external" / "dsmlr"
DST = PROJECT_ROOT / "datasets" / "matched"  / "dsmlr"

# ── Category mapping ────────────────────────────────────────────────────────
# DSMLR category NAME -> target class ID  (None = skip)
DSMLR_TO_TARGET = {
    "_background_":     None,
    "back_bumper":      0,
    "back_glass":       2,
    "back_left_door":   3,
    "back_left_light":  4,
    "back_right_door":  6,
    "back_right_light": 7,
    "front_bumper":     8,
    "front_glass":      10,
    "front_left_door":  11,
    "front_left_light": 12,
    "front_right_door": 14,
    "front_right_light":15,
    "hood":             16,
    "left_mirror":      17,
    "right_mirror":     19,
    "tailgate":         20,
    "trunk":            21,
    "wheel":            22,
}

# 32-class taxonomy — matches combined_data/unified_categories.yaml
NAMES = {
    0:  "wheel",                   1:  "wheel_cap",
    2:  "front_left_light",        3:  "front_right_light",
    4:  "door_glass",              5:  "left_mirror",
    6:  "right_mirror",            7:  "front_bumper",
    8:  "grille",                  9:  "front_left_door",
    10: "front_right_door",        11: "hood",
    12: "pillar",                  13: "back_left_door",
    14: "back_right_door",         15: "front_glass",
    16: "front_left_fender",       17: "front_right_fender",
    18: "back_left_light",         19: "back_right_light",
    20: "rear_quarter_panel",      21: "roof",
    22: "side_skirt_rocker_panel", 23: "fog_lamp",
    24: "back_bumper",             25: "indicator",
    26: "number_plate",            27: "back_glass",
    28: "tailgate",                29: "fuel_lid",
    30: "antenna",                 31: "trunk",
}


def seg_to_yolo_polygon(segmentation, img_w: int, img_h: int):
    """
    Convert a COCO segmentation (list of [x,y,x,y,...] polygon(s) or RLE)
    to a flat list of normalised [x/w, y/h, ...] coordinates.
    Returns None if segmentation is empty or un-parseable.
    """
    if not segmentation:
        return None

    if isinstance(segmentation, dict):
        # RLE — decode to mask, then find contour
        rle = segmentation
        mask = mask_utils.decode(rle)
        contours = _mask_to_polygon(mask)
        if not contours:
            return None
        # Use the largest contour
        pts = max(contours, key=lambda c: len(c))
    elif isinstance(segmentation, list):
        # Polygon list — take the largest polygon
        if not segmentation:
            return None
        pts_raw = max(segmentation, key=lambda p: len(p))
        coords = pts_raw
        pts = [(coords[i], coords[i+1]) for i in range(0, len(coords) - 1, 2)]
    else:
        return None

    if len(pts) < 3:
        return None

    norm = []
    for x, y in pts:
        norm.append(round(float(x) / img_w, 6))
        norm.append(round(float(y) / img_h, 6))
    return norm


def _mask_to_polygon(mask: np.ndarray):
    """Extract polygon contours from a binary mask using OpenCV if available."""
    try:
        import cv2
        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for c in contours:
            c = c.squeeze()
            if c.ndim == 2 and len(c) >= 3:
                result.append([(int(p[0]), int(p[1])) for p in c])
        return result
    except ImportError:
        return []


def convert_split(split: str, coco_json: Path, images_src_dir: Path):
    if not coco_json.exists():
        print(f"  [SKIP] {coco_json.name} not found")
        return 0

    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build category id -> target class id map
    cat_id_to_target = {}
    for cat in data["categories"]:
        target = DSMLR_TO_TARGET.get(cat["name"])
        cat_id_to_target[cat["id"]] = target  # None means skip

    # Build image info
    images_by_id = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_dst_dir = DST / "images" / split
    lbl_dst_dir = DST / "labels" / split
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_anns = 0

    for img_info in data["images"]:
        img_id = img_info["id"]
        img_w  = img_info["width"]
        img_h  = img_info["height"]
        fname  = img_info["file_name"]

        # Copy image
        src_img = images_src_dir / fname
        dst_img = img_dst_dir / fname
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Convert annotations to YOLO lines
        anns = anns_by_image.get(img_id, [])
        yolo_lines = []
        for ann in anns:
            target_id = cat_id_to_target.get(ann["category_id"])
            if target_id is None:
                skipped_anns += 1
                continue

            norm_pts = seg_to_yolo_polygon(ann.get("segmentation"), img_w, img_h)
            if norm_pts is None or len(norm_pts) < 6:
                skipped_anns += 1
                continue

            coords_str = " ".join(str(v) for v in norm_pts)
            yolo_lines.append(f"{target_id} {coords_str}")

        # Write label file (even if empty — consistent with having an image)
        lbl_path = lbl_dst_dir / (Path(fname).stem + ".txt")
        with open(lbl_path, "w") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        converted += 1

    print(f"  [{split}] {converted} images converted, {skipped_anns} annotations skipped (background/unmappable)")
    return converted


def main():
    if not SRC.exists():
        print(f"[ERROR] Source not found: {SRC}")
        print("        Run setup_dataset_structure.py first.")
        return

    print("=" * 60)
    print("Matcher: DSMLR  ->  matched/dsmlr/")
    print("=" * 60)
    print("Category mapping (DSMLR -> 23-class taxonomy):")
    for dsmlr_name, target_id in DSMLR_TO_TARGET.items():
        if target_id is None:
            print(f"  {dsmlr_name:<22} ->  SKIP")
        else:
            print(f"  {dsmlr_name:<22} ->  {target_id:>2}: {NAMES[target_id]}")
    print()
    print("Classes NOT in DSMLR (will simply be absent from its contribution):")
    print("  back_door (1), back_light (5), front_door (9), front_light (13), object (18)")
    print()

    total = 0
    for split in ["train", "val", "test"]:
        ann_json = SRC / "annotations" / f"instances_{split}.json"
        img_src  = SRC / "images" / split
        n = convert_split(split, ann_json, img_src)
        total += n

    # Write data.yaml
    names_block = "\n".join(f"  {i}: {n}" for i, n in NAMES.items())
    yaml_content = (
        f"path: {DST.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"names:\n{names_block}\n"
    )
    with open(DST / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n[OK] matched/dsmlr/ complete: {total} images total")


if __name__ == "__main__":
    main()
