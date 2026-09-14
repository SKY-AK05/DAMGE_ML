"""
yolo_to_coco.py
------------------
Converts a YOLO-segmentation formatted dataset (images/ + labels/ with
normalized polygon .txt files, like Ultralytics carparts-seg) into COCO JSON
format, which Mask2Former and Mask R-CNN both expect for training.

Uses `imagesize` for header-only image dimension reads (no full pixel decode)
and ThreadPoolExecutor for parallel processing.

Usage:
    python yolo_to_coco.py --dataset carparts-seg
"""

import os
import sys
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import imagesize
    _USE_IMAGESIZE = True
except ImportError:
    from PIL import Image
    _USE_IMAGESIZE = False

# 32-class taxonomy — matches combined_data/unified_categories.yaml (COCO IDs 1-32 → list index 0-31)
CARPARTS_SEG_CLASSES = [
    "wheel",                   # 0  (COCO 1)
    "wheel_cap",               # 1  (COCO 2)
    "front_left_light",        # 2  (COCO 3)
    "front_right_light",       # 3  (COCO 4)
    "door_glass",              # 4  (COCO 5)
    "left_mirror",             # 5  (COCO 6)
    "right_mirror",            # 6  (COCO 7)
    "front_bumper",            # 7  (COCO 8)
    "grille",                  # 8  (COCO 9)
    "front_left_door",         # 9  (COCO 10)
    "front_right_door",        # 10 (COCO 11)
    "hood",                    # 11 (COCO 12)
    "pillar",                  # 12 (COCO 13)
    "back_left_door",          # 13 (COCO 14)
    "back_right_door",         # 14 (COCO 15)
    "front_glass",             # 15 (COCO 16)
    "front_left_fender",       # 16 (COCO 17)
    "front_right_fender",      # 17 (COCO 18)
    "back_left_light",         # 18 (COCO 19)
    "back_right_light",        # 19 (COCO 20)
    "rear_quarter_panel",      # 20 (COCO 21)
    "roof",                    # 21 (COCO 22)
    "side_skirt_rocker_panel", # 22 (COCO 23)
    "fog_lamp",                # 23 (COCO 24)
    "back_bumper",             # 24 (COCO 25)
    "indicator",               # 25 (COCO 26)
    "number_plate",            # 26 (COCO 27)
    "back_glass",              # 27 (COCO 28)
    "tailgate",                # 28 (COCO 29)
    "fuel_lid",                # 29 (COCO 30)
    "antenna",                 # 30 (COCO 31)
    "trunk",                   # 31 (COCO 32)
]




def _get_image_size(img_path: Path):
    """Read image dimensions from header only — no pixel decode."""
    if _USE_IMAGESIZE:
        return imagesize.get(str(img_path))  # returns (w, h)
    else:
        with Image.open(img_path) as im:
            return im.size  # (w, h)


def yolo_polygon_to_coco(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    class_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    # normalized x,y pairs -> pixel coords
    points = []
    for i in range(0, len(coords), 2):
        points.extend([coords[i] * img_w, coords[i + 1] * img_h])

    xs = points[0::2]
    ys = points[1::2]
    x_min, y_min = min(xs), min(ys)
    width, height = max(xs) - x_min, max(ys) - y_min
    n = len(xs)
    area = 0.5 * abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(n)))

    return class_id, points, [x_min, y_min, width, height], area


def _process_image(args):
    """Worker: read image header + parse its label file. Returns (img_id, entry) or None."""
    img_id, img_path, labels_dir = args
    try:
        w, h = _get_image_size(img_path)
    except Exception:
        return None

    annotations = []
    label_path = labels_dir / (img_path.stem + ".txt")
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                if not line.strip():
                    continue
                result = yolo_polygon_to_coco(line, w, h)
                if result:
                    annotations.append(result)

    return img_id, img_path.name, w, h, annotations


def convert_split(images_dir, labels_dir, class_names, out_json_path, num_workers=8):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not labels_dir.exists():
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return

    image_files = sorted([
        f for f in images_dir.glob("*")
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    # Parallel header reads + label parsing
    tasks = [(img_id, img_path, labels_dir) for img_id, img_path in enumerate(image_files)]
    results = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(_process_image, t): i for i, t in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    # Assemble COCO JSON in deterministic order
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
    }
    ann_id = 0
    images_with_labels = 0
    for entry in results:
        if entry is None:
            continue
        img_id, file_name, w, h, annotations = entry
        coco["images"].append({"id": img_id, "file_name": file_name, "width": w, "height": h})
        if annotations:
            images_with_labels += 1
        for class_id, seg_points, bbox, area in annotations:
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": class_id,
                "segmentation": [seg_points], "bbox": bbox, "area": area, "iscrowd": 0,
            })
            ann_id += 1

    with open(out_json_path, "w") as f:
        json.dump(coco, f)

    if len(coco['annotations']) == 0:
        print(f"[ERROR] {out_json_path} -- {len(coco['images'])} images, 0 annotations")
        print(f"         Check that labels/ directory exists and contains .txt files with valid YOLO polygon format!")
        print(f"         Labels dir: {labels_dir}")
        print(f"         Images with labels: {images_with_labels}/{len(coco['images'])}")
    else:
        print(f"[OK] {out_json_path} -- {len(coco['images'])} images, {len(coco['annotations'])} annotations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True,
        help=(
            "Dataset name (carparts-seg | custom_carparts | combined_carparts) "
            "OR a direct path to a dataset directory containing images/ and labels/."
        )
    )
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Parallel workers for image header reads (default: 8)")
    args = parser.parse_args()

    if not _USE_IMAGESIZE:
        print("[WARN] `imagesize` not installed — falling back to PIL (slower). "
              "Install with: pip install imagesize")

    # Resolve dataset path — accept both short names and full/relative paths
    ds_path = Path(args.dataset)
    if ds_path.is_dir():
        root = ds_path.resolve()
    elif args.dataset in ("carparts-seg", "custom_carparts", "combined_carparts"):
        root = Path(f"datasets/{args.dataset}").resolve()
    elif args.dataset == "dsmlr-carparts":
        print("[i] dsmlr-carparts is already in COCO format — skipping.")
        return
    else:
        root = Path(f"datasets/{args.dataset}").resolve()

    if not root.exists():
        print(f"[ERROR] Dataset directory not found: {root}")
        return

    class_names = CARPARTS_SEG_CLASSES
    for split in ["train", "val", "test"]:
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        if not images_dir.exists():
            print(f"[skip] {images_dir} not found")
            continue
        out_path = root / f"coco_{split}.json"
        convert_split(images_dir, labels_dir, class_names, out_path, args.num_workers)


if __name__ == "__main__":
    main()
