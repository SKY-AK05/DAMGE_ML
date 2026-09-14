"""
infer_mask2former.py
-----------------------
Runs the FINE-TUNED Mask2Former model (the winner from the model comparison)
on a folder of new images and produces:
  - Annotated images with polygon overlays + class labels drawn on
  - A CVAT 1.1 format annotations.xml (same schema used throughout this project)

This is different from auto_annotate_carparts.py (which uses zero-shot
Grounding DINO + SAM2 with no training). This script uses YOUR trained model,
so it only knows the 32 carparts classes it was fine-tuned on -- but
should be far more accurate for those classes, especially left/right telling
apart, since that's exactly what Mask2Former proved better at in testing.

Usage:
    python infer_mask2former.py --input ./test --output ./mask2former_predictions --weights runs_comparison/mask2former/best_model
"""

import os
import argparse
import json
import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

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

# Distinct colors per class for the overlay (one per class)
COLORS = [
    (255, 99, 71),   (60, 179, 113),  (65, 105, 225),  (255, 215, 0),   (218, 112, 214),
    (0, 206, 209),   (255, 140, 0),   (154, 205, 50),  (199, 21, 133),  (30, 144, 255),
    (255, 20, 147),  (34, 139, 34),   (255, 69, 0),    (147, 112, 219), (0, 191, 255),
    (220, 20, 60),   (46, 139, 87),   (255, 165, 0),   (138, 43, 226),  (0, 250, 154),
    (255, 105, 180), (72, 61, 139),   (240, 230, 140), (210, 105, 30),  (127, 255, 0),
    (255, 0, 127),   (0, 128, 128),   (255, 215, 180), (100, 149, 237), (255, 99, 132),
    (50, 205, 50),   (186, 85, 211),
]


def mask_to_polygon(mask, epsilon_ratio=0.005):
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 30:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_ratio * peri, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None
    return pts


def points_to_cvat_str(pts):
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def build_label_meta_xml(labels_el, class_names):
    for name in class_names:
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = name
        ET.SubElement(label_el, "type").text = "polygon"
        ET.SubElement(label_el, "attributes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of images to annotate")
    parser.add_argument("--output", default="./mask2former_predictions")
    parser.add_argument("--weights", default="runs_comparison/mask2former/best_model",
                         help="Path to the fine-tuned Mask2Former checkpoint folder")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{'OK' if device == 'cuda' else 'WARNING'}] Using device: {device}")

    print(f"[*] Loading fine-tuned Mask2Former from {args.weights} ...")
    processor = Mask2FormerImageProcessor.from_pretrained(args.weights)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.weights).to(device).eval()

    class_names = CARPARTS_SEG_CLASSES  # matches what the model was fine-tuned on

    os.makedirs(args.output, exist_ok=True)
    overlay_dir = os.path.join(args.output, "annotated_images")
    os.makedirs(overlay_dir, exist_ok=True)

    # Build CVAT XML root
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "mask2former_predictions"
    ET.SubElement(task, "mode").text = "annotation"
    labels_el = ET.SubElement(task, "labels")
    build_label_meta_xml(labels_el, class_names)
    ET.SubElement(meta, "dumped").text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

    image_files = sorted([f for f in os.listdir(args.input)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not image_files:
        print(f"[ERROR] No images found in {args.input}")
        return

    print(f"[*] Found {len(image_files)} image(s) to annotate.")

    for img_id, fname in enumerate(image_files):
        img_path = os.path.join(args.input, fname)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        print(f"[{img_id+1}/{len(image_files)}] {fname} ({w}x{h})")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        result = processor.post_process_instance_segmentation(
            outputs, target_sizes=[(h, w)], threshold=args.conf
        )[0]

        image_el = ET.SubElement(root, "image", id=str(img_id), name=fname, width=str(w), height=str(h))

        overlay = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detections_found = 0

        for seg in result.get("segments_info", []):
            label_id = seg["label_id"]
            score = seg.get("score", 1.0)
            if label_id >= len(class_names):
                continue
            label_name = class_names[label_id]

            mask = (result["segmentation"] == seg["id"]).cpu().numpy()
            pts = mask_to_polygon(mask)
            if pts is None:
                continue

            detections_found += 1
            color = COLORS[label_id % len(COLORS)]

            # Draw on overlay
            cv2.polylines(overlay, [pts.astype(np.int32)], True, color, 2)
            label_pos = tuple(pts[0].astype(int))
            cv2.putText(overlay, f"{label_name} {score:.2f}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add to CVAT XML
            ET.SubElement(image_el, "polygon",
                          label=label_name, source="mask2former_finetuned",
                          occluded="0", points=points_to_cvat_str(pts), z_order="0")

        print(f"    -> {detections_found} part(s) detected")
        cv2.imwrite(os.path.join(overlay_dir, fname), overlay)

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    xml_path = os.path.join(args.output, "annotations.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"\n[OK] Annotated images -> {overlay_dir}")
    print(f"[OK] CVAT annotations.xml -> {xml_path}")


if __name__ == "__main__":
    main()
