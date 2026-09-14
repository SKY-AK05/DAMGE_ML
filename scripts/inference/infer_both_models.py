"""
infer_both_models.py
-----------------------
Runs ALL THREE trained models (YOLOv11m-seg, fine-tuned Mask2Former, and
Mask R-CNN) on the same folder of images, so you can directly compare their
real-world output side by side. Results are saved into separate subfolders
under one output directory.

Output structure:
    test_result/
        yolo/
            annotated_images/   (images with polygon overlays + labels)
            annotations.xml     (CVAT format)
        mask2former/
            annotated_images/
            annotations.xml
        maskrcnn/
            annotated_images/
            annotations.xml

Usage:
    python infer_both_models.py --input ./test --output ./test_result
    # (all three weight paths default to the standard locations used by
    #  run_full_pipeline.sh -- override with --yolo_weights / --mask2former_weights
    #  / --maskrcnn_weights if yours are somewhere else)

    # Skip any model you don't want to run:
    python infer_both_models.py --input ./test --output ./test_result --skip_maskrcnn
"""

import os
import argparse
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
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 40:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_ratio * peri, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        return None
    return pts


def suppress_overlapping_masks(items, iou_thresh=0.50):
    """
    Suppresses redundant overlapping detections.
    items: list of dicts each containing 'mask', 'score', ...
    Returns filtered list of items sorted by confidence.
    """
    if not items:
        return []
    sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
    kept = []
    for candidate in sorted_items:
        cand_mask = candidate["mask"]
        cand_area = cand_mask.sum()
        if cand_area == 0:
            continue
        overlap = False
        for prev in kept:
            prev_mask = prev["mask"]
            inter = np.logical_and(cand_mask, prev_mask).sum()
            if inter > 0:
                union = np.logical_or(cand_mask, prev_mask).sum()
                iou = inter / union if union > 0 else 0
                if iou > iou_thresh:
                    overlap = True
                    break
        if not overlap:
            kept.append(candidate)
    return kept


def points_to_cvat_str(pts):
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def box_to_polygon(box):
    """Fallback for degenerate masks -> rectangle as a 4-point polygon."""
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def build_label_meta_xml(labels_el, class_names):
    for name in class_names:
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = name
        ET.SubElement(label_el, "type").text = "polygon"
        ET.SubElement(label_el, "attributes")


def new_cvat_root(task_name, class_names):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "mode").text = "annotation"
    labels_el = ET.SubElement(task, "labels")
    build_label_meta_xml(labels_el, class_names)
    ET.SubElement(meta, "dumped").text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
    return root


def save_xml(root, path):
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml_str)


def run_yolo(input_dir, output_dir, weights_path, class_names, conf):
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print(" Running YOLO inference")
    print("=" * 60)
    print(f"[*] Loading YOLO weights from {weights_path} ...")
    model = YOLO(weights_path)

    overlay_dir = os.path.join(output_dir, "annotated_images")
    os.makedirs(overlay_dir, exist_ok=True)
    root = new_cvat_root("yolo_predictions", class_names)

    image_files = sorted([f for f in os.listdir(input_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    for img_id, fname in enumerate(image_files):
        img_path = os.path.join(input_dir, fname)
        print(f"[{img_id+1}/{len(image_files)}] {fname}")

        results = model.predict(source=img_path, conf=conf, save=False, verbose=False)
        r = results[0]

        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        image_el = ET.SubElement(root, "image", id=str(img_id), name=fname, width=str(w), height=str(h))

        detections_found = 0
        if r.masks is not None:
            for mask_xy, cls_id, score in zip(r.masks.xy, r.boxes.cls.cpu().numpy().astype(int),
                                                r.boxes.conf.cpu().numpy()):
                if cls_id >= len(class_names):
                    continue
                label_name = class_names[cls_id]
                pts = mask_xy.astype(np.int32)
                if len(pts) < 3:
                    continue

                detections_found += 1
                color = COLORS[cls_id % len(COLORS)]
                cv2.polylines(image, [pts], True, color, 2)
                label_pos = tuple(pts[0])
                cv2.putText(image, f"{label_name} {score:.2f}", label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                ET.SubElement(image_el, "polygon",
                              label=label_name, source="yolo11m-seg",
                              occluded="0", points=points_to_cvat_str(pts), z_order="0")

        print(f"    -> {detections_found} part(s) detected")
        cv2.imwrite(os.path.join(overlay_dir, fname), image)

    save_xml(root, os.path.join(output_dir, "annotations.xml"))
    print(f"[OK] YOLO results -> {output_dir}")


def run_mask2former(input_dir, output_dir, weights_path, class_names, conf):
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    print("\n" + "=" * 60)
    print(" Running Mask2Former inference")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading Mask2Former weights from {weights_path} (device: {device}) ...")
    # Load processor: try local dir first (new runs), fall back to HF base model (old runs
    # where processor was not saved alongside the model weights).
    _M2F_BASE_MODEL = "facebook/mask2former-swin-tiny-coco-instance"
    _proc_config = os.path.join(weights_path, "preprocessor_config.json")
    if os.path.exists(_proc_config):
        processor = Mask2FormerImageProcessor.from_pretrained(weights_path)
    else:
        print(f"[!] preprocessor_config.json not found in weights dir — falling back to base model processor ({_M2F_BASE_MODEL})")
        print(f"    (Tip: re-train to auto-save the processor, or copy preprocessor_config.json into the weights dir)")
        processor = Mask2FormerImageProcessor.from_pretrained(_M2F_BASE_MODEL)
        # Also save it into the weights dir so future runs don't need the fallback
        try:
            processor.save_pretrained(weights_path)
            print(f"[*] Saved preprocessor_config.json → {weights_path}")
        except Exception as _e:
            print(f"[!] Could not save processor to weights dir: {_e}")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(weights_path).to(device).eval()

    overlay_dir = os.path.join(output_dir, "annotated_images")
    os.makedirs(overlay_dir, exist_ok=True)
    root = new_cvat_root("mask2former_predictions", class_names)

    image_files = sorted([f for f in os.listdir(input_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    for img_id, fname in enumerate(image_files):
        img_path = os.path.join(input_dir, fname)
        print(f"[{img_id+1}/{len(image_files)}] {fname}")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, target_sizes=[(h, w)], threshold=conf
        )[0]

        image_el = ET.SubElement(root, "image", id=str(img_id), name=fname, width=str(w), height=str(h))
        overlay = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        raw_items = []
        for seg in result.get("segments_info", []):
            label_id = seg["label_id"]
            score = seg.get("score", 1.0)
            if label_id >= len(class_names) or score < conf:
                continue

            mask = (result["segmentation"] == seg["id"]).cpu().numpy()
            pts = mask_to_polygon(mask)
            if pts is None:
                continue

            raw_items.append({
                "label_id": label_id,
                "score": score,
                "mask": mask,
                "pts": pts,
                "label_name": class_names[label_id]
            })

        items = suppress_overlapping_masks(raw_items, iou_thresh=0.50)
        detections_found = len(items)

        for item in items:
            label_id = item["label_id"]
            label_name = item["label_name"]
            score = item["score"]
            pts = item["pts"]

            color = COLORS[label_id % len(COLORS)]
            cv2.polylines(overlay, [pts.astype(np.int32)], True, color, 2)
            label_pos = tuple(pts[0].astype(int))
            cv2.putText(overlay, f"{label_name} {score:.2f}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ET.SubElement(image_el, "polygon",
                          label=label_name, source="mask2former_finetuned",
                          occluded="0", points=points_to_cvat_str(pts), z_order="0")

        print(f"    -> {detections_found} part(s) detected")
        cv2.imwrite(os.path.join(overlay_dir, fname), overlay)

    save_xml(root, os.path.join(output_dir, "annotations.xml"))
    print(f"[OK] Mask2Former results -> {output_dir}")


def run_maskrcnn(input_dir, output_dir, weights_path, class_names, conf):
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    print("\n" + "=" * 60)
    print(" Running Mask R-CNN inference")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading Mask R-CNN weights from {weights_path} (device: {device}) ...")

    num_classes = len(class_names) + 1  # +1 for background, matches training setup
    model = maskrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device).eval()

    overlay_dir = os.path.join(output_dir, "annotated_images")
    os.makedirs(overlay_dir, exist_ok=True)
    root = new_cvat_root("maskrcnn_predictions", class_names)

    image_files = sorted([f for f in os.listdir(input_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    for img_id, fname in enumerate(image_files):
        img_path = os.path.join(input_dir, fname)
        print(f"[{img_id+1}/{len(image_files)}] {fname}")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        image_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

        with torch.no_grad():
            output = model([image_tensor])[0]

        image_el = ET.SubElement(root, "image", id=str(img_id), name=fname, width=str(w), height=str(h))
        overlay = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detections_found = 0

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        masks = output["masks"].cpu().numpy()  # shape [N, 1, H, W], values 0-1

        for box, label, score, mask in zip(boxes, labels, scores, masks):
            if score < conf:
                continue
            class_idx = label - 1  # undo the +1 background offset from training
            if class_idx < 0 or class_idx >= len(class_names):
                continue
            label_name = class_names[class_idx]

            binary_mask = (mask[0] > 0.5)
            pts = mask_to_polygon(binary_mask)
            if pts is None:
                pts = box_to_polygon(box)  # fall back to a box outline if mask is degenerate

            detections_found += 1
            color = COLORS[class_idx % len(COLORS)]
            cv2.polylines(overlay, [pts.astype(np.int32)], True, color, 2)
            label_pos = tuple(pts[0].astype(int))
            cv2.putText(overlay, f"{label_name} {score:.2f}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ET.SubElement(image_el, "polygon",
                          label=label_name, source="maskrcnn_finetuned",
                          occluded="0", points=points_to_cvat_str(pts), z_order="0")

        print(f"    -> {detections_found} part(s) detected")
        cv2.imwrite(os.path.join(overlay_dir, fname), overlay)

    save_xml(root, os.path.join(output_dir, "annotations.xml"))
    print(f"[OK] Mask R-CNN results -> {output_dir}")


def _default_yolo_weights():
    """Find the most recent YOLO weights file."""
    if os.path.exists("last_yolo_weights_path.txt"):
        with open("last_yolo_weights_path.txt") as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path

    runs_dir = Path("runs_comparison")
    if runs_dir.exists():
        weight_files = list(runs_dir.glob("**/yolo11m-seg*/weights/best.pt")) + list(runs_dir.glob("**/yolo11m-seg*/best.pt"))
        if not weight_files:
            weight_files = list(runs_dir.glob("run_*/yolo11*-seg/*/weights/best.pt"))
        
        if weight_files:
            weight_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(weight_files[0])

    mlflow_weights = list(Path("runs/mlflow").glob("**/weights/best.pt")) if Path("runs/mlflow").exists() else []
    if mlflow_weights:
        mlflow_weights.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(mlflow_weights[0])

    if os.path.exists("yolo11m-seg.pt"):
        return "yolo11m-seg.pt"

    return "runs_comparison/yolo11m-seg/weights/best.pt"


def _default_mask2former_weights():
    """Find the most recent Mask2Former weights directory."""
    if os.path.exists("last_m2f_weights_path.txt"):
        with open("last_m2f_weights_path.txt") as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path

    runs_dir = Path("runs_comparison")
    if runs_dir.exists():
        model_dirs = [p.parent for p in runs_dir.glob("**/mask2former*/weights/best/model.safetensors")]
        if not model_dirs:
            model_dirs = [p.parent for p in runs_dir.glob("**/mask2former*/best_model/model.safetensors")]
        if not model_dirs:
            model_dirs = list(runs_dir.glob("**/mask2former*/weights/best")) + list(runs_dir.glob("**/mask2former*/best_model"))
        
        if model_dirs:
            model_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(model_dirs[0])

    runs_dir = Path("runs")
    if runs_dir.exists():
        model_dirs = [p.parent for p in runs_dir.glob("**/mask2former*/model.safetensors")]
        if model_dirs:
            model_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(model_dirs[0])

    return "runs_comparison/mask2former/weights/best"


def _default_maskrcnn_weights():
    """Find the most recent Mask R-CNN weights file."""
    if os.path.exists("last_mrcnn_weights_path.txt"):
        with open("last_mrcnn_weights_path.txt") as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path

    runs_dir = Path("runs_comparison")
    if runs_dir.exists():
        weight_files = list(runs_dir.glob("**/maskrcnn*/weights/best.pt")) + list(runs_dir.glob("**/maskrcnn*/best_model.pt"))
        if weight_files:
            weight_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(weight_files[0])

    runs_dir = Path("runs")
    if runs_dir.exists():
        weight_files = list(runs_dir.glob("**/maskrcnn*/best*.pt"))
        if weight_files:
            weight_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(weight_files[0])

    return "runs_comparison/maskrcnn/weights/best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of images to annotate")
    parser.add_argument("--output", default="./test_result", help="Base output folder")
    parser.add_argument("--yolo_weights", default=_default_yolo_weights())
    parser.add_argument("--mask2former_weights", default=_default_mask2former_weights())
    parser.add_argument("--maskrcnn_weights", default=_default_maskrcnn_weights())
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--skip_yolo", action="store_true")
    parser.add_argument("--skip_mask2former", action="store_true")
    parser.add_argument("--skip_maskrcnn", action="store_true")
    args = parser.parse_args()

    class_names = CARPARTS_SEG_CLASSES
    os.makedirs(args.output, exist_ok=True)
    ran = []

    if not args.skip_yolo:
        if not os.path.exists(args.yolo_weights):
            print(f"\n[!] Skipping YOLO inference: weights file not found at '{args.yolo_weights}'")
        else:
            yolo_out = os.path.join(args.output, "yolo")
            os.makedirs(yolo_out, exist_ok=True)
            run_yolo(args.input, yolo_out, args.yolo_weights, class_names, args.conf)
            ran.append("yolo")

    if not args.skip_mask2former:
        if not os.path.exists(args.mask2former_weights):
            print(f"\n[!] Skipping Mask2Former inference: weights directory not found at '{args.mask2former_weights}'")
        else:
            m2f_out = os.path.join(args.output, "mask2former")
            os.makedirs(m2f_out, exist_ok=True)
            run_mask2former(args.input, m2f_out, args.mask2former_weights, class_names, args.conf)
            ran.append("mask2former")

    if not args.skip_maskrcnn:
        if not os.path.exists(args.maskrcnn_weights):
            print(f"\n[!] Skipping Mask R-CNN inference: weights file not found at '{args.maskrcnn_weights}'")
        else:
            mrcnn_out = os.path.join(args.output, "maskrcnn")
            os.makedirs(mrcnn_out, exist_ok=True)
            run_maskrcnn(args.input, mrcnn_out, args.maskrcnn_weights, class_names, args.conf)
            ran.append("maskrcnn")

    print("\n" + "=" * 60)
    if ran:
        print(f" DONE. Results saved under: {args.output}/" + f"  and  {args.output}/".join(ran))
    else:
        print("[!] No models were executed because no valid weights files were found.")
        print("    Please ensure trained model weights exist or pass valid paths via CLI.")
        import sys
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()