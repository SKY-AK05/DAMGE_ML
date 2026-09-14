"""
evaluate_confusion_matrix.py
--------------------------------
Runs each trained model on the test set, matches predictions to ground truth
boxes (IoU > 0.5), and builds a confusion matrix. Specifically flags and
prints out front/rear and left/right mixup rates. Target models: YOLO11m-seg,
Mask R-CNN, and Mask2Former.

Usage:
    python evaluate_confusion_matrix.py --model yolo --weights runs_comparison/yolo11m-seg/weights/best.pt --dataset combined_carparts
    python evaluate_confusion_matrix.py --model mask2former --weights runs_comparison/mask2former/weights/best --dataset combined_carparts
    python evaluate_confusion_matrix.py --model maskrcnn --weights runs_comparison/maskrcnn/weights/best.pt --dataset combined_carparts
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

# Pairs we specifically care about mixing up (this is the real question being tested)
CONFUSION_PAIRS_TO_WATCH = [
    ("front_left_door",    "back_left_door"),
    ("front_right_door",   "back_right_door"),
    ("front_left_door",    "front_right_door"),
    ("back_left_door",     "back_right_door"),
    ("front_left_light",   "front_right_light"),
    ("back_left_light",    "back_right_light"),
    ("left_mirror",        "right_mirror"),
    ("front_left_fender",  "front_right_fender"),
    ("front_left_light",   "back_left_light"),
    ("front_right_light",  "back_right_light"),
]



def box_iou(box_a, box_b):
    """box = [x_min, y_min, x_max, y_max]"""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0


def get_predictions_yolo(weights_path, test_images_dir, class_names):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    results = model.predict(source=test_images_dir, conf=0.25, save=False)
    preds_by_image = {}
    for r in results:
        img_name = Path(r.path).name
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
        classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.empty((0,))
        preds_by_image[img_name] = [
            {"box": box.tolist(), "class": class_names[c]} for box, c in zip(boxes, classes)
        ]
    return preds_by_image


def get_predictions_maskrcnn(weights_path, test_images_dir, test_json_path, class_names, conf_threshold=0.5):
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(class_names) + 1  # +1 for background, matches training setup

    model = maskrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device).eval()

    with open(test_json_path) as f:
        coco = json.load(f)

    preds_by_image = {}
    for img_info in coco["images"]:
        img_path = Path(test_images_dir) / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        image_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

        with torch.no_grad():
            output = model([image_tensor])[0]

        preds = []
        for box, label, score in zip(output["boxes"].cpu().numpy(),
                                       output["labels"].cpu().numpy(),
                                       output["scores"].cpu().numpy()):
            if score < conf_threshold:
                continue
            class_idx = label - 1  # undo the +1 background offset from training
            if class_idx < 0 or class_idx >= len(class_names):
                continue
            preds.append({"box": box.tolist(), "class": class_names[class_idx]})
        preds_by_image[img_info["file_name"]] = preds

    return preds_by_image


def get_predictions_mask2former(weights_path, test_images_dir, test_json_path, class_names, conf_threshold=0.5):
    import torch
    from PIL import Image
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Mask2FormerImageProcessor.from_pretrained(weights_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(weights_path).to(device).eval()

    with open(test_json_path) as f:
        coco = json.load(f)

    preds_by_image = {}
    for img_info in coco["images"]:
        img_path = Path(test_images_dir) / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, target_sizes=[(img_info["height"], img_info["width"])]
        )[0]

        preds = []
        for seg in result.get("segments_info", []):
            if seg.get("score", 0.0) < conf_threshold:
                continue
            mask = (result["segmentation"] == seg["id"]).cpu().numpy()
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            preds.append({"box": box, "class": class_names[seg["label_id"]]})
        preds_by_image[img_info["file_name"]] = preds

    return preds_by_image




def load_ground_truth(test_json_path, class_names):
    with open(test_json_path) as f:
        coco = json.load(f)
    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    gt_by_image = {}
    for ann in coco["annotations"]:
        fname = id_to_name[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        box = [x, y, x + w, y + h]
        cls_name = class_names[ann["category_id"]]
        gt_by_image.setdefault(fname, []).append({"box": box, "class": cls_name})
    return gt_by_image


def build_confusion_matrix(gt_by_image, pred_by_image, class_names, iou_threshold=0.5):
    n = len(class_names)
    matrix = np.zeros((n + 1, n + 1), dtype=int)  # +1 row/col for "missed" / "background"
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    background_idx = n

    for fname, gts in gt_by_image.items():
        preds = pred_by_image.get(fname, [])
        matched_pred = set()

        for gt in gts:
            best_iou, best_pred_idx = 0, -1
            for i, pred in enumerate(preds):
                if i in matched_pred:
                    continue
                iou = box_iou(gt["box"], pred["box"])
                if iou > best_iou:
                    best_iou, best_pred_idx = iou, i

            gt_idx = class_to_idx[gt["class"]]
            if best_iou >= iou_threshold:
                pred_idx = class_to_idx[preds[best_pred_idx]["class"]]
                matrix[gt_idx, pred_idx] += 1
                matched_pred.add(best_pred_idx)
            else:
                matrix[gt_idx, background_idx] += 1  # missed detection

        for i, pred in enumerate(preds):
            if i not in matched_pred:
                pred_idx = class_to_idx[pred["class"]]
                matrix[background_idx, pred_idx] += 1  # false positive

    return matrix


def print_watch_pair_confusion(matrix, class_names):
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print("\n[FRONT/REAR & LEFT/RIGHT MIXUP RATES]")
    print("-" * 60)
    for a, b in CONFUSION_PAIRS_TO_WATCH:
        if a not in class_to_idx or b not in class_to_idx:
            continue
        ia, ib = class_to_idx[a], class_to_idx[b]
        a_total = matrix[ia, :].sum()
        b_total = matrix[ib, :].sum()
        a_to_b = matrix[ia, ib]
        b_to_a = matrix[ib, ia]
        a_to_b_rate = a_to_b / a_total if a_total > 0 else 0
        b_to_a_rate = b_to_a / b_total if b_total > 0 else 0
        print(f"{a:20s} -> {b:20s}: {a_to_b}/{a_total} ({a_to_b_rate:.1%})")
        print(f"{b:20s} -> {a:20s}: {b_to_a}/{b_total} ({b_to_a_rate:.1%})")
        print("-" * 60)


def plot_confusion_matrix(matrix, class_names, out_path):
    labels = class_names + ["background/missed"]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Confusion matrix plot saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["yolo", "mask2former", "maskrcnn"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="combined_carparts", help="Dataset name or path (default: combined_carparts)")
    parser.add_argument("--out_dir", default="runs_comparison/confusion_matrices", help="Output directory for confusion matrices")
    args = parser.parse_args()

    class_names = CARPARTS_SEG_CLASSES
    ds_path = Path(args.dataset)
    if ds_path.is_dir():
        base = ds_path
    else:
        base = PROJECT_ROOT / "datasets" / args.dataset

    if (base / "coco_test.json").exists():
        test_images_dir = str(base / "images" / "test")
        test_json = str(base / "coco_test.json")
    elif (base / "coco_val.json").exists():
        test_images_dir = str(base / "images" / "val")
        test_json = str(base / "coco_val.json")
    elif (base / "annotations" / "instances_test.json").exists():
        test_images_dir = str(base / "images" / "test")
        test_json = str(base / "annotations" / "instances_test.json")
    else:
        test_images_dir = str(base / "images" / "val")
        test_json = str(base / "annotations" / "instances_val.json")

    if Path(test_json).exists():
        with open(test_json) as f:
            cats = json.load(f).get("categories", [])
            if cats:
                class_names = [c["name"] for c in cats]

    print(f"[*] Running inference with {args.model} on {args.dataset} test set ...")
    if args.model == "yolo":
        preds = get_predictions_yolo(args.weights, test_images_dir, class_names)
    elif args.model == "mask2former":
        preds = get_predictions_mask2former(args.weights, test_images_dir, test_json, class_names)
    elif args.model == "maskrcnn":
        preds = get_predictions_maskrcnn(args.weights, test_images_dir, test_json, class_names)

    gts = load_ground_truth(test_json, class_names)

    print("[*] Building confusion matrix ...")
    matrix = build_confusion_matrix(gts, preds, class_names)

    print_watch_pair_confusion(matrix, class_names)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(matrix, class_names, out_dir / f"{args.model}_{args.dataset}_confusion.png")

    # Save raw matrix + summary as JSON too, for later side-by-side comparison across models
    np.save(out_dir / f"{args.model}_{args.dataset}_matrix.npy", matrix)
    print(f"\n[DONE] Results saved in {out_dir}")


if __name__ == "__main__":
    main()
