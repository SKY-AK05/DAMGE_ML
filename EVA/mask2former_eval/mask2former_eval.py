# -*- coding: utf-8 -*-
"""
mask2former_eval.py
===================
Mask2Former inference, annotation, and evaluation pipeline for car-parts
segmentation.

Usage (Windows PowerShell):
    python mask2former_eval.py `
        --input  "C:\\...\\GROUND" `
        --model  "C:\\...\\analytics_export\\analytics_export\\best_model\\best" `
        --output "C:\\...\\outputs" `
        [--gt_xml "C:\\...\\GROUND\\annotations.xml"] `
        [--conf 0.5] `
        [--iou_thresh 0.5] `
        [--device cpu]

Outputs
-------
outputs/
  predicted_annotations/     -- original image with predicted masks overlaid
  ground_truth_annotations/  -- original image with GT masks overlaid
  comparison_images/         -- side-by-side GT | Prediction
  error_cases/               -- only the error annotations highlighted
  metrics/
    overall_metrics.json
    per_class_metrics.csv
    confusion_matrix.csv
  summary_report.html
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — only pulled in after arg-parse so --help works without torch
# ---------------------------------------------------------------------------
def _import_model_libs():
    try:
        import torch
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
        return torch, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    except ImportError as exc:
        logger.error(
            "Required library not found: %s\n"
            "Install with:  pip install torch transformers",
            exc,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Local modules (sibling files)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from utils import (
    ALL_CLASSES,
    MODEL_ID2LABEL,
    SUPPORTED_EXTENSIONS,
    Annotation,
    MatchResult,
    build_confusion_matrix,
    compute_metrics,
    discover_images,
    draw_annotations,
    draw_comparison,
    draw_error_cases,
    load_cvat_xml,
    match_predictions_to_gt,
    safe_load_image,
    save_csv,
    save_json,
)
from report_generator import generate_html_report


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: Path, device: str):
    """Load the fine-tuned Mask2Former model and its image processor."""
    torch, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor = _import_model_libs()

    logger.info("Loading processor from %s", checkpoint)
    processor = Mask2FormerImageProcessor.from_pretrained(str(checkpoint))

    logger.info("Loading model from %s  (device=%s)", checkpoint, device)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(str(checkpoint))
    model = model.to(device)
    model.eval()
    logger.info("Model loaded — %d classes", len(MODEL_ID2LABEL))
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    image: Image.Image,
    model,
    processor,
    device: str,
    conf_threshold: float,
) -> List[Annotation]:
    """
    Run Mask2Former on a single PIL Image.

    Returns a list of Annotation objects (one per detected instance that
    passes the confidence threshold).
    """
    torch, _, _ = _import_model_libs()

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

    # Post-process — instance segmentation
    target_sizes = [image.size[::-1]]   # (H, W)
    results = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=target_sizes,
        threshold=conf_threshold,
    )[0]

    annotations: List[Annotation] = []
    img_h, img_w = image.size[1], image.size[0]

    segments_info = results.get("segments_info", [])
    pred_mask_tensor = results.get("segmentation")   # (H, W) int tensor

    if pred_mask_tensor is None or len(segments_info) == 0:
        return annotations

    pred_mask_np = pred_mask_tensor.cpu().numpy().astype(np.int32)

    for seg in segments_info:
        seg_id    = int(seg["id"])
        label_id  = int(seg["label_id"])
        score     = float(seg.get("score", 0.0))

        if score < conf_threshold:
            continue

        label = MODEL_ID2LABEL.get(label_id)
        if label is None:
            logger.debug("Skipping unknown label_id %d", label_id)
            continue

        # Extract polygon from binary mask
        binary = (pred_mask_np == seg_id).astype(np.uint8)
        if binary.sum() < 9:   # too small — skip
            continue

        points = _mask_to_polygon(binary)
        if points is None:
            continue

        ann = Annotation(
            label=label,
            raw_label=label,
            points=points,
            source="pred",
            image_name="",     # filled by caller
            score=score,
        )
        annotations.append(ann)

    return annotations


def _mask_to_polygon(binary: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert a binary mask to the largest polygon contour.

    Returns an Nx2 float32 array of (x, y) points, or None if no contour
    with ≥ 3 points is found.
    """
    import cv2
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep the largest contour by area
    largest = max(contours, key=cv2.contourArea)

    # Simplify slightly to reduce point count (epsilon = 1.5 px)
    epsilon = 1.5
    approx = cv2.approxPolyDP(largest, epsilon, closed=True)
    pts = approx.reshape(-1, 2).astype(np.float32)

    if len(pts) < 3:
        return None
    return pts


# ---------------------------------------------------------------------------
# Annotation review helper
# ---------------------------------------------------------------------------

def _classify_mismatch(record: dict) -> str:
    """
    Return a human-readable error-type string for annotation review output.

    Distinguishes:
        GENUINE_MODEL_ERROR      — mask ok, model clearly predicted the wrong class
        POSSIBLE_GT_ERROR        — model prediction visually plausible, GT may be wrong
        BOUNDARY_ERROR           — class correct but mask IoU below threshold
        FALSE_POSITIVE           — no matching GT at all
        MISSED_DETECTION         — GT object not found by model
    """
    rtype = record["type"]
    if rtype == MatchResult.TP_CORRECT_CLASS:
        return "CORRECT"
    if rtype == MatchResult.TP_WRONG_CLASS:
        # We cannot auto-determine which side is wrong — flag both possibilities
        iou = record["iou"]
        if iou >= 0.7:
            return "CLASSIFICATION_ERROR (IoU good — inspect GT label)"
        else:
            return "CLASSIFICATION_ERROR (IoU moderate)"
    if rtype == MatchResult.FP_LOW_OVERLAP:
        iou = record["iou"]
        if iou > 0.1:
            return f"BOUNDARY_ERROR (IoU={iou:.2f} — mask shape poor)"
        return "FALSE_POSITIVE (no GT overlap)"
    if rtype == MatchResult.FP_NO_GT:
        return "FALSE_POSITIVE (no GT for this image)"
    if rtype == MatchResult.GT_MISSED:
        return "MISSED_DETECTION"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

def _make_output_dirs(base: Path) -> dict:
    dirs = {
        "pred":    base / "predicted_annotations",
        "gt":      base / "ground_truth_annotations",
        "comp":    base / "comparison_images",
        "errors":  base / "error_cases",
        "metrics": base / "metrics",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Save annotation XML (CVAT 1.1 format)
# ---------------------------------------------------------------------------

def _save_predictions_xml(
    all_preds: Dict[str, List[Annotation]],
    image_sizes: Dict[str, Tuple[int, int]],
    out_path: Path,
) -> None:
    """Write all predictions to a CVAT 1.1 XML file for import into CVAT."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "mask2former_predictions"
    ET.SubElement(task, "mode").text = "annotation"

    labels_el = ET.SubElement(task, "labels")
    for cls in ALL_CLASSES:
        lbl = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl, "name").text = cls
        ET.SubElement(lbl, "type").text = "polygon"
        ET.SubElement(lbl, "attributes")

    for img_id, (img_stem, preds) in enumerate(all_preds.items()):
        h, w = image_sizes.get(img_stem, (0, 0))
        img_el = ET.SubElement(root, "image",
                               id=str(img_id),
                               name=img_stem + ".jpg",
                               width=str(w),
                               height=str(h))
        for ann in preds:
            pts_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in ann.points)
            poly_el = ET.SubElement(img_el, "polygon",
                                    label=ann.raw_label,
                                    source="mask2former_finetuned",
                                    occluded="0",
                                    points=pts_str,
                                    z_order="0")
            if ann.score is not None:
                poly_el.set("score", f"{ann.score:.4f}")

    def _indent(elem, level=0):
        pad = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = pad + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = pad
            for child in elem:
                _indent(child, level + 1)
            if not child.tail or not child.tail.strip():  # noqa: F821
                child.tail = pad
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = pad

    _indent(root)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(out_path, "wb") as fh:
        tree.write(fh, encoding="utf-8", xml_declaration=True)
    logger.info("Saved predictions XML → %s", out_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:

    input_dir  = Path(args.input)
    model_dir  = Path(args.model)
    output_dir = Path(args.output)
    gt_xml     = Path(args.gt_xml) if args.gt_xml else (input_dir / "annotations.xml")
    device     = args.device
    conf_thresh = args.conf
    iou_thresh  = args.iou_thresh

    # ── Validate paths ────────────────────────────────────────────────────
    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)
    if not model_dir.exists():
        logger.error("Model checkpoint not found: %s", model_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = _make_output_dirs(output_dir)

    # ── Load model ────────────────────────────────────────────────────────
    model, processor = load_model(model_dir, device)

    # ── Load GT annotations ───────────────────────────────────────────────
    gt_data: Dict[str, List[Annotation]] = {}
    gt_available = False

    if gt_xml.exists():
        try:
            gt_data = load_cvat_xml(gt_xml, source_tag="gt")
            gt_available = True
            total_gt_imgs = len(gt_data)
            total_gt_anns = sum(len(v) for v in gt_data.values())
            logger.info(
                "GT XML loaded: %d images, %d annotations  (%s)",
                total_gt_imgs, total_gt_anns, gt_xml,
            )
            # Warn about unmapped labels
            unmapped = defaultdict(int)
            for anns in gt_data.values():
                for ann in anns:
                    if ann.norm_status == "unmapped":
                        unmapped[ann.raw_label] += 1
            if unmapped:
                logger.warning(
                    "GT labels with NO model mapping (will be skipped in evaluation): %s",
                    dict(unmapped),
                )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Could not load GT XML: %s — running inference only", exc)
    else:
        logger.info(
            "No GT XML found at %s — running inference only (no evaluation).", gt_xml
        )

    # ── Discover images ───────────────────────────────────────────────────
    images = discover_images(input_dir)
    if not images:
        logger.error("No supported images found in %s", input_dir)
        sys.exit(1)
    logger.info("Found %d images in %s", len(images), input_dir)

    # ── Per-image processing ──────────────────────────────────────────────
    all_preds:      Dict[str, List[Annotation]] = {}
    image_sizes:    Dict[str, Tuple[int, int]]  = {}   # stem → (H, W)
    all_match_records: List[dict] = []
    per_image_summary: List[dict] = []

    predicted_imgs:  List[Path] = []
    gt_imgs:         List[Path] = []
    comparison_imgs: List[Path] = []
    error_imgs:      List[Path] = []

    t0 = time.time()

    for idx, img_path in enumerate(images, 1):
        stem = img_path.stem.lower()
        logger.info("[%d/%d] %s", idx, len(images), img_path.name)

        # Load image
        pil_img = safe_load_image(img_path)
        if pil_img is None:
            continue

        img_w, img_h = pil_img.size
        image_sizes[stem] = (img_h, img_w)

        # ── Inference ────────────────────────────────────────────────────
        preds = run_inference(pil_img, model, processor, device, conf_thresh)
        for ann in preds:
            ann.image_name = stem
        all_preds[stem] = preds

        # ── Save predicted annotation image ──────────────────────────────
        pred_img = draw_annotations(pil_img, preds, alpha=0.35, draw_label=True)
        pred_out = dirs["pred"] / f"{img_path.stem}_pred.jpg"
        pred_img.save(pred_out, quality=92)
        predicted_imgs.append(pred_out)

        # ── GT annotations ────────────────────────────────────────────────
        gt_anns_raw = gt_data.get(stem, [])
        # Filter to only mapped annotations for evaluation
        gt_anns = [a for a in gt_anns_raw if a.label is not None]
        # Also note unmapped ones for the annotation review output
        unmapped_gt = [a for a in gt_anns_raw if a.label is None]

        if gt_available and gt_anns_raw:
            gt_img = draw_annotations(pil_img, gt_anns_raw, alpha=0.40, draw_label=True, label_prefix="GT: ")
            gt_out = dirs["gt"] / f"{img_path.stem}_gt.jpg"
            gt_img.save(gt_out, quality=92)
            gt_imgs.append(gt_out)

        # ── Matching + metrics ────────────────────────────────────────────
        img_records: List[dict] = []

        if gt_available:
            if gt_anns:
                img_records = match_predictions_to_gt(
                    preds, gt_anns, img_h, img_w, iou_threshold=iou_thresh
                )
            elif preds:
                # Predictions exist but no GT for this image
                img_records = [
                    {"type": MatchResult.FP_NO_GT, "pred": p, "gt": None, "iou": 0.0, "dice": 0.0}
                    for p in preds
                ]

            all_match_records.extend(img_records)

            # ── Per-image summary ─────────────────────────────────────────
            tp   = sum(1 for r in img_records if r["type"] == MatchResult.TP_CORRECT_CLASS)
            fp   = sum(1 for r in img_records if r["type"] in
                       (MatchResult.FP_LOW_OVERLAP, MatchResult.FP_NO_GT))
            fn   = sum(1 for r in img_records if r["type"] == MatchResult.GT_MISSED)
            wc   = sum(1 for r in img_records if r["type"] == MatchResult.TP_WRONG_CLASS)
            matched_ious = [r["iou"] for r in img_records
                            if r["type"] in (MatchResult.TP_CORRECT_CLASS, MatchResult.TP_WRONG_CLASS)]
            mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

            per_image_summary.append({
                "image":    img_path.name,
                "n_gt":     len(gt_anns),
                "n_pred":   len(preds),
                "tp":       tp,
                "fp":       fp,
                "fn":       fn,
                "wrong_class": wc,
                "mean_iou": round(mean_iou, 4),
            })

            # ── Comparison image ──────────────────────────────────────────
            if gt_anns:
                comp = draw_comparison(pil_img, gt_anns, preds, img_records, img_h, img_w)
                comp_out = dirs["comp"] / f"{img_path.stem}_comparison.jpg"
                comp.save(comp_out, quality=88)
                comparison_imgs.append(comp_out)

            # ── Error cases ───────────────────────────────────────────────
            err_img = draw_error_cases(pil_img, img_records, img_h, img_w)
            if err_img is not None:
                err_out = dirs["errors"] / f"{img_path.stem}_errors.jpg"
                err_img.save(err_out, quality=88)
                error_imgs.append(err_out)

            # ── Detailed annotation review log ────────────────────────────
            if img_records:
                _log_annotation_review(
                    img_path.name, img_records, unmapped_gt, dirs["errors"]
                )

    elapsed = time.time() - t0
    logger.info("Inference done — %.1fs for %d images (%.2fs/image)",
                elapsed, len(images), elapsed / max(len(images), 1))

    # ── Save predictions XML ──────────────────────────────────────────────
    xml_out = output_dir / "predictions.xml"
    _save_predictions_xml(all_preds, image_sizes, xml_out)

    # ── Aggregate metrics ─────────────────────────────────────────────────
    if gt_available and all_match_records:
        overall, per_class = compute_metrics(all_match_records, ALL_CLASSES)
        cm = build_confusion_matrix(all_match_records, ALL_CLASSES)

        # Save overall metrics
        save_json(
            {
                **overall,
                "conf_threshold": conf_thresh,
                "iou_threshold":  iou_thresh,
                "total_images":   len(images),
                "images_with_gt": len([s for s in per_image_summary if s["n_gt"] > 0]),
            },
            dirs["metrics"] / "overall_metrics.json",
        )

        # Save per-class metrics CSV
        pc_rows = [
            {"class": cls, **per_class.get(cls, {})}
            for cls in ALL_CLASSES
        ]
        save_csv(pc_rows, dirs["metrics"] / "per_class_metrics.csv")

        # Save confusion matrix CSV
        _save_cm_csv(cm, ALL_CLASSES, dirs["metrics"] / "confusion_matrix.csv")

        # Save per-image summary CSV
        save_csv(per_image_summary, dirs["metrics"] / "per_image_summary.csv")

        logger.info(
            "Metrics — Precision=%.3f  Recall=%.3f  F1=%.3f  MeanIoU=%.3f",
            overall["precision"], overall["recall"], overall["f1"], overall["mean_iou"],
        )

        # Print per-class summary to console
        _print_class_summary(per_class)

        # ── HTML report ───────────────────────────────────────────────────
        report = generate_html_report(
            output_dir=output_dir,
            overall_metrics=overall,
            per_class_metrics=per_class,
            confusion_matrix=cm,
            all_classes=ALL_CLASSES,
            match_summary=per_image_summary,
            predicted_images=predicted_imgs,
            gt_images=gt_imgs,
            comparison_images=comparison_imgs,
            error_images=error_imgs,
            model_checkpoint=str(model_dir),
            input_dir=str(input_dir),
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            total_images=len(images),
            images_with_gt=len([s for s in per_image_summary if s["n_gt"] > 0]),
        )
        logger.info("HTML report → %s", report)

    else:
        logger.info(
            "No GT data available or no matches computed — "
            "skipping metrics and HTML report."
        )

    logger.info("All outputs saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Annotation review log helper
# ---------------------------------------------------------------------------

def _log_annotation_review(
    img_name: str,
    records: List[dict],
    unmapped_gt: List[Annotation],
    errors_dir: Path,
) -> None:
    """
    Write a human-readable .txt annotation review file per image.
    Never overwrites GT — only reports findings.
    """
    lines = [
        f"ANNOTATION REVIEW — {img_name}",
        "=" * 60,
        "",
        "This file helps you manually inspect annotation quality.",
        "DO NOT use this to auto-correct ground-truth labels.",
        "",
    ]

    if unmapped_gt:
        lines += ["UNMAPPED GT LABELS (not evaluated):"]
        for ann in unmapped_gt:
            lines.append(f"  - raw_label={ann.raw_label!r}  position={ann.position!r}")
        lines.append("")

    for i, rec in enumerate(records, 1):
        rtype   = rec["type"]
        pred    = rec["pred"]
        gt      = rec["gt"]
        iou     = rec["iou"]
        dice    = rec["dice"]
        verdict = _classify_mismatch(rec)

        lines.append(f"[{i}] {verdict}")

        if rtype == MatchResult.TP_CORRECT_CLASS:
            lines.append(f"      Pred={pred.label}  Score={pred.score:.3f}  IoU={iou:.3f}  Dice={dice:.3f}")

        elif rtype == MatchResult.TP_WRONG_CLASS:
            lines.append(f"      Pred label : {pred.label}  (score={pred.score:.3f})")
            lines.append(f"      GT label   : {gt.label}  (raw={gt.raw_label!r}, pos={gt.position!r})")
            lines.append(f"      IoU={iou:.3f}  Dice={dice:.3f}")
            lines.append("      INSPECT: Is the GT label correct for this polygon?")
            lines.append("               If the model's prediction looks plausible,")
            lines.append("               this may be a GT annotation error.")

        elif rtype in (MatchResult.FP_LOW_OVERLAP, MatchResult.FP_NO_GT):
            lines.append(f"      Pred={pred.label}  Score={pred.score:.3f}  Best IoU={iou:.3f}")
            if iou > 0.1:
                lines.append("      INSPECT: Mask shape may be inaccurate (boundary error).")
            else:
                lines.append("      No corresponding GT polygon found.")

        elif rtype == MatchResult.GT_MISSED:
            lines.append(f"      GT={gt.label}  (raw={gt.raw_label!r})")
            lines.append("      Model produced no prediction for this GT instance.")

        lines.append("")

    review_path = errors_dir / f"{Path(img_name).stem}_review.txt"
    with open(review_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_cm_csv(cm: np.ndarray, labels: List[str], path: Path) -> None:
    """Save confusion matrix as CSV with labelled rows/columns."""
    import csv
    header = ["GT \\ Pred"] + labels + ["UNMATCHED"]
    row_labels = labels + ["NO_GT"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r, row_lbl in enumerate(row_labels):
            writer.writerow([row_lbl] + [int(cm[r, c]) for c in range(cm.shape[1])])


def _print_class_summary(per_class: dict) -> None:
    """Print a compact per-class table to the console."""
    header = f"{'Class':<28} {'P':>6} {'R':>6} {'F1':>6} {'mIoU':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'WC':>5}"
    print("\n" + "=" * 75)
    print("PER-CLASS RESULTS")
    print("=" * 75)
    print(header)
    print("-" * 75)

    # Sort by F1 descending
    sorted_classes = sorted(
        per_class.items(),
        key=lambda kv: kv[1].get("f1", 0.0),
        reverse=True,
    )
    for cls, m in sorted_classes:
        print(
            f"{cls:<28} "
            f"{m.get('precision',0):>6.3f} "
            f"{m.get('recall',0):>6.3f} "
            f"{m.get('f1',0):>6.3f} "
            f"{m.get('mean_iou',0):>6.3f} "
            f"{m.get('tp',0):>5} "
            f"{m.get('fp',0):>5} "
            f"{m.get('fn',0):>5} "
            f"{m.get('wrong_class',0):>5}"
        )
    print("=" * 75 + "\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    # Default paths relative to THIS script's location
    _script_dir = Path(__file__).parent
    _workspace  = _script_dir.parent  # annotation_cleaning_pipeline/

    default_model = str(
        _workspace
        / "analytics_export"
        / "analytics_export"
        / "best_model"
        / "best"
    )
    default_input  = str(_workspace / "GROUND")
    default_output = str(_workspace / "outputs")

    parser = argparse.ArgumentParser(
        description="Mask2Former car-parts inference and evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Basic inference + evaluation (GT XML auto-detected from --input dir):
  python mask2former_eval.py

# Override all paths explicitly:
  python mask2former_eval.py ^
    --input  "C:\\...\\GROUND" ^
    --model  "C:\\...\\best_model\\best" ^
    --output "C:\\...\\outputs" ^
    --gt_xml "C:\\...\\GROUND\\annotations.xml" ^
    --conf   0.45 ^
    --device cuda

# Inference only (no GT):
  python mask2former_eval.py --input "C:\\...\\images_folder" --no_gt
""",
    )

    parser.add_argument(
        "--input",
        default=default_input,
        help="Directory containing input images (searched recursively). "
             f"Default: {default_input}",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="Path to fine-tuned Mask2Former checkpoint directory. "
             f"Default: {default_model}",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Output root directory. Default: {default_output}",
    )
    parser.add_argument(
        "--gt_xml",
        default=None,
        help="Path to CVAT 1.1 annotations.xml for ground-truth evaluation. "
             "If omitted, the script looks for annotations.xml inside --input.",
    )
    parser.add_argument(
        "--no_gt",
        action="store_true",
        default=False,
        help="Skip GT loading entirely — inference-only mode.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        metavar="THRESHOLD",
        help="Confidence threshold for predicted instances (0–1). Default: 0.5",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        metavar="THRESHOLD",
        help="IoU threshold for matching a prediction to a GT instance. Default: 0.5",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device. Default: cpu",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Override GT path when --no_gt is set
    if args.no_gt:
        args.gt_xml = "__NONE__"   # sentinel — load_cvat_xml won't be called

    logger.info("=" * 60)
    logger.info("Mask2Former Car-Parts Evaluation Pipeline")
    logger.info("  Input : %s", args.input)
    logger.info("  Model : %s", args.model)
    logger.info("  Output: %s", args.output)
    logger.info("  Conf  : %.2f   IoU thresh: %.2f   Device: %s",
                args.conf, args.iou_thresh, args.device)
    logger.info("=" * 60)

    run_pipeline(args)


if __name__ == "__main__":
    main()
