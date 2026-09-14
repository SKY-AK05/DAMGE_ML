# -*- coding: utf-8 -*-
"""
utils.py — shared helpers for the Mask2Former evaluation pipeline.

Covers:
  - The 32-class label mapping and GT→model label normalisation
  - CVAT XML annotation loading (both GT and prediction XMLs)
  - Polygon ↔ binary-mask conversion
  - Per-instance IoU and Dice computation
  - Colour palette and image-drawing helpers
  - Per-class and overall metric aggregation
  - Confusion-matrix builder
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  CLASS DEFINITIONS
# ---------------------------------------------------------------------------

# Exact id2label from the fine-tuned model config.json
MODEL_ID2LABEL: Dict[int, str] = {
    0: "wheel",
    1: "wheel_cap",
    2: "front_left_light",
    3: "front_right_light",
    4: "door_glass",
    5: "left_mirror",
    6: "right_mirror",
    7: "front_bumper",
    8: "grille",
    9: "front_left_door",
    10: "front_right_door",
    11: "hood",
    12: "pillar",
    13: "back_left_door",
    14: "back_right_door",
    15: "front_glass",
    16: "front_left_fender",
    17: "front_right_fender",
    18: "back_left_light",
    19: "back_right_light",
    20: "rear_quarter_panel",
    21: "roof",
    22: "side_skirt_rocker_panel",
    23: "fog_lamp",
    24: "back_bumper",
    25: "indicator",
    26: "number_plate",
    27: "back_glass",
    28: "tailgate",
    29: "fuel_lid",
    30: "antenna",
    31: "trunk",
}

MODEL_LABEL2ID: Dict[str, int] = {v: k for k, v in MODEL_ID2LABEL.items()}
ALL_CLASSES: List[str] = [MODEL_ID2LABEL[i] for i in range(len(MODEL_ID2LABEL))]

# ---------------------------------------------------------------------------
# 2.  GT LABEL NORMALISATION
# ---------------------------------------------------------------------------
# Ground-truth XML uses human-readable labels that differ from the model's
# snake_case names.  This table maps (gt_label, position_attribute) pairs to
# model labels.  When position is irrelevant, use None as the key.
#
# The mapping is intentionally explicit so that any unmapped GT label is
# flagged as a LABEL_MAPPING_MISMATCH rather than silently dropped.

_GT_LABEL_MAP: Dict[Tuple[str, Optional[str]], str] = {
    # Bumpers
    ("front bumper",  None):          "front_bumper",
    ("front bumper",  "front"):       "front_bumper",
    ("rear bumper",   None):          "back_bumper",
    ("rear bumper",   "rear"):        "back_bumper",
    # Hood / bonnet
    ("bonnet",        None):          "hood",
    ("hood",          None):          "hood",
    # Trunk
    ("trunk lid",     None):          "trunk",
    ("trunk",         None):          "trunk",
    # Fenders
    ("front fender",  "left"):        "front_left_fender",
    ("front fender",  "right"):       "front_right_fender",
    ("front fender",  None):          "front_left_fender",   # fallback
    # Doors
    ("front door",    "front-left"):  "front_left_door",
    ("front door",    "front-right"): "front_right_door",
    ("front door",    None):          "front_left_door",
    ("rear door",     "rear-left"):   "back_left_door",
    ("rear door",     "rear-right"):  "back_right_door",
    ("rear door",     None):          "back_left_door",
    # Rear quarter
    ("rear quarter panel", "left"):   "rear_quarter_panel",
    ("rear quarter panel", "right"):  "rear_quarter_panel",
    ("rear quarter panel", None):     "rear_quarter_panel",
    # Headlamps / taillights
    ("headlamp",      "left"):        "front_left_light",
    ("headlamp",      "right"):       "front_right_light",
    ("headlamp",      None):          "front_left_light",
    ("taillight",     "left"):        "back_left_light",
    ("taillight",     "right"):       "back_right_light",
    ("taillight",     None):          "back_left_light",
    # Indicators / fog lamps
    ("indicator",     "left"):        "indicator",
    ("indicator",     "right"):       "indicator",
    ("indicator",     None):          "indicator",
    ("fog lamp",      "left"):        "fog_lamp",
    ("fog lamp",      "right"):       "fog_lamp",
    ("fog lamp",      None):          "fog_lamp",
    # Mirrors (ORVM = Outside Rear View Mirror)
    ("orvm",          "left"):        "left_mirror",
    ("orvm",          "right"):       "right_mirror",
    ("orvm",          None):          "left_mirror",
    ("mirror",        "left"):        "left_mirror",
    ("mirror",        "right"):       "right_mirror",
    ("mirror",        None):          "left_mirror",
    ("left_mirror",   None):          "left_mirror",
    ("right_mirror",  None):          "right_mirror",
    # Glass
    ("windscreen",    "front"):       "front_glass",
    ("windscreen",    None):          "front_glass",
    ("front glass",   None):          "front_glass",
    ("rear windscreen", "rear"):      "back_glass",
    ("rear windscreen", None):        "back_glass",
    ("back glass",    None):          "back_glass",
    ("door glass",    "front-left"):  "door_glass",
    ("door glass",    "front-right"): "door_glass",
    ("door glass",    "rear-left"):   "door_glass",
    ("door glass",    "rear-right"):  "door_glass",
    ("door glass",    None):          "door_glass",
    # Roof
    ("roof",          None):          "roof",
    # Wheel / wheel cap
    ("wheel",         "front-left"):  "wheel",
    ("wheel",         "front-right"): "wheel",
    ("wheel",         "rear-left"):   "wheel",
    ("wheel",         "rear-right"):  "wheel",
    ("wheel",         None):          "wheel",
    ("wheel cap",     "front-left"):  "wheel_cap",
    ("wheel cap",     "front-right"): "wheel_cap",
    ("wheel cap",     "rear-left"):   "wheel_cap",
    ("wheel cap",     "rear-right"):  "wheel_cap",
    ("wheel cap",     None):          "wheel_cap",
    # Grille
    ("grille",        None):          "grille",
    # Fuel lid
    ("fuel lid",      "left"):        "fuel_lid",
    ("fuel lid",      "right"):       "fuel_lid",
    ("fuel lid",      None):          "fuel_lid",
    # Side skirt
    ("side skirt / rocker panel", "left"):  "side_skirt_rocker_panel",
    ("side skirt / rocker panel", "right"): "side_skirt_rocker_panel",
    ("side skirt / rocker panel", None):    "side_skirt_rocker_panel",
    ("side skirt",    None):          "side_skirt_rocker_panel",
    ("rocker panel",  None):          "side_skirt_rocker_panel",
    # Number plate
    ("number plate",  "front"):       "number_plate",
    ("number plate",  "rear"):        "number_plate",
    ("number plate",  None):          "number_plate",
    # Antenna
    ("antenna",       None):          "antenna",
    # Pillar
    ("pillar",        "left side"):   "pillar",
    ("pillar",        "right side"):  "pillar",
    ("pillar",        None):          "pillar",
    # Tailgate
    ("tailgate",      None):          "tailgate",
}


def normalise_gt_label(raw_label: str, position: Optional[str]) -> Tuple[Optional[str], str]:
    """
    Map a GT label + position to the model's snake_case label.

    Returns:
        (model_label, status) where status is one of:
            'ok'                  — clean mapping found
            'unmapped'            — no mapping exists for this GT label
    """
    key_label = raw_label.strip().lower()
    key_pos   = position.strip().lower() if position else None

    # Try exact match with position first, then without
    for key in [(key_label, key_pos), (key_label, None)]:
        if key in _GT_LABEL_MAP:
            return _GT_LABEL_MAP[key], "ok"

    return None, "unmapped"


# ---------------------------------------------------------------------------
# 3.  COLOUR PALETTE
# ---------------------------------------------------------------------------

def _build_palette() -> Dict[str, Tuple[int, int, int]]:
    """Generate visually distinct colours for all 32 classes."""
    import colorsys
    colours = {}
    n = len(ALL_CLASSES)
    for i, name in enumerate(ALL_CLASSES):
        hue = i / n
        sat = 0.75 + 0.25 * (i % 2)
        val = 0.85 + 0.15 * ((i // 2) % 2)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colours[name] = (int(r * 255), int(g * 255), int(b * 255))
    return colours


CLASS_COLOURS: Dict[str, Tuple[int, int, int]] = _build_palette()
# Extra colour for unmapped / unknown labels
UNKNOWN_COLOUR = (180, 180, 180)


# ---------------------------------------------------------------------------
# 4.  ANNOTATION DATA STRUCTURES
# ---------------------------------------------------------------------------

class Annotation:
    """
    One polygon annotation (either GT or predicted).

    Attributes:
        label        : model snake_case label (None if unmapped)
        raw_label    : original label string from the XML
        position     : Position attribute value from GT XML (may be None)
        norm_status  : 'ok' | 'unmapped' (GT only)
        points       : Nx2 float32 array of polygon vertices in (x, y) order
        score        : confidence (predictions only, None for GT)
        source       : 'gt' or 'pred'
        image_name   : filename of the parent image
    """
    __slots__ = (
        "label", "raw_label", "position", "norm_status",
        "points", "score", "source", "image_name",
    )

    def __init__(
        self,
        label: Optional[str],
        raw_label: str,
        points: np.ndarray,
        source: str,
        image_name: str,
        position: Optional[str] = None,
        norm_status: str = "ok",
        score: Optional[float] = None,
    ):
        self.label       = label
        self.raw_label   = raw_label
        self.position    = position
        self.norm_status = norm_status
        self.points      = points
        self.score       = score
        self.source      = source
        self.image_name  = image_name

    def to_mask(self, h: int, w: int) -> np.ndarray:
        """Rasterise the polygon into a boolean mask of shape (h, w)."""
        pts = self.points.astype(np.int32).reshape((-1, 1, 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)


# ---------------------------------------------------------------------------
# 5.  CVAT XML LOADER
# ---------------------------------------------------------------------------

def load_cvat_xml(
    xml_path: Path,
    source_tag: str = "gt",
) -> Dict[str, List[Annotation]]:
    """
    Parse a CVAT 1.1 XML file.

    Returns a dict mapping image-name (stem only, lowercase) → list of
    Annotation objects.

    For GT XMLs:  normalises labels via the GT→model map.
    For pred XMLs: uses the label as-is (already in model format).
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"Annotation XML not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise ValueError(f"Cannot parse XML {xml_path}: {exc}") from exc

    root = tree.getroot()
    result: Dict[str, List[Annotation]] = {}
    unmapped_log = set()

    for img_el in root.findall("image"):
        raw_name  = img_el.get("name", "")
        img_stem  = Path(raw_name).stem.lower()

        annotations: List[Annotation] = []

        for poly_el in img_el.findall("polygon"):
            raw_label = poly_el.get("label", "").strip()
            points_str = poly_el.get("points", "")
            score_str  = poly_el.get("score") or poly_el.get("confidence")

            # Parse points
            try:
                pts_flat = [float(v) for v in points_str.replace(";", ",").split(",")]
                points = np.array(pts_flat, dtype=np.float32).reshape(-1, 2)
            except (ValueError, TypeError):
                logger.warning("Skipping polygon with invalid points in image %s", raw_name)
                continue

            if len(points) < 3:
                continue

            # Parse score
            score: Optional[float] = None
            if score_str:
                try:
                    score = float(score_str)
                except ValueError:
                    pass

            # Parse position attribute (GT XMLs)
            position: Optional[str] = None
            for attr_el in poly_el.findall("attribute"):
                if attr_el.get("name", "").lower() == "position":
                    position = (attr_el.text or "").strip() or None

            if source_tag == "gt":
                label, norm_status = normalise_gt_label(raw_label, position)
                if norm_status == "unmapped" and raw_label not in unmapped_log:
                    unmapped_log.add(raw_label)
                    logger.warning(
                        "GT label %r (position=%r) has no mapping to a model class.",
                        raw_label, position,
                    )
            else:
                # Prediction XML — label already in snake_case
                label = raw_label if raw_label in MODEL_LABEL2ID else None
                norm_status = "ok" if label else "unmapped"

            ann = Annotation(
                label=label,
                raw_label=raw_label,
                points=points,
                source=source_tag,
                image_name=img_stem,
                position=position,
                norm_status=norm_status,
                score=score,
            )
            annotations.append(ann)

        result[img_stem] = annotations

    return result


# ---------------------------------------------------------------------------
# 6.  MASK IoU / DICE
# ---------------------------------------------------------------------------

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks."""
    inter = float((mask_a & mask_b).sum())
    union = float((mask_a | mask_b).sum())
    return inter / union if union > 0 else 0.0


def mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Dice/F1 between two boolean masks."""
    inter = float((mask_a & mask_b).sum())
    denom = float(mask_a.sum() + mask_b.sum())
    return 2 * inter / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# 7.  MATCHING  (greedy IoU matching)
# ---------------------------------------------------------------------------

class MatchResult:
    """
    Outcome of matching one predicted annotation against all GT annotations.

    Error types:
        TP_CORRECT_CLASS   — IoU ≥ threshold AND class matches
        TP_WRONG_CLASS     — IoU ≥ threshold BUT class differs (classification error)
        FP_LOW_OVERLAP     — no GT instance with IoU ≥ threshold (false positive / boundary error)
        FP_NO_GT           — no GT annotation exists at all for this image
        GT_MISSED          — GT instance not matched by any prediction (false negative)
    """
    TP_CORRECT_CLASS = "TP_CORRECT_CLASS"
    TP_WRONG_CLASS   = "TP_WRONG_CLASS"     # mask ok, label wrong
    FP_LOW_OVERLAP   = "FP_LOW_OVERLAP"     # prediction has no GT match
    FP_NO_GT         = "FP_NO_GT"
    GT_MISSED        = "GT_MISSED"


def match_predictions_to_gt(
    pred_anns: List[Annotation],
    gt_anns:   List[Annotation],
    img_h: int,
    img_w: int,
    iou_threshold: float = 0.5,
) -> List[dict]:
    """
    Greedy bipartite matching (highest IoU first) between predictions and GT.

    Returns a list of dicts, one per prediction or unmatched GT, with keys:
        type, pred, gt, iou, dice
    """
    records = []
    unmatched_gt = list(range(len(gt_anns)))

    # Build GT masks once
    gt_masks = [ann.to_mask(img_h, img_w) for ann in gt_anns]

    # Sort predictions by score descending (higher confidence first)
    pred_order = sorted(
        range(len(pred_anns)),
        key=lambda i: pred_anns[i].score if pred_anns[i].score is not None else 0.0,
        reverse=True,
    )

    for pi in pred_order:
        pred = pred_anns[pi]
        pred_mask = pred.to_mask(img_h, img_w)

        if not unmatched_gt:
            # All GT instances already consumed — this is a false positive
            records.append({
                "type": MatchResult.FP_LOW_OVERLAP,
                "pred": pred,
                "gt":   None,
                "iou":  0.0,
                "dice": 0.0,
            })
            continue

        # Compute IoU with all remaining GT masks
        ious = [(mask_iou(pred_mask, gt_masks[gi]), gi) for gi in unmatched_gt]
        best_iou, best_gi = max(ious, key=lambda x: x[0])

        if best_iou >= iou_threshold:
            gt = gt_anns[best_gi]
            dice = mask_dice(pred_mask, gt_masks[best_gi])
            unmatched_gt.remove(best_gi)

            if pred.label == gt.label:
                records.append({
                    "type": MatchResult.TP_CORRECT_CLASS,
                    "pred": pred, "gt": gt,
                    "iou": best_iou, "dice": dice,
                })
            else:
                records.append({
                    "type": MatchResult.TP_WRONG_CLASS,
                    "pred": pred, "gt": gt,
                    "iou": best_iou, "dice": dice,
                })
        else:
            records.append({
                "type": MatchResult.FP_LOW_OVERLAP,
                "pred": pred,
                "gt":   None,
                "iou":  best_iou,
                "dice": 0.0,
            })

    # Remaining unmatched GT → missed detections
    for gi in unmatched_gt:
        records.append({
            "type": MatchResult.GT_MISSED,
            "pred": None,
            "gt":   gt_anns[gi],
            "iou":  0.0,
            "dice": 0.0,
        })

    return records


# ---------------------------------------------------------------------------
# 8.  METRIC AGGREGATION
# ---------------------------------------------------------------------------

def compute_metrics(
    all_match_records: List[dict],
    all_classes: List[str],
) -> Tuple[dict, dict]:
    """
    Compute overall and per-class metrics from match records.

    Returns:
        overall_metrics  : dict with keys precision, recall, f1, mean_iou,
                           mean_dice, tp, fp, fn, wrong_class
        per_class_metrics: dict[class_name] → same keys
    """
    from collections import defaultdict

    per_class: Dict[str, dict] = {c: dict(tp=0, fp=0, fn=0, wc=0, iou_sum=0.0, dice_sum=0.0, n_matched=0)
                                  for c in all_classes}

    totals = dict(tp=0, fp=0, fn=0, wc=0, iou_sum=0.0, dice_sum=0.0, n_matched=0)

    for rec in all_match_records:
        rtype = rec["type"]
        pred  = rec["pred"]
        gt    = rec["gt"]

        if rtype == MatchResult.TP_CORRECT_CLASS:
            label = pred.label
            if label and label in per_class:
                per_class[label]["tp"]        += 1
                per_class[label]["iou_sum"]   += rec["iou"]
                per_class[label]["dice_sum"]  += rec["dice"]
                per_class[label]["n_matched"] += 1
            totals["tp"]        += 1
            totals["iou_sum"]   += rec["iou"]
            totals["dice_sum"]  += rec["dice"]
            totals["n_matched"] += 1

        elif rtype == MatchResult.TP_WRONG_CLASS:
            # Count as FP for predicted class, FN for GT class
            pred_label = pred.label
            gt_label   = gt.label if gt else None
            if pred_label and pred_label in per_class:
                per_class[pred_label]["fp"] += 1
                per_class[pred_label]["wc"] += 1
            if gt_label and gt_label in per_class:
                per_class[gt_label]["fn"] += 1
            totals["fp"] += 1
            totals["fn"] += 1
            totals["wc"] += 1

        elif rtype in (MatchResult.FP_LOW_OVERLAP, MatchResult.FP_NO_GT):
            pred_label = pred.label if pred else None
            if pred_label and pred_label in per_class:
                per_class[pred_label]["fp"] += 1
            totals["fp"] += 1

        elif rtype == MatchResult.GT_MISSED:
            gt_label = gt.label if gt else None
            if gt_label and gt_label in per_class:
                per_class[gt_label]["fn"] += 1
            totals["fn"] += 1

    def _precision_recall_f1(d: dict) -> dict:
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou  = d["iou_sum"]  / d["n_matched"] if d["n_matched"] > 0 else 0.0
        mean_dice = d["dice_sum"] / d["n_matched"] if d["n_matched"] > 0 else 0.0
        return dict(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            mean_iou=round(mean_iou, 4),
            mean_dice=round(mean_dice, 4),
            tp=tp, fp=fp, fn=fn,
            wrong_class=d["wc"],
        )

    per_class_out = {c: _precision_recall_f1(per_class[c]) for c in all_classes}
    overall_out   = _precision_recall_f1(totals)
    return overall_out, per_class_out


# ---------------------------------------------------------------------------
# 9.  CONFUSION MATRIX
# ---------------------------------------------------------------------------

def build_confusion_matrix(
    match_records: List[dict],
    all_classes: List[str],
) -> np.ndarray:
    """
    Build a (N+1) × (N+1) confusion matrix.

    Rows = GT label index (last row = missed / FN with no pred class)
    Cols = Predicted label index (last col = false positives with no GT class)

    Only TP_CORRECT_CLASS and TP_WRONG_CLASS matches contribute cells where
    both pred and GT label are known.
    """
    n = len(all_classes)
    label_idx = {c: i for i, c in enumerate(all_classes)}
    cm = np.zeros((n + 1, n + 1), dtype=np.int32)

    for rec in match_records:
        rtype = rec["type"]
        pred  = rec["pred"]
        gt    = rec["gt"]

        pred_label = pred.label if pred and pred.label in label_idx else None
        gt_label   = gt.label   if gt   and gt.label   in label_idx else None

        if rtype in (MatchResult.TP_CORRECT_CLASS, MatchResult.TP_WRONG_CLASS):
            r = label_idx.get(gt_label,   n)
            c = label_idx.get(pred_label, n)
            cm[r, c] += 1
        elif rtype in (MatchResult.FP_LOW_OVERLAP, MatchResult.FP_NO_GT):
            c = label_idx.get(pred_label, n)
            cm[n, c] += 1   # no GT → last row
        elif rtype == MatchResult.GT_MISSED:
            r = label_idx.get(gt_label, n)
            cm[r, n] += 1   # no pred → last col

    return cm


# ---------------------------------------------------------------------------
# 10.  DRAWING HELPERS
# ---------------------------------------------------------------------------

def _try_load_font(size: int = 14):
    """Load a TTF font if available, otherwise fall back to the PIL default."""
    try:
        # Common Windows font
        return ImageFont.truetype("arial.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


def draw_annotations(
    image: Image.Image,
    annotations: List[Annotation],
    alpha: float = 0.35,
    draw_label: bool = True,
    label_prefix: str = "",
    border_width: int = 2,
    font_size: int = 13,
) -> Image.Image:
    """
    Draw polygon annotations onto a PIL Image.

    Uses a semi-transparent fill + solid border per instance.
    Returns a new PIL Image (original is not modified).
    """
    img = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _try_load_font(font_size)

    for ann in annotations:
        if len(ann.points) < 3:
            continue

        colour = CLASS_COLOURS.get(ann.label or "", UNKNOWN_COLOUR)
        r, g, b = colour
        poly_pts = [(float(x), float(y)) for x, y in ann.points]

        # Semi-transparent fill
        draw.polygon(poly_pts, fill=(r, g, b, int(255 * alpha)))
        # Solid border
        draw.line(poly_pts + [poly_pts[0]], fill=(r, g, b, 230), width=border_width)

        if draw_label:
            display_label = ann.label or ann.raw_label
            if label_prefix:
                display_label = f"{label_prefix}{display_label}"
            if ann.score is not None:
                display_label += f" {ann.score:.2f}"

            # Label background
            cx = int(np.mean([p[0] for p in poly_pts]))
            cy = int(np.mean([p[1] for p in poly_pts]))
            try:
                bbox = font.getbbox(display_label)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                tw, th = font.getsize(display_label)  # type: ignore[attr-defined]

            tx = max(0, cx - tw // 2)
            ty = max(0, cy - th // 2)
            draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
                           fill=(0, 0, 0, 180))
            draw.text((tx, ty), display_label, fill=(255, 255, 255, 255), font=font)

    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def draw_comparison(
    image: Image.Image,
    gt_anns: List[Annotation],
    pred_anns: List[Annotation],
    match_records: List[dict],
    img_h: int,
    img_w: int,
    font_size: int = 12,
) -> Image.Image:
    """
    Draw a side-by-side comparison: GT (left) | Predicted (right).
    Highlight mismatches in the match_records.
    """
    gt_img   = draw_annotations(image.copy(), gt_anns,   alpha=0.40, label_prefix="GT: ",  font_size=font_size)
    pred_img = draw_annotations(image.copy(), pred_anns, alpha=0.35, label_prefix="",       font_size=font_size)

    # Overlay mismatch indicators on prediction panel
    draw = ImageDraw.Draw(pred_img)
    font = _try_load_font(font_size)
    for rec in match_records:
        rtype = rec["type"]
        if rtype == MatchResult.TP_WRONG_CLASS and rec["pred"] and rec["gt"]:
            pred_ann = rec["pred"]
            pts = [(float(x), float(y)) for x, y in pred_ann.points]
            # Red dashed border for wrong class
            draw.line(pts + [pts[0]], fill=(255, 50, 50, 255), width=3)
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            note = f"⚠ should be {rec['gt'].label}"
            draw.rectangle([cx - 2, cy - 2, cx + 200, cy + 16], fill=(200, 0, 0))
            draw.text((cx, cy), note, fill=(255, 255, 255), font=font)

    # Concatenate side by side
    combined_w = gt_img.width + pred_img.width + 4
    combined_h = max(gt_img.height, pred_img.height) + 24
    combined   = Image.new("RGB", (combined_w, combined_h), (40, 40, 40))
    combined.paste(gt_img, (0, 24))
    combined.paste(pred_img, (gt_img.width + 4, 24))

    header_draw = ImageDraw.Draw(combined)
    header_font = _try_load_font(14)
    header_draw.text((10, 4), "GROUND TRUTH",   fill=(100, 230, 100), font=header_font)
    header_draw.text((gt_img.width + 14, 4), "MODEL PREDICTION", fill=(100, 180, 255), font=header_font)

    return combined


def draw_error_cases(
    image: Image.Image,
    match_records: List[dict],
    img_h: int,
    img_w: int,
    font_size: int = 12,
) -> Optional[Image.Image]:
    """
    Draw only error cases on the image with colour-coded borders and labels.

    Error colour coding:
        Red   — TP_WRONG_CLASS (classification error)
        Orange— FP_LOW_OVERLAP (boundary / false positive)
        Purple— GT_MISSED      (missed detection)
    Returns None if there are no errors.
    """
    errors = [r for r in match_records if r["type"] != MatchResult.TP_CORRECT_CLASS]
    if not errors:
        return None

    img = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    font    = _try_load_font(font_size)

    colour_map = {
        MatchResult.TP_WRONG_CLASS:   (255,  60,  60, 200),   # Red
        MatchResult.FP_LOW_OVERLAP:   (255, 160,   0, 200),   # Orange
        MatchResult.FP_NO_GT:         (255, 160,   0, 200),
        MatchResult.GT_MISSED:        (160,  60, 200, 200),   # Purple
    }

    for rec in errors:
        rtype  = rec["type"]
        colour = colour_map.get(rtype, (180, 180, 180, 200))
        ann    = rec["pred"] or rec["gt"]
        if ann is None or len(ann.points) < 3:
            continue

        pts = [(float(x), float(y)) for x, y in ann.points]
        draw.polygon(pts, fill=(*colour[:3], 60))
        draw.line(pts + [pts[0]], fill=colour, width=3)

        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))

        if rtype == MatchResult.TP_WRONG_CLASS:
            note = f"WRONG CLASS pred={rec['pred'].label} gt={rec['gt'].label} IoU={rec['iou']:.2f}"
        elif rtype in (MatchResult.FP_LOW_OVERLAP, MatchResult.FP_NO_GT):
            note = f"FALSE POS {ann.label} IoU={rec['iou']:.2f}"
        else:
            note = f"MISSED {ann.label}"

        try:
            bbox = font.getbbox(note)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = len(note) * 7, 14

        tx = max(0, min(cx, img.width - tw - 4))
        ty = max(0, min(cy, img.height - th - 4))
        draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], fill=(0, 0, 0, 190))
        draw.text((tx, ty), note, fill=(255, 255, 255, 255), font=font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result


# ---------------------------------------------------------------------------
# 11.  IMAGE FILE DISCOVERY
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def discover_images(root: Path) -> List[Path]:
    """Recursively find all supported image files under root."""
    found = []
    for ext in SUPPORTED_EXTENSIONS:
        found.extend(root.rglob(f"*{ext}"))
        found.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(found))


def safe_load_image(path: Path) -> Optional[Image.Image]:
    """Load an image, returning None and logging a warning on failure."""
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        return img
    except Exception as exc:
        logger.warning("Cannot load image %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# 12.  OUTPUT HELPERS
# ---------------------------------------------------------------------------

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def save_csv(rows: List[dict], path: Path) -> None:
    """Save a list of dicts as a CSV file."""
    import csv
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
