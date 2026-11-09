"""
Hungarian instance matching module.
Performs class-agnostic one-to-one assignment between predicted teacher instances
and verified ground-truth instances using mask IoU (or box IoU).
Categorizes into: TP, FP_class, FP_loc, FP_unmatched, and FN.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_utils

from pipelines.adaptive_distillation.teachers.teacher_inference import TeacherInstance


@dataclass
class GroundTruthInstance:
    gt_id: int
    category_id: int
    category_name: str
    box: List[float]                       # [x1, y1, x2, y2]
    mask_rle: Dict[str, Any]
    area: float
    iscrowd: bool = False


@dataclass
class MatchResult:
    teacher_instance_id: int
    matched_gt_id: Optional[int]
    status: str                            # "TP", "FP_class", "FP_loc", "FP_unmatched"
    iou: float
    predicted_canonical_class: Optional[int]
    gt_canonical_class: Optional[int]
    quality_score: float                   # Quality metric based on IoU


class InstanceMatcher:
    """Class-agnostic Hungarian matcher with class-error and localization-error diagnosis."""

    def __init__(self, tau_match: float = 0.50, tau_low: float = 0.10, metric: str = "mask_iou"):
        self.tau_match = tau_match
        self.tau_low = tau_low
        self.metric = metric

    @staticmethod
    def compute_box_iou(box_a: List[float], box_b: List[float]) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    @staticmethod
    def compute_mask_iou(rle_a: Dict[str, Any], rle_b: Dict[str, Any]) -> float:
        ious = mask_utils.iou([rle_a], [rle_b], [0])
        return float(ious[0, 0])

    def match(
        self,
        teacher_instances: List[TeacherInstance],
        gt_instances: List[GroundTruthInstance]
    ) -> Tuple[List[MatchResult], List[int]]:
        """
        Matches teacher predictions to ground truth.
        Returns:
            - List of MatchResult for each teacher instance.
            - List of unmatched GT instance IDs (FN - missed ground-truth).
        """
        num_preds = len(teacher_instances)
        num_gts = len(gt_instances)

        if num_preds == 0:
            # All GTs are missed
            fn_ids = [gt.gt_id for gt in gt_instances]
            return [], fn_ids

        if num_gts == 0:
            # All predictions are unmatched FPs
            results = [
                MatchResult(
                    teacher_instance_id=ti.instance_id,
                    matched_gt_id=None,
                    status="FP_unmatched",
                    iou=0.0,
                    predicted_canonical_class=ti.canonical_class_id,
                    gt_canonical_class=None,
                    quality_score=0.0
                )
                for ti in teacher_instances
            ]
            return results, []

        # 1. Build cost matrix (negative IoU for Hungarian assignment)
        cost_matrix = np.zeros((num_preds, num_gts), dtype=np.float32)
        iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float32)

        for i, ti in enumerate(teacher_instances):
            for j, gt in enumerate(gt_instances):
                if self.metric == "mask_iou" and ti.mask_rle and gt.mask_rle:
                    iou = self.compute_mask_iou(ti.mask_rle, gt.mask_rle)
                else:
                    iou = self.compute_box_iou(ti.box, gt.box)
                iou_matrix[i, j] = iou
                cost_matrix[i, j] = -iou

        # 2. Hungarian assignment
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        matched_preds = set()
        matched_gts = set()
        results_by_pred_idx: Dict[int, MatchResult] = {}

        for p_idx, g_idx in zip(pred_indices, gt_indices):
            iou = float(iou_matrix[p_idx, g_idx])
            ti = teacher_instances[p_idx]
            gt = gt_instances[g_idx]

            pred_cid = ti.canonical_class_id
            gt_cid = gt.category_id

            if iou >= self.tau_match:
                matched_preds.add(p_idx)
                matched_gts.add(g_idx)
                if pred_cid is not None and pred_cid == gt_cid:
                    status = "TP"
                else:
                    status = "FP_class"  # localized well, but wrong class

                results_by_pred_idx[p_idx] = MatchResult(
                    teacher_instance_id=ti.instance_id,
                    matched_gt_id=gt.gt_id,
                    status=status,
                    iou=iou,
                    predicted_canonical_class=pred_cid,
                    gt_canonical_class=gt_cid,
                    quality_score=iou
                )

            elif iou >= self.tau_low:
                # Poor localization
                matched_preds.add(p_idx)
                matched_gts.add(g_idx)
                status = "FP_loc"

                results_by_pred_idx[p_idx] = MatchResult(
                    teacher_instance_id=ti.instance_id,
                    matched_gt_id=gt.gt_id,
                    status=status,
                    iou=iou,
                    predicted_canonical_class=pred_cid,
                    gt_canonical_class=gt_cid,
                    quality_score=iou
                )

        # 3. Handle unmatched predictions
        for p_idx, ti in enumerate(teacher_instances):
            if p_idx not in results_by_pred_idx:
                best_iou = float(np.max(iou_matrix[p_idx, :])) if num_gts > 0 else 0.0
                results_by_pred_idx[p_idx] = MatchResult(
                    teacher_instance_id=ti.instance_id,
                    matched_gt_id=None,
                    status="FP_unmatched",
                    iou=best_iou,
                    predicted_canonical_class=ti.canonical_class_id,
                    gt_canonical_class=None,
                    quality_score=0.0
                )

        # 4. Handle unmatched GTs (FN)
        fn_ids = [gt.gt_id for g_idx, gt in enumerate(gt_instances) if g_idx not in matched_gts]

        sorted_results = [results_by_pred_idx[i] for i in range(num_preds)]
        return sorted_results, fn_ids

