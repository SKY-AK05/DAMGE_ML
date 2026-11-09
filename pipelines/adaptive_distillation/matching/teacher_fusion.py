"""
Teacher Fusion and Disagreement Arbitration module.
Implements:
1. Multi-teacher instance clustering via mask IoU
2. Weighted mask and box fusion for agreed instances
3. GT-arbitrated class conflict resolution
4. Solo-detection penalty kappa and disagreement logging
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import cv2
from pycocotools import mask as mask_utils

from pipelines.adaptive_distillation.config import FusionConfig
from pipelines.adaptive_distillation.teachers.teacher_inference import TeacherInstance


@dataclass
class FusedInstance:
    fused_id: int
    canonical_class_id: int
    canonical_class_name: Optional[str]
    box: List[float]                       # [x1, y1, x2, y2]
    mask_rle: Dict[str, Any]
    mask_prob_downscaled: Optional[np.ndarray]
    effective_weight: float
    contributing_teachers: List[str]
    is_fused: bool                         # True if >= 2 teachers contributed
    disagreement_flag: bool                # True if teachers had class conflict


class TeacherFusion:
    """Clustering, arbitration, and fusion across N teachers."""

    def __init__(self, config: FusionConfig):
        self.config = config

    @staticmethod
    def mask_iou(rle_a: Dict[str, Any], rle_b: Dict[str, Any]) -> float:
        if not rle_a or not rle_b:
            return 0.0
        ious = mask_utils.iou([rle_a], [rle_b], [0])
        return float(ious[0, 0])

    def fuse_image_instances(
        self,
        teacher_instances_by_teacher: Dict[str, List[Tuple[TeacherInstance, float]]],
        gt_canonical_classes: Optional[List[int]] = None
    ) -> List[FusedInstance]:
        """
        teacher_instances_by_teacher: {teacher_name: [(instance, weight_w), ...]}
        Fuses instances across all contributing teachers for a single image.
        """
        all_items: List[Tuple[str, TeacherInstance, float]] = []
        for t_name, inst_list in teacher_instances_by_teacher.items():
            for inst, w in inst_list:
                if w > 0.0 and inst.canonical_class_id is not None:
                    all_items.append((t_name, inst, w))

        if not all_items:
            return []

        # Sort all items by weight descending
        all_items.sort(key=lambda x: x[2], reverse=True)

        # Cluster items by pairwise mask IoU >= tau_cluster
        tau_cluster = self.config.tau_cluster
        clusters: List[List[Tuple[str, TeacherInstance, float]]] = []
        used = set()

        for i in range(len(all_items)):
            if i in used:
                continue
            curr_cluster = [all_items[i]]
            used.add(i)

            for j in range(i + 1, len(all_items)):
                if j in used:
                    continue
                # Compare IoU with cluster representative
                iou = self.mask_iou(all_items[i][1].mask_rle, all_items[j][1].mask_rle)
                if iou >= tau_cluster:
                    curr_cluster.append(all_items[j])
                    used.add(j)

            clusters.append(curr_cluster)

        # Process each cluster
        fused_results: List[FusedInstance] = []
        fused_id = 0

        for cluster in clusters:
            if len(cluster) == 1:
                # Solo detection
                t_name, inst, w = cluster[0]
                # Apply solo penalty kappa
                w_solo = w * self.config.solo_penalty
                fused_results.append(FusedInstance(
                    fused_id=fused_id,
                    canonical_class_id=inst.canonical_class_id,
                    canonical_class_name=inst.canonical_class_name,
                    box=inst.box,
                    mask_rle=inst.mask_rle,
                    mask_prob_downscaled=inst.mask_prob_downscaled,
                    effective_weight=w_solo,
                    contributing_teachers=[t_name],
                    is_fused=False,
                    disagreement_flag=False
                ))
                fused_id += 1
                continue

            # Multi-teacher cluster
            class_ids = [item[1].canonical_class_id for item in cluster]
            all_agree = (len(set(class_ids)) == 1)

            if all_agree:
                # Teachers agree on class -> weighted fusion
                cid = class_ids[0]
                cname = cluster[0][1].canonical_class_name
                weights = np.array([item[2] for item in cluster], dtype=np.float32)
                total_w = float(np.sum(weights))
                norm_w = weights / total_w if total_w > 0 else np.ones_like(weights) / len(weights)

                # 1. Weighted Box Fusion (WBF)
                fused_box = [0.0, 0.0, 0.0, 0.0]
                for idx, item in enumerate(cluster):
                    b = item[1].box
                    for coord_idx in range(4):
                        fused_box[coord_idx] += float(norm_w[idx] * b[coord_idx])

                # 2. Weighted Mask Probability Fusion
                fused_downscaled = None
                down_masks = [item[1].mask_prob_downscaled for item in cluster if item[1].mask_prob_downscaled is not None]
                if down_masks and len(down_masks) == len(cluster):
                    # Check shape consistency
                    first_shape = down_masks[0].shape
                    if all(m.shape == first_shape for m in down_masks):
                        fused_float = np.zeros(first_shape, dtype=np.float32)
                        for idx, m in enumerate(down_masks):
                            fused_float += norm_w[idx] * m.astype(np.float32)
                        fused_downscaled = np.clip(fused_float, 0, 255).astype(np.uint8)

                # Representative mask RLE: use highest weight teacher's mask
                best_item = max(cluster, key=lambda x: x[2])
                fused_rle = best_item[1].mask_rle

                fused_results.append(FusedInstance(
                    fused_id=fused_id,
                    canonical_class_id=cid,
                    canonical_class_name=cname,
                    box=fused_box,
                    mask_rle=fused_rle,
                    mask_prob_downscaled=fused_downscaled,
                    effective_weight=min(total_w, 1.0),
                    contributing_teachers=[item[0] for item in cluster],
                    is_fused=True,
                    disagreement_flag=False
                ))
                fused_id += 1

            else:
                # Class conflict!
                # 1. If GT is available, check which teacher matches GT
                best_match = None
                if gt_canonical_classes:
                    for item in cluster:
                        if item[1].canonical_class_id in gt_canonical_classes:
                            best_match = item
                            break

                if best_match is not None:
                    # GT arbitrated
                    t_name, inst, w = best_match
                    fused_results.append(FusedInstance(
                        fused_id=fused_id,
                        canonical_class_id=inst.canonical_class_id,
                        canonical_class_name=inst.canonical_class_name,
                        box=inst.box,
                        mask_rle=inst.mask_rle,
                        mask_prob_downscaled=inst.mask_prob_downscaled,
                        effective_weight=w,
                        contributing_teachers=[t_name],
                        is_fused=False,
                        disagreement_flag=True
                    ))
                    fused_id += 1
                else:
                    # Margin check: keep highest weight teacher only if w_max / (w_min + eps) >= disagreement_ratio
                    cluster_sorted = sorted(cluster, key=lambda x: x[2], reverse=True)
                    w_max = cluster_sorted[0][2]
                    w_min = cluster_sorted[-1][2]
                    ratio = w_max / (w_min + 1e-4)
                    if ratio >= self.config.disagreement_ratio:
                        top_t, top_inst, top_w = cluster_sorted[0]
                        fused_results.append(FusedInstance(
                            fused_id=fused_id,
                            canonical_class_id=top_inst.canonical_class_id,
                            canonical_class_name=top_inst.canonical_class_name,
                            box=top_inst.box,
                            mask_rle=top_inst.mask_rle,
                            mask_prob_downscaled=top_inst.mask_prob_downscaled,
                            effective_weight=top_w,
                            contributing_teachers=[top_t],
                            is_fused=False,
                            disagreement_flag=True
                        ))
                        fused_id += 1
                    # Else: Discard cluster! Do not teach student ambiguous noise.

        return fused_results

