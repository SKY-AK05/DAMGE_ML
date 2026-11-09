"""
Distillation Losses module — Mask2Former Student C.

Implements:
  L_total = L_GT + lambda(t) * L_KD

For Mask2Former:
  - L_GT:  model(pixel_values, mask_labels, class_labels).loss
           (internally: cross-entropy + mask + dice via Hungarian matching)
  - L_KD:  same forward pass with teacher pseudo-targets as (mask_labels, class_labels)
           Vetoed instances (omega=0) are excluded before constructing pseudo-targets.
           Teacher absence is never treated as background.

IMPORTANT — MaskRCNN label indexing removed:
  Mask2Former uses 0-indexed class labels (0..num_classes-1).
  There is NO background class index in the class_labels tensor.
  The 'no_object' token is handled internally by the model's Hungarian matcher.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipelines.adaptive_distillation.config import DistillationConfig
from pipelines.adaptive_distillation.cache.cache_reader import CachedInstanceRecord


@dataclass
class DistillationLossStats:
    loss_total: float
    loss_gt: float
    loss_kd: float
    effective_mass: float          # Sum of active teacher weights
    n_instances_distilled: int
    n_vetoed_ignored: int
    kd_gt_ratio: float = 0.0       # R_KD/GT = (lam * L_kd) / (L_gt + eps)
    mean_weight: float = 0.0       # bar(omega) = effective_mass / n_instances_distilled


class DistillationLoss(nn.Module):
    """
    Total objective: L_total = L_GT + lambda(t) * L_KD

    Works with Mask2Former student via HuggingFace transformers API.
    """

    def __init__(
        self,
        config: DistillationConfig,
        num_classes: int = 32,
        tau: float = 2.0,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.tau = tau
        self.eps = 1e-6

    def _build_m2f_targets(
        self,
        gt_or_kd_items: List[Dict],
        device: torch.device,
        image_hw: Tuple[int, int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Converts a list of per-image dicts with keys 'boxes', 'labels', 'masks'
        (or pre-built mask_labels / class_labels) into the lists expected by
        Mask2Former:
          mask_labels:  list[Tensor(N_inst, H, W)] float, one per image
          class_labels: list[Tensor(N_inst,)]      int64, one per image

        Classes are 0-indexed. No +1 background offset.
        """
        H, W = image_hw
        mask_labels_batch = []
        class_labels_batch = []

        for item in gt_or_kd_items:
            if item is None:
                # No instances: single empty tensor
                mask_labels_batch.append(
                    torch.zeros((0, H, W), dtype=torch.float32, device=device)
                )
                class_labels_batch.append(
                    torch.zeros((0,), dtype=torch.int64, device=device)
                )
                continue

            labels = item.get("labels")
            masks = item.get("masks")
            boxes = item.get("boxes")

            if labels is None or len(labels) == 0:
                mask_labels_batch.append(
                    torch.zeros((0, H, W), dtype=torch.float32, device=device)
                )
                class_labels_batch.append(
                    torch.zeros((0,), dtype=torch.int64, device=device)
                )
                continue

            labels_t = labels.to(device).long()

            # Mask2Former expects 0-indexed class labels
            # If labels arrived 1-based (from MaskRCNN-style data), shift back
            if labels_t.numel() > 0 and labels_t.min() >= 1:
                labels_t = labels_t - 1
            labels_t = labels_t.clamp(0, self.num_classes - 1)

            if masks is not None and len(masks) > 0:
                m_t = masks.to(device).float()
                if m_t.shape[-2:] != (H, W):
                    m_t = F.interpolate(
                        m_t.unsqueeze(0), size=(H, W), mode="nearest"
                    ).squeeze(0)
            else:
                # Fall back to box masks if no segmentation masks available
                m_t = torch.zeros((len(labels_t), H, W), dtype=torch.float32, device=device)
                if boxes is not None and len(boxes) > 0:
                    boxes_t = boxes.to(device)
                    for bi, box in enumerate(boxes_t):
                        x1, y1, x2, y2 = (
                            int(box[0].clamp(0, W - 1)),
                            int(box[1].clamp(0, H - 1)),
                            int(box[2].clamp(0, W)),
                            int(box[3].clamp(0, H)),
                        )
                        if x2 > x1 and y2 > y1:
                            m_t[bi, y1:y2, x1:x2] = 1.0

            mask_labels_batch.append(m_t)
            class_labels_batch.append(labels_t)

        return mask_labels_batch, class_labels_batch

    def compute_gt_loss(
        self,
        student_model: nn.Module,
        pixel_values: torch.Tensor,
        gt_targets: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Computes standard Mask2Former supervised loss on ground truth.

        Args:
            student_model: Mask2FormerForUniversalSegmentation in train mode.
            pixel_values:  Preprocessed image batch (B, 3, H_proc, W_proc).
            gt_targets:    List of dicts with 'boxes', 'labels', 'masks'.
        """
        H, W = pixel_values.shape[2], pixel_values.shape[3]
        device = pixel_values.device

        mask_labels, class_labels = self._build_m2f_targets(gt_targets, device, (H, W))

        outputs = student_model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        if outputs.loss is None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return outputs.loss

    def compute_kd_loss(
        self,
        student_model: nn.Module,
        pixel_values: torch.Tensor,
        batch_teacher_instances: List[List[CachedInstanceRecord]],
    ) -> Tuple[torch.Tensor, float, int, int]:
        """
        Computes KD loss using accepted teacher instances as pseudo-GT targets.

        Veto policy:
          - omega <= 0: instance excluded (GT mismatch or poor localization)
          - canonical_class_id is None: instance excluded (taxonomy gap)
          - Teacher absence: ignored (never treated as background)

        Returns:
          (kd_loss, effective_mass, n_distilled, n_vetoed)
        """
        device = pixel_values.device
        H, W = pixel_values.shape[2], pixel_values.shape[3]
        effective_mass = 0.0
        n_distilled = 0
        n_vetoed = 0

        kd_targets = []
        active_indices = []

        for img_idx, inst_list in enumerate(batch_teacher_instances):
            img_boxes, img_labels, img_weights, img_masks = [], [], [], []

            for inst in inst_list:
                w_val = inst.omega
                if w_val <= 0.0 or inst.canonical_class_id is None:
                    n_vetoed += 1
                    continue
                if inst.canonical_class_id < 0 or inst.canonical_class_id >= self.num_classes:
                    n_vetoed += 1
                    continue

                img_boxes.append(inst.box)
                img_labels.append(inst.canonical_class_id)   # 0-indexed for M2F
                img_weights.append(w_val)
                effective_mass += w_val
                n_distilled += 1

                # Decode binary mask from RLE
                decoded_mask = None
                if inst.mask_rle and inst.mask_rle.get("counts"):
                    try:
                        from pycocotools import mask as mask_utils
                        rle_in = dict(inst.mask_rle)
                        m_dec = mask_utils.decode(rle_in)
                        if m_dec.shape != (H, W):
                            import cv2
                            m_dec = cv2.resize(m_dec, (W, H), interpolation=cv2.INTER_NEAREST)
                        decoded_mask = m_dec.astype(np.float32)
                    except Exception:
                        decoded_mask = None

                if decoded_mask is None:
                    # Box fallback
                    m_box = np.zeros((H, W), dtype=np.float32)
                    bx1, by1, bx2, by2 = [int(round(c)) for c in inst.box]
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(W, bx2), min(H, by2)
                    if bx2 > bx1 and by2 > by1:
                        m_box[by1:by2, bx1:bx2] = 1.0
                    decoded_mask = m_box

                img_masks.append(torch.from_numpy(decoded_mask))

            if img_boxes:
                kd_targets.append({
                    "labels": torch.tensor(img_labels, dtype=torch.int64, device=device),
                    "masks":  torch.stack(img_masks, dim=0).to(device),
                    "weights": torch.tensor(img_weights, dtype=torch.float32, device=device),
                })
                active_indices.append(img_idx)
            else:
                kd_targets.append(None)

        if effective_mass <= self.eps or n_distilled == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0, n_vetoed

        if not active_indices:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0, n_vetoed

        # Build pseudo-GT lists for the active sub-batch
        sub_pv = pixel_values[active_indices]
        H_pv, W_pv = sub_pv.shape[2], sub_pv.shape[3]

        sub_mask_labels = []
        sub_class_labels = []
        for idx in active_indices:
            t = kd_targets[idx]
            if t is None:
                sub_mask_labels.append(torch.zeros((0, H_pv, W_pv), dtype=torch.float32, device=device))
                sub_class_labels.append(torch.zeros((0,), dtype=torch.int64, device=device))
            else:
                ml = t["masks"].float()
                if ml.shape[-2:] != (H_pv, W_pv):
                    ml = F.interpolate(ml.unsqueeze(0), size=(H_pv, W_pv), mode="nearest").squeeze(0)
                sub_mask_labels.append(ml)
                sub_class_labels.append(t["labels"])

        # Student forward on pseudo-targets
        kd_outputs = student_model(
            pixel_values=sub_pv,
            mask_labels=sub_mask_labels,
            class_labels=sub_class_labels,
        )

        if kd_outputs.loss is None:
            return torch.tensor(0.0, device=device, requires_grad=True), effective_mass, n_distilled, n_vetoed

        raw_kd_loss = kd_outputs.loss

        # Scale by mean instance reliability weight
        mean_instance_weight = float(effective_mass / max(n_distilled, 1))
        norm_kd_loss = (raw_kd_loss / max(len(active_indices), 1)) * mean_instance_weight

        return norm_kd_loss, effective_mass, n_distilled, n_vetoed

    def forward(
        self,
        student_model: nn.Module,
        pixel_values: torch.Tensor,
        gt_targets: List[Dict[str, torch.Tensor]],
        batch_teacher_instances: List[List[CachedInstanceRecord]],
        lam: float = 1.0,
    ) -> Tuple[torch.Tensor, DistillationLossStats]:
        """
        L_total = L_GT + lam * L_KD

        Args:
            student_model:            Mask2Former in train mode.
            pixel_values:             Preprocessed image batch (B, 3, H, W).
            gt_targets:               Per-image GT dicts with boxes/labels/masks.
            batch_teacher_instances:  Per-image lists of CachedInstanceRecord.
            lam:                      Current KD weight (from lambda schedule).
        """
        loss_gt = self.compute_gt_loss(student_model, pixel_values, gt_targets)

        if lam > 0.0 and self.config.enabled:
            loss_kd, eff_mass, n_dist, n_vetoed = self.compute_kd_loss(
                student_model, pixel_values, batch_teacher_instances
            )
        else:
            loss_kd = torch.tensor(0.0, device=pixel_values.device)
            eff_mass = 0.0
            n_dist = 0
            n_vetoed = 0

        loss_total = loss_gt + lam * loss_kd

        gt_val = float(loss_gt.item())
        kd_val = float(loss_kd.item())
        ratio = float((lam * kd_val) / (gt_val + self.eps))
        mean_w = float(eff_mass / max(n_dist, 1))

        stats = DistillationLossStats(
            loss_total=float(loss_total.item()),
            loss_gt=gt_val,
            loss_kd=kd_val,
            effective_mass=eff_mass,
            n_instances_distilled=n_dist,
            n_vetoed_ignored=n_vetoed,
            kd_gt_ratio=ratio,
            mean_weight=mean_w,
        )

        return loss_total, stats
