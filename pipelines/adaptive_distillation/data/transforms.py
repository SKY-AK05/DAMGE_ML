"""
Transforms module for synchronous geometric and photometric augmentation.
Guarantees that images, ground-truth annotations, and cached teacher instances
undergo identical geometric transformations (L0 to L2 augmentation ladder).
"""

from typing import Dict, List, Tuple, Any, Optional
import random
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from pipelines.adaptive_distillation.cache.cache_reader import CachedInstanceRecord


class AlignedTransforms:
    """Synchronously applies identical geometric transformations to image, GT, and teacher targets."""

    def __init__(self, level: str = "L0", hflip_prob: float = 0.5, target_size: int = 640):
        self.level = level.upper()
        self.hflip_prob = hflip_prob
        self.target_size = target_size

    def __call__(
        self,
        image_np: np.ndarray,
        gt_boxes: np.ndarray,                  # [N, 4] xyxy
        gt_masks: List[np.ndarray],            # [N, H, W] uint8
        teacher_instances: List[CachedInstanceRecord]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[CachedInstanceRecord]]:
        """
        Transforms image and all target sets in lockstep.
        """
        h, w = image_np.shape[:2]

        # 1. Base resize to canonical target_size (preserving aspect ratio with letterboxing)
        r = min(self.target_size / h, self.target_size / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2

        # Resize image
        img_resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=image_np.dtype)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized

        # Scale GT boxes
        transformed_gt_boxes = np.zeros_like(gt_boxes)
        if len(gt_boxes) > 0:
            transformed_gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * r + pad_x
            transformed_gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * r + pad_y

        # Scale GT masks
        transformed_gt_masks = []
        for m in gt_masks:
            m_res = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            m_pad = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            m_pad[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = m_res
            transformed_gt_masks.append(m_pad)

        # Scale Teacher instances
        transformed_teachers = []
        for t_inst in teacher_instances:
            # Scale box
            bx1, by1, bx2, by2 = t_inst.box
            new_box = [
                bx1 * r + pad_x,
                by1 * r + pad_y,
                bx2 * r + pad_x,
                by2 * r + pad_y
            ]

            # Decode mask, resize, pad, and re-encode to RLE
            m_orig = mask_utils.decode(t_inst.mask_rle)
            m_res = cv2.resize(m_orig, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            m_pad = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            m_pad[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = m_res

            new_rle = mask_utils.encode(np.asfortranarray(m_pad))
            new_rle["counts"] = new_rle["counts"].decode("utf-8")

            # Scale downscaled mask prob if present
            new_down = None
            if t_inst.mask_prob_downscaled is not None:
                # Resize downscaled mask accordingly
                new_down = cv2.resize(m_pad, (self.target_size // 4, self.target_size // 4), interpolation=cv2.INTER_AREA)

            transformed_teachers.append(CachedInstanceRecord(
                canonical_class_id=t_inst.canonical_class_id,
                confidence=t_inst.confidence,
                box=new_box,
                mask_rle=new_rle,
                mask_prob_downscaled=new_down,
                omega=t_inst.omega,
                status=t_inst.status
            ))

        # 2. Geometric level L2: Horizontal Flip
        if self.level in ("L2", "L3") and random.random() < self.hflip_prob:
            canvas = np.fliplr(canvas).copy()

            if len(transformed_gt_boxes) > 0:
                old_x1 = transformed_gt_boxes[:, 0].copy()
                old_x2 = transformed_gt_boxes[:, 2].copy()
                transformed_gt_boxes[:, 0] = self.target_size - old_x2
                transformed_gt_boxes[:, 2] = self.target_size - old_x1

            transformed_gt_masks = [np.fliplr(m).copy() for m in transformed_gt_masks]

            flipped_teachers = []
            for t_inst in transformed_teachers:
                bx1, by1, bx2, by2 = t_inst.box
                f_box = [self.target_size - bx2, by1, self.target_size - bx1, by2]

                m_dec = mask_utils.decode(t_inst.mask_rle)
                m_flip = np.fliplr(m_dec).copy()
                f_rle = mask_utils.encode(np.asfortranarray(m_flip))
                f_rle["counts"] = f_rle["counts"].decode("utf-8")

                f_down = None
                if t_inst.mask_prob_downscaled is not None:
                    f_down = np.fliplr(t_inst.mask_prob_downscaled).copy()

                flipped_teachers.append(CachedInstanceRecord(
                    canonical_class_id=t_inst.canonical_class_id,
                    confidence=t_inst.confidence,
                    box=f_box,
                    mask_rle=f_rle,
                    mask_prob_downscaled=f_down,
                    omega=t_inst.omega,
                    status=t_inst.status
                ))
            transformed_teachers = flipped_teachers

        # 3. Photometric level L1 (image only)
        if self.level in ("L1", "L2", "L3"):
            # Slight random brightness/contrast
            alpha = random.uniform(0.9, 1.1)
            beta = random.randint(-10, 10)
            canvas = np.clip(canvas.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return canvas, transformed_gt_boxes, transformed_gt_masks, transformed_teachers

