"""
Dataset Adapter module — Mask2Former Student C.

Streams images, ground-truth annotations, and cached teacher predictions.
Integrates Mask2FormerImageProcessor to produce pixel_values alongside
raw GT boxes/masks/labels for the evaluator.

The M2F processor handles:
  - Resize to model input size
  - Normalize with ImageNet mean/std
  - Pad to fixed size

Both `pixel_values` (for student forward pass) and `gt` (boxes/labels/masks
for COCO evaluator) are returned per sample.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils

from pipelines.adaptive_distillation.cache.cache_reader import CacheReader, CachedImageRecord, CachedInstanceRecord
from pipelines.adaptive_distillation.data.transforms import AlignedTransforms
from pipelines.adaptive_distillation.matching.teacher_fusion import TeacherFusion


class DistillationDataset(Dataset):
    """
    Integrates image loading, ground truth, teacher caches, and M2F preprocessing.

    Returns per-sample dict:
      - 'pixel_values':  Tensor(3, H_proc, W_proc) — M2F preprocessed
      - 'gt':            Dict with 'boxes', 'labels', 'masks', 'image_id' — raw COCO format
      - 'teachers':      Dict[teacher_name -> CachedImageRecord]
      - 'transformed_teacher_instances': List[CachedInstanceRecord] — fused/filtered teacher instances
      - 'image_id':      str
    """

    def __init__(
        self,
        images_dir,
        coco_json_path,
        cache_readers: Optional[Dict[str, CacheReader]] = None,
        transforms: Optional[AlignedTransforms] = None,
        num_classes: int = 32,
        processor=None,     # Mask2FormerImageProcessor; if None, raw normalized tensor returned
        fusion_engine: Optional[TeacherFusion] = None,
    ):
        self.images_dir = Path(images_dir)
        self.coco_json_path = Path(coco_json_path)
        self.cache_readers = cache_readers or {}
        self.transforms = transforms or AlignedTransforms(level="L0")
        self.num_classes = num_classes
        self.processor = processor
        self.fusion_engine = fusion_engine

        with open(self.coco_json_path, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self.images_by_id = {str(img["id"]): img for img in self.coco_data.get("images", [])}
        self.anns_by_image: Dict[str, List[Dict[str, Any]]] = {}
        for ann in self.coco_data.get("annotations", []):
            iid = str(ann["image_id"])
            self.anns_by_image.setdefault(iid, []).append(ann)

        self.image_ids = sorted(list(self.images_by_id.keys()))

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = self.image_ids[idx]
        img_info = self.images_by_id[image_id]
        img_path = self.images_dir / img_info["file_name"]

        # Load image as numpy RGB
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Load GT annotations
        anns = self.anns_by_image.get(image_id, [])
        gt_boxes, gt_classes, gt_masks = [], [], []

        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            gt_boxes.append([bx, by, bx + bw, by + bh])
            gt_classes.append(int(ann["category_id"]))

            seg = ann.get("segmentation")
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                rle = mask_utils.merge(rles)
            elif isinstance(seg, dict):
                rle = seg
            else:
                continue
            m = mask_utils.decode(rle)
            gt_masks.append(m.astype(np.uint8))

        gt_boxes_np = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
        gt_classes_np = np.array(gt_classes, dtype=np.int64) if gt_classes else np.zeros((0,), dtype=np.int64)

        # Load cached teacher predictions
        teacher_records: Dict[str, CachedImageRecord] = {}
        for t_name, reader in self.cache_readers.items():
            try:
                rec = reader.read_image(image_id)
                teacher_records[t_name] = rec
            except Exception:
                teacher_records[t_name] = CachedImageRecord(
                    image_id=image_id, width=w, height=h, is_empty=True, instances=[]
                )

        # Multi-teacher fusion vs single-teacher streaming
        all_teacher_instances = []
        if len(self.cache_readers) > 1 and self.fusion_engine is not None:
            by_teacher = {}
            for t_name, rec in teacher_records.items():
                by_teacher[t_name] = [(inst, inst.omega) for inst in rec.instances]
            fused_list = self.fusion_engine.fuse_image_instances(by_teacher, gt_canonical_classes=list(gt_classes_np))
            for fi in fused_list:
                if fi.effective_weight > 0.0 and fi.canonical_class_id is not None:
                    all_teacher_instances.append(CachedInstanceRecord(
                        canonical_class_id=fi.canonical_class_id,
                        confidence=fi.effective_weight,
                        box=fi.box,
                        mask_rle=fi.mask_rle,
                        mask_prob_downscaled=fi.mask_prob_downscaled,
                        omega=fi.effective_weight,
                        status="FUSED" if fi.is_fused else "SOLO",
                    ))
        else:
            for t_name, rec in teacher_records.items():
                all_teacher_instances.extend(rec.instances)

        # Apply spatial/photometric transforms (L0 = none, L1+ = augmentation)
        trans_img, trans_boxes, trans_masks, trans_teachers = self.transforms(
            img_rgb, gt_boxes_np, gt_masks, all_teacher_instances
        )

        # Build GT target dict (raw COCO format for evaluator)
        masks_tensor = (
            torch.from_numpy(np.stack(trans_masks, axis=0))
            if trans_masks
            else torch.zeros((0, trans_img.shape[0], trans_img.shape[1]), dtype=torch.uint8)
        )
        gt_target = {
            "boxes": torch.from_numpy(trans_boxes).float(),
            "labels": torch.from_numpy(gt_classes_np).long(),
            "masks": masks_tensor,
            "image_id": image_id,
        }

        # Preprocess image for Mask2Former
        if self.processor is not None:
            pil_img = PILImage.fromarray(trans_img)
            inputs = self.processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)   # (3, H_proc, W_proc)
        else:
            # Fallback: normalize to [0,1] float32 (3, H, W) — not M2F compatible but
            # avoids crash if processor is not set during unit tests
            pixel_values = torch.from_numpy(trans_img).permute(2, 0, 1).float() / 255.0

        return {
            "pixel_values": pixel_values,
            "gt": gt_target,
            "teachers": teacher_records,
            "transformed_teacher_instances": trans_teachers,
            "image_id": image_id,
            # Keep raw image tensor for any evaluator that needs it
            "image": torch.from_numpy(trans_img).permute(2, 0, 1).float() / 255.0,
        }


def distillation_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator handling variable instance counts per image."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    images = torch.stack([item["image"] for item in batch], dim=0)
    gts = [item["gt"] for item in batch]
    teachers = [item["teachers"] for item in batch]
    teacher_instances = [item["transformed_teacher_instances"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "images": images,
        "gts": gts,
        "teachers": teachers,
        "teacher_instances": teacher_instances,
        "image_ids": image_ids,
    }
