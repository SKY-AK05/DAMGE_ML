"""
Teacher Inference module.
Executes batch/image inference under deterministic preprocessing and permissive thresholds,
exporting standardized detection instances with boxes, masks, and soft target fields.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from pipelines.adaptive_distillation.teachers.teacher_loader import LoadedTeacher
from pipelines.adaptive_distillation.matching.taxonomy_mapper import TaxonomyMapper


@dataclass
class TeacherInstance:
    instance_id: int
    raw_class_id: int
    raw_class_name: str
    canonical_class_id: Optional[int]
    canonical_class_name: Optional[str]
    confidence: float
    box: List[float]                       # [x1, y1, x2, y2]
    mask_rle: Dict[str, Any]               # pycocotools RLE
    mask_prob_downscaled: Optional[np.ndarray] = None  # Downsampled uint8 or float16 mask probability
    soft_class_probs: Optional[List[float]] = None     # Calibrated class distribution


@dataclass
class ImageTeacherOutput:
    image_id: str
    file_name: str
    width: int
    height: int
    teacher_name: str
    instances: List[TeacherInstance]
    is_empty: bool                         # True if teacher produced 0 valid detections above floor


class TeacherInference:
    """Runs inference with permissive thresholds to populate calibration and distillation cache."""

    def __init__(self, mapper: Optional[TaxonomyMapper] = None, mask_downscale: float = 0.25):
        self.mapper = mapper or TaxonomyMapper()
        self.mask_downscale = mask_downscale

    def infer_yolo(
        self, loaded: LoadedTeacher, img_path: str, image_id: str
    ) -> ImageTeacherOutput:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        h, w = img_bgr.shape[:2]

        conf_floor = loaded.config.conf_floor
        nms_iou = loaded.config.nms_iou

        # YOLO predict
        results = loaded.model.predict(
            source=img_path,
            conf=conf_floor,
            iou=nms_iou,
            save=False,
            verbose=False,
            device=loaded.device
        )
        r = results[0]

        instances: List[TeacherInstance] = []
        if r.masks is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            # masks: r.masks.data is shape [N, H_mask, W_mask] float tensor
            raw_masks = r.masks.data.cpu().numpy()

            for inst_id, (box, conf, cls_id, raw_mask) in enumerate(zip(boxes, confs, clss, raw_masks)):
                if cls_id >= len(loaded.class_names):
                    continue
                raw_name = loaded.class_names[cls_id]
                can_id, can_name = self.mapper.map_teacher_class_to_canonical(cls_id, loaded.class_names)

                # Resize mask to full image dimensions
                mask_full = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                binary_mask = (mask_full > 0.5).astype(np.uint8)

                if binary_mask.sum() == 0:
                    continue

                # RLE
                rle = mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                # Downscaled mask for soft KD
                if self.mask_downscale < 1.0:
                    down_h, down_w = max(1, int(h * self.mask_downscale)), max(1, int(w * self.mask_downscale))
                    down_prob = cv2.resize(mask_full, (down_w, down_h), interpolation=cv2.INTER_AREA)
                    down_prob_uint8 = np.clip(down_prob * 255.0, 0, 255).astype(np.uint8)
                else:
                    down_prob_uint8 = np.clip(mask_full * 255.0, 0, 255).astype(np.uint8)

                instances.append(TeacherInstance(
                    instance_id=inst_id,
                    raw_class_id=cls_id,
                    raw_class_name=raw_name,
                    canonical_class_id=can_id,
                    canonical_class_name=can_name,
                    confidence=float(conf),
                    box=[float(b) for b in box],
                    mask_rle=rle,
                    mask_prob_downscaled=down_prob_uint8
                ))

        return ImageTeacherOutput(
            image_id=image_id,
            file_name=img_path.split("/")[-1],
            width=w,
            height=h,
            teacher_name=loaded.config.name,
            instances=instances,
            is_empty=(len(instances) == 0)
        )

    def infer_mask2former(
        self, loaded: LoadedTeacher, img_path: str, image_id: str
    ) -> ImageTeacherOutput:
        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size

        conf_floor = loaded.config.conf_floor
        processor = loaded.processor
        model = loaded.model

        inputs = processor(images=img_pil, return_tensors="pt").to(loaded.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Post process instance segmentation
        results = processor.post_process_instance_segmentation(
            outputs, target_sizes=[(h, w)], threshold=conf_floor
        )[0]

        instances: List[TeacherInstance] = []
        seg_info = results.get("segments_info", [])
        seg_map = results.get("segmentation").cpu().numpy()

        for inst_id, seg in enumerate(seg_info):
            lbl_id = seg["label_id"]
            score = float(seg.get("score", 1.0))
            if score < conf_floor:
                continue

            raw_name = loaded.class_names[lbl_id] if lbl_id < len(loaded.class_names) else f"class_{lbl_id}"
            can_id, can_name = self.mapper.map_teacher_class_to_canonical(lbl_id, loaded.class_names)

            binary_mask = (seg_map == seg["id"]).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue

            # Compute box from mask
            ys, xs = np.where(binary_mask)
            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

            # RLE
            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            # Downscaled mask
            if self.mask_downscale < 1.0:
                down_h, down_w = max(1, int(h * self.mask_downscale)), max(1, int(w * self.mask_downscale))
                down_prob = cv2.resize(binary_mask.astype(np.float32), (down_w, down_h), interpolation=cv2.INTER_AREA)
                down_prob_uint8 = np.clip(down_prob * 255.0, 0, 255).astype(np.uint8)
            else:
                down_prob_uint8 = binary_mask * 255

            instances.append(TeacherInstance(
                instance_id=inst_id,
                raw_class_id=lbl_id,
                raw_class_name=raw_name,
                canonical_class_id=can_id,
                canonical_class_name=can_name,
                confidence=score,
                box=box,
                mask_rle=rle,
                mask_prob_downscaled=down_prob_uint8
            ))

        return ImageTeacherOutput(
            image_id=image_id,
            file_name=img_path.split("/")[-1],
            width=w,
            height=h,
            teacher_name=loaded.config.name,
            instances=instances,
            is_empty=(len(instances) == 0)
        )

    def infer_image(
        self, loaded: LoadedTeacher, img_path: str, image_id: str
    ) -> ImageTeacherOutput:
        arch = loaded.config.arch.lower()
        if "yolo" in arch:
            return self.infer_yolo(loaded, img_path, image_id)
        elif "mask2former" in arch or "m2f" in arch:
            return self.infer_mask2former(loaded, img_path, image_id)
        else:
            raise NotImplementedError(f"Inference not implemented for arch: {arch}")

