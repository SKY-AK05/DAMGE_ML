"""
Teacher Loader module.
Handles loading, freezing, hash computation, and memory release for YOLO,
Mask2Former, and Mask R-CNN models.
"""

import os
import gc
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from pipelines.adaptive_distillation.config import TeacherConfig
from pipelines.adaptive_distillation.matching.taxonomy_mapper import CLASSES_23, CLASSES_32


@dataclass
class LoadedTeacher:
    config: TeacherConfig
    model: Any
    processor: Optional[Any]
    class_names: List[str]
    device: str
    checkpoint_sha256: str


def compute_path_sha256(path: Path | str) -> str:
    """Computes SHA-256 hash of a file or directory of weights."""
    p = Path(path)
    h = hashlib.sha256()
    if p.is_file():
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
    elif p.is_dir():
        for file in sorted(p.rglob("*")):
            if file.is_file() and not file.name.startswith("."):
                h.update(file.name.encode())
                with open(file, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
    else:
        h.update(str(p).encode())
    return h.hexdigest()


class TeacherLoader:
    """Loads and freezes teachers sequentially, ensuring zero parameter updates."""

    @staticmethod
    def load(cfg: TeacherConfig, device: Optional[str] = None) -> LoadedTeacher:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_path = Path(cfg.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found at: {ckpt_path}")

        sha256_hash = compute_path_sha256(ckpt_path)
        processor = None
        model = None
        class_names = []

        # 1. YOLO
        if cfg.arch.lower() in ("yolo", "yolo11", "yolo11m-seg", "yolov8-seg"):
            from ultralytics import YOLO
            yolo_wrapper = YOLO(str(ckpt_path))
            model = yolo_wrapper
            # Get class names
            raw_names = getattr(yolo_wrapper.model, "names", {}) or {}
            if isinstance(raw_names, dict):
                class_names = [raw_names[i] for i in sorted(raw_names.keys())]
            else:
                class_names = list(raw_names)
            
            # Freeze underlying PyTorch module
            if hasattr(yolo_wrapper, "model") and isinstance(yolo_wrapper.model, torch.nn.Module):
                yolo_wrapper.model.eval()
                for param in yolo_wrapper.model.parameters():
                    param.requires_grad = False

        # 2. Mask2Former
        elif cfg.arch.lower() in ("mask2former", "m2f"):
            from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
            _proc_config = ckpt_path / "preprocessor_config.json"
            if _proc_config.exists():
                processor = Mask2FormerImageProcessor.from_pretrained(str(ckpt_path))
            else:
                processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
            
            model = Mask2FormerForUniversalSegmentation.from_pretrained(str(ckpt_path))
            model.to(device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            # Extract class names from id2label
            id2label = getattr(model.config, "id2label", {}) or {}
            if id2label:
                class_names = [id2label[str(i)] if str(i) in id2label else id2label.get(i, f"class_{i}") 
                               for i in range(len(id2label))]
            else:
                class_names = CLASSES_32 if cfg.class_map == "32_class" else CLASSES_23

        # 3. Mask R-CNN
        elif cfg.arch.lower() in ("maskrcnn", "mask_rcnn"):
            import torchvision
            from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

            class_names = CLASSES_23 if cfg.class_map == "23_class" else CLASSES_32
            num_classes = len(class_names) + 1  # background

            m = maskrcnn_resnet50_fpn_v2(weights=None)
            in_features = m.roi_heads.box_predictor.cls_score.in_features
            m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            in_features_mask = m.roi_heads.mask_predictor.conv5_mask.in_channels
            m.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
            m.load_state_dict(torch.load(str(ckpt_path), map_location=device))
            m.to(device)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            model = m

        else:
            raise ValueError(f"Unsupported teacher architecture: {cfg.arch}")

        return LoadedTeacher(
            config=cfg,
            model=model,
            processor=processor,
            class_names=class_names,
            device=device,
            checkpoint_sha256=sha256_hash
        )

    @staticmethod
    def unload(loaded: LoadedTeacher) -> None:
        """Completely frees model from GPU memory and runs garbage collection."""
        if hasattr(loaded.model, "to"):
            try:
                loaded.model.to("cpu")
            except Exception:
                pass
        loaded.model = None
        loaded.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

