"""
Evaluator module — Mask2Former Student C.

Runs inference with Mask2FormerForUniversalSegmentation and evaluates
using pycocotools COCOeval (box mAP + mask mAP).

Mask2Former post-processing uses:
  processor.post_process_instance_segmentation(outputs, target_sizes=[(H, W)])
which returns per-image dicts with 'segments_info' and 'segmentation' (panoptic map).
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from pipelines.adaptive_distillation.config import EvaluationConfig
from pipelines.adaptive_distillation.matching.taxonomy_mapper import CLASSES_23


class PipelineEvaluator:
    """Evaluates Mask2Former student on COCO-format datasets."""

    def __init__(
        self,
        val_json_path,
        class_names: List[str] = CLASSES_23,
        config: Optional[EvaluationConfig] = None,
    ):
        self.val_json_path = str(val_json_path)
        self.class_names = list(class_names)
        self.config = config or EvaluationConfig()
        self.coco_gt = COCO(self.val_json_path)

    def run_m2f_inference(
        self,
        model: nn.Module,
        val_images_dir: str,
        device: str = "cuda",
        conf_thresh: float = 0.001,
    ) -> List[Dict[str, Any]]:
        """
        Runs Mask2Former inference on all images in the COCO val set.

        Returns:
            predictions: list of COCO-format dicts with
              image_id, category_id, bbox, score, segmentation (RLE)
        """
        import json as _json
        processor = getattr(model, "processor", None)
        if processor is None:
            raise RuntimeError(
                "Student model has no .processor attribute. "
                "Make sure StudentBuilder.build() was used."
            )

        with open(self.val_json_path, "r", encoding="utf-8") as f:
            val_coco = _json.load(f)

        # Build label_id -> COCO category_id mapping
        # M2F output: 0-indexed label_id -> class name -> COCO category_id
        id2label = getattr(model.config, "id2label", {})
        coco_cats = {c["name"]: c["id"] for c in val_coco.get("categories", [])}
        label_id_to_cat_id: Dict[int, int] = {}
        for lid, name in id2label.items():
            lid_int = int(lid)
            if name in coco_cats:
                label_id_to_cat_id[lid_int] = coco_cats[name]

        val_images = val_coco.get("images", [])
        predictions = []

        model.eval()
        val_dir = Path(val_images_dir)

        with torch.no_grad():
            for img_info in val_images:
                iid = img_info["id"]
                fname = img_info["file_name"]
                img_path = val_dir / fname
                if not img_path.exists():
                    continue

                orig_w, orig_h = img_info.get("width", 0), img_info.get("height", 0)
                pil_img = PILImage.open(img_path).convert("RGB")
                if orig_w == 0 or orig_h == 0:
                    orig_w, orig_h = pil_img.size

                inputs = processor(images=pil_img, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)

                outputs = model(pixel_values=pixel_values)

                results = processor.post_process_instance_segmentation(
                    outputs,
                    target_sizes=[(orig_h, orig_w)],
                    threshold=conf_thresh,
                )[0]

                seg_map = results.get("segmentation")
                if seg_map is not None:
                    seg_map_np = seg_map.cpu().numpy()
                else:
                    seg_map_np = None

                for seg in results.get("segments_info", []):
                    score = float(seg.get("score", 1.0))
                    if score < conf_thresh:
                        continue
                    label_id = int(seg["label_id"])
                    cat_id = label_id_to_cat_id.get(label_id, label_id + 1)

                    if seg_map_np is not None:
                        binary_mask = (seg_map_np == seg["id"]).astype(np.uint8)
                    else:
                        binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

                    if binary_mask.sum() == 0:
                        continue

                    # Bounding box from mask
                    ys, xs = np.where(binary_mask)
                    x1, y1 = int(xs.min()), int(ys.min())
                    x2, y2 = int(xs.max()), int(ys.max())
                    bw = max(0, x2 - x1)
                    bh = max(0, y2 - y1)

                    # RLE encode
                    rle = mask_utils.encode(np.asfortranarray(binary_mask))
                    rle["counts"] = rle["counts"].decode("utf-8")

                    predictions.append({
                        "image_id": int(iid),
                        "category_id": cat_id,
                        "bbox": [float(x1), float(y1), float(bw), float(bh)],
                        "score": score,
                        "segmentation": rle,
                    })

        return predictions

    def evaluate_predictions(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluates a list of COCO-format predictions against ground truth.
        Returns mask mAP50, mask mAP50-95, box mAP50, box mAP50-95, per-class AP50.
        """
        if not predictions:
            return {
                "mask_mAP50": 0.0,
                "mask_mAP50_95": 0.0,
                "box_mAP50": 0.0,
                "box_mAP50_95": 0.0,
                "per_class": {},
            }

        formatted_preds = []
        for p in predictions:
            iid = p["image_id"]
            if isinstance(iid, str) and iid.isdigit():
                iid = int(iid)
            formatted_preds.append({
                "image_id": iid,
                "category_id": p["category_id"],
                "bbox": [round(float(x), 2) for x in p["bbox"]],
                "score": float(p["score"]),
                "segmentation": p.get("segmentation"),
            })

        coco_dt = self.coco_gt.loadRes(formatted_preds)

        # Box mAP
        coco_eval_box = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval_box.evaluate()
        coco_eval_box.accumulate()
        coco_eval_box.summarize()
        box_map50_95 = float(coco_eval_box.stats[0])
        box_map50 = float(coco_eval_box.stats[1])

        # Mask mAP
        coco_eval_mask = COCOeval(self.coco_gt, coco_dt, "segm")
        coco_eval_mask.evaluate()
        coco_eval_mask.accumulate()
        coco_eval_mask.summarize()
        mask_map50_95 = float(coco_eval_mask.stats[0])
        mask_map50 = float(coco_eval_mask.stats[1])

        # Per-class AP50
        per_class_results = {}
        cat_ids = sorted(self.coco_gt.getCatIds())
        for idx, cid in enumerate(cat_ids):
            cats = self.coco_gt.loadCats([cid])
            c_name = cats[0]["name"] if cats else (
                self.class_names[idx] if idx < len(self.class_names) else f"class_{cid}"
            )
            prec = coco_eval_mask.eval.get("precision")
            if prec is not None and idx < prec.shape[2]:
                ap50 = np.mean(prec[0, :, idx, 0, 2])
                per_class_results[c_name] = max(0.0, float(ap50)) if ap50 >= 0 else 0.0

        return {
            "mask_mAP50": mask_map50,
            "mask_mAP50_95": mask_map50_95,
            "box_mAP50": box_map50,
            "box_mAP50_95": box_map50_95,
            "per_class": per_class_results,
        }

    def compute_paired_bootstrap_ci(
        self,
        predictions_a: List[Dict[str, Any]],
        predictions_b: List[Dict[str, Any]],
        num_bootstrap: int = 200,
    ) -> Tuple[float, float, float]:
        all_img_ids = sorted(list(self.coco_gt.getImgIds()))
        rng = np.random.default_rng(self.config.seed)
        deltas = []
        for _ in range(num_bootstrap):
            sample_ids = set(rng.choice(all_img_ids, size=len(all_img_ids), replace=True))
            preds_a_sub = [p for p in predictions_a if p["image_id"] in sample_ids]
            preds_b_sub = [p for p in predictions_b if p["image_id"] in sample_ids]
            if not preds_a_sub or not preds_b_sub:
                continue
            score_a = float(np.mean([p["score"] for p in preds_a_sub]))
            score_b = float(np.mean([p["score"] for p in preds_b_sub]))
            deltas.append(score_b - score_a)
        if not deltas:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(deltas)),
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        )
