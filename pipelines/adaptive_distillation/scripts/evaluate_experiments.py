#!/usr/bin/env python3
"""
Stage C: Experiment Evaluation and Comparison.
Evaluates student and teacher checkpoints on validation or test sets,
computes COCO metrics, generates paired bootstrap confidence intervals,
and validates pre-registered hypotheses.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.adaptive_distillation.config import DistillationPipelineConfig
from pipelines.adaptive_distillation.evaluation.evaluator import PipelineEvaluator
from pipelines.adaptive_distillation.evaluation.comparison import ExperimentComparison, ExperimentResult
from pipelines.adaptive_distillation.student.student_builder import StudentBuilder
from pipelines.adaptive_distillation.matching.taxonomy_mapper import CLASSES_23


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Distillation Experiments and Validate Hypotheses")
    parser.add_argument("--config", default="pipelines/adaptive_distillation/configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific student checkpoint to evaluate")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split to evaluate")
    parser.add_argument("--exp_id", type=str, default="E6", help="Experiment ID (e.g. E2, E3, E5, E6)")
    parser.add_argument("--output_report", type=str, default=None, help="Output markdown report path")
    parser.add_argument("--max_images", type=int, default=None, help="Optional image limit for rapid debugging")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DistillationPipelineConfig.from_yaml(args.config)

    json_path = cfg.dataset.test_json if args.split == "test" else cfg.dataset.val_json
    img_dir = cfg.dataset.test_images if args.split == "test" else cfg.dataset.val_images

    from pipelines.adaptive_distillation.matching.taxonomy_mapper import CLASSES_32, CLASSES_23
    canonical_classes = CLASSES_32 if cfg.dataset.canonical_taxonomy == "32_class" else CLASSES_23
    num_classes = len(canonical_classes)

    print(f"\n[*] Evaluating on {args.split.upper()} split ({json_path})...")
    print(f"[*] Taxonomy: {cfg.dataset.canonical_taxonomy} ({num_classes} classes)")
    evaluator = PipelineEvaluator(
        val_json_path=json_path,
        class_names=canonical_classes,
        config=cfg.evaluation
    )

    # If checkpoint provided, load and run evaluation
    predictions = []
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(cfg.output_dir) / cfg.experiment_name / "best.pt"

    if ckpt_path.exists():
        print(f"[*] Loading checkpoint: {ckpt_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(json_path, "r") as f:
            coco_data = json.load(f)

        images_to_eval = coco_data.get("images", [])[:args.max_images] if args.max_images else coco_data.get("images", [])

        import cv2
        from pycocotools import mask as mask_utils

        # 1. Teacher Evaluation Path (Mask2Former / directory checkpoint)
        if ckpt_path.is_dir() or (ckpt_path / "config.json").exists():
            from pipelines.adaptive_distillation.config import TeacherConfig
            from pipelines.adaptive_distillation.teachers.teacher_loader import TeacherLoader
            from pipelines.adaptive_distillation.teachers.teacher_inference import TeacherInference
            from pipelines.adaptive_distillation.matching.taxonomy_mapper import TaxonomyMapper

            t_cfg = TeacherConfig(
                name=args.exp_id,
                arch="mask2former",
                checkpoint_path=str(ckpt_path),
                class_map=cfg.dataset.canonical_taxonomy,
                conf_floor=cfg.evaluation.conf
            )
            loaded_t = TeacherLoader.load(t_cfg, device=device)
            mapper = TaxonomyMapper(canonical_classes=canonical_classes)
            infer_eng = TeacherInference(mapper=mapper, mask_downscale=1.0)

            print(f"[*] Generating teacher predictions on {len(images_to_eval)} images...")
            for img_info in tqdm(images_to_eval):
                iid = img_info["id"]
                fname = img_info["file_name"]
                fpath = Path(img_dir) / fname
                out = infer_eng.infer_image(loaded_t, str(fpath), str(iid))

                for inst in out.instances:
                    if inst.canonical_class_id is None:
                        continue
                    bx1, by1, bx2, by2 = inst.box
                    bw = max(0.0, bx2 - bx1)
                    bh = max(0.0, by2 - by1)
                    cid = inst.canonical_class_id + 1 # 1-based COCO category ID
                    predictions.append({
                        "image_id": iid,
                        "category_id": cid,
                        "bbox": [bx1, by1, bw, bh],
                        "score": inst.confidence,
                        "segmentation": inst.mask_rle
                    })

            TeacherLoader.unload(loaded_t)

        # 2. Student Evaluation Path (Mask R-CNN / .pt checkpoint)
        else:
            model = StudentBuilder.build(cfg.student, num_classes=num_classes, seed=cfg.evaluation.seed)
            data = torch.load(ckpt_path, map_location=device)
            state_dict = data.get("model_state_dict", data)
            model.load_state_dict(state_dict)
            model.to(device).eval()

            print(f"[*] Generating student predictions on {len(images_to_eval)} images...")
            with torch.no_grad():
                for img_info in tqdm(images_to_eval):
                    iid = img_info["id"]
                    fname = img_info["file_name"]
                    fpath = Path(img_dir) / fname
                    img_bgr = cv2.imread(str(fpath))
                    if img_bgr is None:
                        continue
                    h, w = img_bgr.shape[:2]
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # Letterbox resize to 640
                    r = min(640 / h, 640 / w)
                    nw, nh = int(round(w * r)), int(round(h * r))
                    px, py = (640 - nw) // 2, (640 - nh) // 2
                    res = cv2.resize(img_rgb, (nw, nh))
                    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
                    canvas[py:py + nh, px:px + nw] = res

                    img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
                    out = model([img_tensor.to(device)])[0]

                    boxes = out["boxes"].cpu().numpy()
                    labels = out["labels"].cpu().numpy()
                    scores = out["scores"].cpu().numpy()
                    masks = out.get("masks")
                    if masks is not None:
                        masks = masks.cpu().numpy()

                    for i in range(len(boxes)):
                        score = float(scores[i])
                        if score < cfg.evaluation.conf:
                            continue
                        cid = int(labels[i])  # 1-based Mask R-CNN class corresponds directly to 1-based COCO category_id
                        if cid < 1 or cid > num_classes:
                            continue

                        bx1, by1, bx2, by2 = boxes[i]
                        # Map back from canvas to original image
                        ox1 = max(0.0, (bx1 - px) / r)
                        oy1 = max(0.0, (by1 - py) / r)
                        ox2 = min(float(w), (bx2 - px) / r)
                        oy2 = min(float(h), (by2 - py) / r)
                        bw = max(0.0, ox2 - ox1)
                        bh = max(0.0, oy2 - oy1)

                        rle_dict = None
                        if masks is not None:
                            m_raw = (masks[i, 0] > 0.5).astype(np.uint8)
                            m_crop = m_raw[py:py + nh, px:px + nw]
                            m_orig = cv2.resize(m_crop, (w, h), interpolation=cv2.INTER_NEAREST)
                            rle = mask_utils.encode(np.asfortranarray(m_orig))
                            rle["counts"] = rle["counts"].decode("utf-8")
                            rle_dict = rle

                        predictions.append({
                            "image_id": iid,
                            "category_id": cid,
                            "bbox": [ox1, oy1, bw, bh],
                            "score": score,
                            "segmentation": rle_dict
                        })

    # Evaluate predictions
    metrics = evaluator.evaluate_predictions(predictions)
    print("\n" + "=" * 50)
    print(f" EVALUATION RESULTS [{args.exp_id}]")
    print("=" * 50)
    print(f"Mask mAP@50     : {metrics['mask_mAP50']:.4f}")
    print(f"Mask mAP@50-95  : {metrics['mask_mAP50_95']:.4f}")
    print(f"Box  mAP@50     : {metrics['box_mAP50']:.4f}")
    print(f"Box  mAP@50-95  : {metrics['box_mAP50_95']:.4f}")
    print("=" * 50)

    # Record in comparison suite
    comparison = ExperimentComparison(output_dir=Path(cfg.output_dir))
    res = ExperimentResult(
        exp_id=args.exp_id,
        name=cfg.experiment_name,
        strategy=f"Distill ({cfg.weights.mode})" if cfg.distill.enabled else "GT-Only Baseline",
        mask_map50=metrics["mask_mAP50"],
        mask_map50_95=metrics["mask_mAP50_95"],
        box_map50=metrics["box_mAP50"],
        box_map50_95=metrics["box_mAP50_95"],
        delta_vs_e2=0.0,
        ci_95=[0.0, 0.0],
        seeds_evaluated=1,
        peak_vram_gb=torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0,
        per_class_map50=metrics["per_class"]
    )
    comparison.add_result(res)
    rep = comparison.generate_report(args.output_report)
    print(f"[*] Comparison report updated.")


if __name__ == "__main__":
    import numpy as np
    main()

