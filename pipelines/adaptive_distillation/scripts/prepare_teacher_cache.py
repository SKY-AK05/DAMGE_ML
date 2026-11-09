#!/usr/bin/env python3
"""
Stage A: Teacher Preparation, Reliability Calibration, and Persistent Cache Generation.
Executes sequential teacher inference, Hungarian GT matching, validation calibration,
and sharded cache creation.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.adaptive_distillation.config import DistillationPipelineConfig
from pipelines.adaptive_distillation.teachers.teacher_scanner import TeacherScanner
from pipelines.adaptive_distillation.teachers.teacher_loader import TeacherLoader
from pipelines.adaptive_distillation.teachers.teacher_inference import TeacherInference
from pipelines.adaptive_distillation.matching.instance_matcher import InstanceMatcher, GroundTruthInstance
from pipelines.adaptive_distillation.matching.taxonomy_mapper import TaxonomyMapper, CLASSES_23, CLASSES_32
from pipelines.adaptive_distillation.matching.reliability_calculator import ReliabilityCalculator
from pipelines.adaptive_distillation.matching.trust_weighting import TrustWeighting
from pipelines.adaptive_distillation.cache.cache_builder import CacheBuilder
from pipelines.adaptive_distillation.cache.cache_manifest import CacheManifestManager
from pipelines.adaptive_distillation.cache.cache_validator import CacheValidator


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Teacher Prediction Cache and Calibrate Reliability")
    parser.add_argument("--config", default="pipelines/adaptive_distillation/configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--select_teachers", action="store_true", help="Interactive prompt to choose teachers from runs_comparison")
    parser.add_argument("--teachers", nargs="+", help="Specific teachers by index or run_id")
    parser.add_argument("--max_images", type=int, default=None, help="Optional image limit for rapid debugging")
    return parser.parse_args()


def load_gt_instances_for_image(
    anns: List[dict],
    cat_id_to_canonical: Dict[int, int],
    h: int,
    w: int
) -> List[GroundTruthInstance]:
    gt_list = []
    for a in anns:
        bx, by, bw, bh = a["bbox"]
        if bw <= 0 or bh <= 0:
            continue
        seg = a.get("segmentation")
        from pycocotools import mask as mask_utils
        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(seg, dict):
            rle = seg
        else:
            continue
        rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
        can_id = cat_id_to_canonical.get(a["category_id"], a["category_id"] - 1)
        gt_list.append(GroundTruthInstance(
            gt_id=a["id"],
            category_id=can_id,
            category_name="",
            box=[bx, by, bx + bw, by + bh],
            mask_rle=rle,
            area=float(a.get("area", bw * bh)),
            iscrowd=bool(a.get("iscrowd", 0))
        ))
    return gt_list


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = DistillationPipelineConfig.from_yaml(cfg_path)

    # 1. Teacher selection from runs_comparison if requested
    scanner = TeacherScanner()
    if args.select_teachers:
        chosen_teachers = scanner.interactive_select()
        cfg.teachers = chosen_teachers
        cfg.save_yaml(cfg_path)
    elif args.teachers:
        chosen_teachers = scanner.select_by_spec(args.teachers)
        cfg.teachers = chosen_teachers
        cfg.save_yaml(cfg_path)

    if not cfg.teachers:
        print("[!] No teachers configured. Launching scanner selection...")
        cfg.teachers = scanner.interactive_select()
        cfg.save_yaml(cfg_path)

    print(f"\n[*] Starting Phase A: Teacher Cache Preparation for {len(cfg.teachers)} teachers:")
    for t in cfg.teachers:
        print(f"    - {t.name}: {t.arch} ({t.checkpoint_path})")

    # Determine calibration and training manifests
    calib_json_path = getattr(cfg.dataset, "calib_json", None) or cfg.dataset.val_json
    print(f"[*] Loading calibration split from: {calib_json_path}")
    with open(calib_json_path, "r") as f:
        val_coco = json.load(f)

    print(f"[*] Loading training split from: {cfg.dataset.train_json}")
    with open(cfg.dataset.train_json, "r") as f:
        train_coco = json.load(f)

    train_images = train_coco.get("images", [])
    val_images = val_coco.get("images", [])
    if args.max_images:
        train_images = train_images[:args.max_images]
        val_images = val_images[:args.max_images]

    canonical_classes = CLASSES_32 if cfg.dataset.canonical_taxonomy == "32_class" else CLASSES_23
    print(f"[*] Canonical taxonomy: {cfg.dataset.canonical_taxonomy} ({len(canonical_classes)} classes)")

    mapper = TaxonomyMapper(canonical_classes=canonical_classes)
    matcher = InstanceMatcher(tau_match=cfg.matching.tau_match, tau_low=cfg.matching.tau_low)
    infer_engine = TeacherInference(mapper=mapper, mask_downscale=cfg.cache.mask_scale)
    rel_calc = ReliabilityCalculator(canonical_classes=canonical_classes, shrinkage_m=cfg.reliability.shrinkage_m)
    trust_weighting = TrustWeighting(cfg.weights)

    # Pre-index annotations by image ID for O(1) retrieval
    val_anns_by_img: Dict[str, List[dict]] = {}
    for a in val_coco.get("annotations", []):
        val_anns_by_img.setdefault(str(a["image_id"]), []).append(a)

    train_anns_by_img: Dict[str, List[dict]] = {}
    for a in train_coco.get("annotations", []):
        train_anns_by_img.setdefault(str(a["image_id"]), []).append(a)

    cat_id_to_can: Dict[int, int] = {}
    for c in train_coco.get("categories", []):
        cat_name = c["name"]
        if cat_name in mapper.canonical_name_to_id:
            cat_id_to_can[c["id"]] = mapper.canonical_name_to_id[cat_name]
        else:
            cat_id_to_can[c["id"]] = c["id"] - 1

    cache_dir = Path(cfg.cache.dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # PHASE A1: RELIABILITY CALIBRATION
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" [PHASE A1] RELIABILITY CALIBRATION (EXPLORATORY CALIBRATION SPLIT)")
    print("=" * 65)
    print(" [!] NOTE: Calibration split (652 images) was partitioned from the validation pool")
    print("     used for original teacher checkpoint selection. Metrics are exploratory.")

    reliability_artifacts = []
    teacher_val_matches: Dict[str, list] = {}

    for teacher_cfg in cfg.teachers:
        t_name = teacher_cfg.name
        print(f"\n[*] Calibrating {t_name} on {len(val_images)} images...")
        loaded_t = TeacherLoader.load(teacher_cfg)

        val_matches = []
        for img_info in tqdm(val_images, desc=f"Calib {t_name}"):
            iid = str(img_info["id"])
            img_p = Path(cfg.dataset.val_images) / img_info["file_name"]
            out = infer_engine.infer_image(loaded_t, str(img_p), iid)
            gt_instances = load_gt_instances_for_image(
                val_anns_by_img.get(iid, []), cat_id_to_can, out.height, out.width
            )
            matches, fn_ids = matcher.match(out.instances, gt_instances)
            val_matches.append((iid, matches, fn_ids))

        teacher_val_matches[t_name] = val_matches
        rel_art = rel_calc.compute_teacher_reliability(t_name, val_matches)
        reliability_artifacts.append(rel_art)
        print(f"[*] {t_name} Exploratory Reliability r_global = {rel_art.r_global:.4f}")
        TeacherLoader.unload(loaded_t)
        print(f"[OK] Released memory for {t_name}.")

    # Precompute relative teacher reliabilities per class for multi-teacher shares
    all_r_hats: Dict[int, Dict[str, float]] = {}
    for cid in range(len(canonical_classes)):
        all_r_hats[cid] = {}
        for art in reliability_artifacts:
            cr = art.class_reliabilities.get(cid)
            all_r_hats[cid][art.teacher_name] = cr.r_shrunk if cr is not None else art.r_global

    # -------------------------------------------------------------
    # PHASE A2: TRAINING CACHE GENERATION
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" [PHASE A2] TRAINING PREDICTION CACHE GENERATION (5,212 IMAGES)")
    print("=" * 65)

    for teacher_cfg in cfg.teachers:
        t_name = teacher_cfg.name
        idx_file = cache_dir / f"index_{t_name}.json"
        t_dir = cache_dir / t_name
        if idx_file.exists() and t_dir.exists():
            try:
                with open(idx_file, "r") as f:
                    existing_idx = json.load(f)
                if len(existing_idx) == len(train_images):
                    print(f"\n[OK] Found complete existing training cache for {t_name} ({len(existing_idx)} images). Skipping re-inference.")
                    continue
            except Exception:
                pass

        print(f"\n[*] Generating training cache for {t_name} ({len(train_images)} images)...")
        loaded_t = TeacherLoader.load(teacher_cfg)

        builder = CacheBuilder(cache_dir, t_name, shard_size=cfg.cache.shard_size)
        train_records = []

        for img_info in tqdm(train_images, desc=f"Train {t_name}"):
            iid = str(img_info["id"])
            img_p = Path(cfg.dataset.train_images) / img_info["file_name"]
            out = infer_engine.infer_image(loaded_t, str(img_p), iid)
            gt_instances = load_gt_instances_for_image(
                train_anns_by_img.get(iid, []), cat_id_to_can, out.height, out.width
            )
            matches, fn_ids = matcher.match(out.instances, gt_instances)

            # Assign weights
            weights = []
            for m in matches:
                cid = m.predicted_canonical_class if m.predicted_canonical_class is not None else 0
                w_res = trust_weighting.compute_full_weight(
                    teacher_name=t_name,
                    class_id=cid,
                    status=m.status,
                    iou=m.iou,
                    r_hat_teachers=all_r_hats.get(cid, {t_name: 0.80}),
                    r_ref=0.60
                )
                weights.append(w_res)

            train_records.append((out, matches, weights))

        print(f"[*] Writing sharded cache for {t_name} ({len(train_records)} images)...")
        builder.write_records(train_records)
        TeacherLoader.unload(loaded_t)
        print(f"[OK] Completed and unloaded {t_name}.")

    # Generate calibration report with selection bias disclaimer
    rel_report_path = Path("reliability") / "calibration_report.md"
    rel_report_path.parent.mkdir(parents=True, exist_ok=True)
    rel_calc.generate_markdown_report(reliability_artifacts, str(rel_report_path))
    
    warning_header = (
        "> [!WARNING] **Calibration Provenance & Selection Bias Notice**\n"
        "> The 652 calibration images (`calib_652.json`) were partitioned from the original 1,304 validation\n"
        "> pool on which Teacher A and Teacher B checkpoints were selected. These reliability metrics must\n"
        "> be treated as exploratory and potentially selection-biased until independent re-calibration is conducted.\n\n"
    )
    if rel_report_path.exists():
        content = rel_report_path.read_text(encoding="utf-8")
        rel_report_path.write_text(warning_header + content, encoding="utf-8")
    print(f"[*] Saved reliability report -> {rel_report_path}")

    # Generate cache manifest
    manifest = CacheManifestManager.generate_manifest(cfg, n_images=len(train_images))
    manifest_path = CacheManifestManager.save_manifest(manifest, cache_dir)
    print(f"[*] Saved cache manifest -> {manifest_path} (Cache Key: {manifest['cache_key'][:12]}...)")

    # Run cache validation
    expected_ids = [str(img["id"]) for img in train_images]
    validator = CacheValidator(cache_dir, cfg)
    is_valid, errors, summary = validator.validate_cache(expected_ids)

    if not is_valid:
        print("[ERROR] Cache validation failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\n[SUCCESS] Phase A Complete! Teacher cache generated and validated successfully.")


if __name__ == "__main__":
    main()
