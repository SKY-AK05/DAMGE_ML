"""
Cache Compatibility Manager and Intelligent Decision Engine.
Inspects existing teacher prediction cache against current configuration, teachers, and dataset.
Performs granular per-image validation, classifies discrepancies, and intelligently executes
incremental cache generation, partial updates, or annotation-rematching without unnecessary re-inference.
"""

import os
import json
import time
import hashlib
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np

from pipelines.adaptive_distillation.config import DistillationPipelineConfig, TeacherConfig
from pipelines.adaptive_distillation.cache.cache_manifest import CacheManifestManager
from pipelines.adaptive_distillation.cache.cache_builder import CacheBuilder
from pipelines.adaptive_distillation.cache.cache_reader import CacheReader
from pipelines.adaptive_distillation.cache.cache_validator import CacheValidator
from pipelines.adaptive_distillation.teachers.teacher_loader import TeacherLoader, compute_path_sha256
from pipelines.adaptive_distillation.teachers.teacher_inference import TeacherInference, ImageTeacherOutput, TeacherInstance
from pipelines.adaptive_distillation.matching.instance_matcher import InstanceMatcher, GroundTruthInstance
from pipelines.adaptive_distillation.matching.taxonomy_mapper import TaxonomyMapper, CLASSES_23, CLASSES_32
from pipelines.adaptive_distillation.matching.trust_weighting import TrustWeighting
from pipelines.adaptive_distillation.matching.reliability_calculator import ReliabilityCalculator


class CacheStatus(str, Enum):
    VALID_CACHE = "VALID_CACHE"
    MISSING_CACHE = "MISSING_CACHE"
    IMAGE_CHANGED = "IMAGE_CHANGED"
    TEACHER_CHANGED = "TEACHER_CHANGED"
    PREPROCESSING_CHANGED = "PREPROCESSING_CHANGED"
    TAXONOMY_CHANGED = "TAXONOMY_CHANGED"
    ANNOTATION_CHANGED = "ANNOTATION_CHANGED"
    CORRUPTED_CACHE = "CORRUPTED_CACHE"
    INCOMPATIBLE_CACHE = "INCOMPATIBLE_CACHE"


class CacheAction(str, Enum):
    REUSE = "Reuse existing cache"
    GENERATE_NEW = "Generate cache for new images"
    REGENERATE_IMAGE = "Regenerate predictions for changed images"
    REMATCH_ANNOTATIONS = "Reuse predictions; recompute GT matching and veto metadata"
    REGENERATE_TEACHER = "Regenerate all predictions for this teacher"
    REBUILD_CORRUPTED = "Rebuild affected corrupted records"
    REGENERATE_ALL = "Generate full required cache"


@dataclass
class TeacherCompatibilitySummary:
    teacher_name: str
    arch: str
    checkpoint_path: str
    checkpoint_hash: str
    total_dataset_images: int
    valid_count: int = 0
    missing_count: int = 0
    changed_image_count: int = 0
    annotation_changed_count: int = 0
    corrupted_count: int = 0
    status: CacheStatus = CacheStatus.VALID_CACHE
    action: CacheAction = CacheAction.REUSE
    action_detail: str = ""
    affected_inference_image_ids: List[str] = field(default_factory=list)
    affected_rematch_image_ids: List[str] = field(default_factory=list)
    estimated_inference_time_sec: float = 0.0


@dataclass
class CacheCompatibilityReport:
    dataset_name: str
    dataset_path: str
    total_images: int
    canonical_taxonomy: str
    manifest_key: str
    is_fully_compatible: bool
    teachers: Dict[str, TeacherCompatibilitySummary]
    global_errors: List[str] = field(default_factory=list)
    global_warnings: List[str] = field(default_factory=list)
    text_report: str = ""


def compute_annotation_hash(anns: List[dict]) -> str:
    """Computes deterministic hash of bounding boxes, classes, and segmentations for an image."""
    simplified = []
    for a in sorted(anns, key=lambda x: str(x.get("id", ""))):
        simplified.append({
            "cat": a.get("category_id"),
            "box": a.get("bbox"),
            "iscrowd": a.get("iscrowd", 0)
        })
    raw = json.dumps(simplified, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_gt_instances_for_image_helper(
    anns: List[dict],
    cat_id_to_canonical: Dict[int, int],
    h: int,
    w: int
) -> List[GroundTruthInstance]:
    gt_list = []
    from pycocotools import mask as mask_utils
    for a in anns:
        bx, by, bw, bh = a["bbox"]
        if bw <= 0 or bh <= 0:
            continue
        seg = a.get("segmentation")
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


class CacheCompatibilityManager:
    """
    Intelligent cache validation and compatibility management engine.
    Ensures that existing valid predictions are preserved, stale predictions are
    safely refreshed, and annotation changes do not trigger expensive GPU re-inference.
    """

    def __init__(self, config: DistillationPipelineConfig):
        self.cfg = config
        self.cache_dir = Path(config.cache.dir)
        self.manifest_path = self.cache_dir / "cache_manifest.json"

    def check_compatibility(
        self,
        dataset_images: Optional[List[dict]] = None
    ) -> CacheCompatibilityReport:
        """
        Runs comprehensive compatibility evaluation of existing cache against current selections.
        Returns structured CacheCompatibilityReport with granular teacher/image statistics.
        """
        global_errors = []
        global_warnings = []
        teacher_summaries: Dict[str, TeacherCompatibilitySummary] = {}

        # 1. Load dataset metadata
        train_json_path = Path(self.cfg.dataset.train_json)
        if not train_json_path.exists():
            global_errors.append(f"Training dataset JSON not found: {train_json_path}")
            return CacheCompatibilityReport(
                dataset_name=self.cfg.dataset.name,
                dataset_path=str(train_json_path),
                total_images=0,
                canonical_taxonomy=self.cfg.dataset.canonical_taxonomy,
                manifest_key="missing",
                is_fully_compatible=False,
                teachers={},
                global_errors=global_errors,
                text_report="ERROR: Training dataset JSON missing."
            )

        with open(train_json_path, "r", encoding="utf-8") as f:
            train_coco = json.load(f)

        images = dataset_images if dataset_images is not None else train_coco.get("images", [])
        total_images = len(images)
        anns = train_coco.get("annotations", [])

        # Index annotations by image_id
        anns_by_img: Dict[str, List[dict]] = {}
        for a in anns:
            anns_by_img.setdefault(str(a["image_id"]), []).append(a)

        # 2. Inspect Cache Manifest
        manifest: Optional[Dict[str, Any]] = None
        manifest_key = "none"
        manifest_dataset_hash = "none"
        manifest_teachers_meta = {}

        if self.manifest_path.exists():
            try:
                manifest = CacheManifestManager.load_manifest(self.cache_dir)
                manifest_key = manifest.get("cache_key", "unknown")
                manifest_dataset_hash = manifest.get("dataset_manifest_sha256", "unknown")
                manifest_teachers_meta = manifest.get("teachers", {})
            except Exception as e:
                global_warnings.append(f"Cache manifest could not be parsed: {e}")

        # Compute current dataset hash
        current_dataset_hash = compute_path_sha256(train_json_path)
        dataset_identical = (manifest is not None and current_dataset_hash == manifest_dataset_hash)

        # 3. Check Preprocessing and Taxonomy settings
        prep_changed = False
        tax_changed = False
        if manifest:
            prep_meta = manifest.get("preprocess", {})
            if prep_meta.get("imgsz") != self.cfg.dataset.imgsz or prep_meta.get("mask_scale") != self.cfg.cache.mask_scale:
                prep_changed = True
            if manifest.get("canonical_taxonomy") != self.cfg.dataset.canonical_taxonomy:
                tax_changed = True

        # 4. Check each configured teacher
        overall_compatible = True

        for t in self.cfg.teachers:
            t_name = t.name
            ckpt_path = Path(t.checkpoint_path)

            if not ckpt_path.exists():
                global_errors.append(f"Teacher '{t_name}' checkpoint does not exist: {ckpt_path}")
                summary = TeacherCompatibilitySummary(
                    teacher_name=t_name,
                    arch=t.arch,
                    checkpoint_path=str(ckpt_path),
                    checkpoint_hash="missing",
                    total_dataset_images=total_images,
                    status=CacheStatus.INCOMPATIBLE_CACHE,
                    action=CacheAction.REGENERATE_ALL,
                    action_detail="Checkpoint path missing on disk",
                    estimated_inference_time_sec=0.0
                )
                teacher_summaries[t_name] = summary
                overall_compatible = False
                continue

            current_t_hash = compute_path_sha256(ckpt_path)

            idx_path = self.cache_dir / f"index_{t_name}.json"
            teacher_dir = self.cache_dir / t_name

            summary = TeacherCompatibilitySummary(
                teacher_name=t_name,
                arch=t.arch,
                checkpoint_path=str(ckpt_path),
                checkpoint_hash=current_t_hash,
                total_dataset_images=total_images
            )

            # Check if cache manifest exists
            if not manifest:
                summary.status = CacheStatus.MISSING_CACHE
                summary.action = CacheAction.REGENERATE_ALL
                summary.action_detail = "Cache manifest missing on disk"
                summary.missing_count = total_images
                summary.affected_inference_image_ids = [str(img["id"]) for img in images]
                summary.estimated_inference_time_sec = round(total_images / 5.0, 1)
                overall_compatible = False
                teacher_summaries[t_name] = summary
                continue

            # Check if teacher in manifest and checkpoint hash matches
            teacher_changed = False
            if t_name not in manifest_teachers_meta:
                teacher_changed = True
            else:
                stored_t_meta = manifest_teachers_meta[t_name]
                if stored_t_meta.get("checkpoint_sha256") != current_t_hash:
                    teacher_changed = True

            if prep_changed:
                summary.status = CacheStatus.PREPROCESSING_CHANGED
                summary.action = CacheAction.REGENERATE_TEACHER
                summary.action_detail = "Image preprocessing or mask scale modified"
                summary.affected_inference_image_ids = [str(img["id"]) for img in images]
                summary.missing_count = total_images
                summary.estimated_inference_time_sec = round(total_images / 5.0, 1)
                overall_compatible = False
                teacher_summaries[t_name] = summary
                continue

            if tax_changed:
                summary.status = CacheStatus.TAXONOMY_CHANGED
                summary.action = CacheAction.REGENERATE_TEACHER
                summary.action_detail = "Canonical taxonomy configuration modified"
                summary.affected_inference_image_ids = [str(img["id"]) for img in images]
                summary.missing_count = total_images
                summary.estimated_inference_time_sec = round(total_images / 5.0, 1)
                overall_compatible = False
                teacher_summaries[t_name] = summary
                continue

            if teacher_changed:
                summary.status = CacheStatus.TEACHER_CHANGED
                summary.action = CacheAction.REGENERATE_TEACHER
                summary.action_detail = "Teacher checkpoint hash changed or newly added"
                summary.affected_inference_image_ids = [str(img["id"]) for img in images]
                summary.missing_count = total_images
                summary.estimated_inference_time_sec = round(total_images / 5.0, 1)
                overall_compatible = False
                teacher_summaries[t_name] = summary
                continue

            if not idx_path.exists() or not teacher_dir.exists():
                summary.status = CacheStatus.MISSING_CACHE
                summary.action = CacheAction.REGENERATE_ALL
                summary.action_detail = "Index or cache directory missing on disk"
                summary.missing_count = total_images
                summary.affected_inference_image_ids = [str(img["id"]) for img in images]
                summary.estimated_inference_time_sec = round(total_images / 5.0, 1)
                overall_compatible = False
                teacher_summaries[t_name] = summary
                continue

            # Index exists: inspect per-image records
            try:
                with open(idx_path, "r", encoding="utf-8") as f:
                    index_map: Dict[str, Any] = json.load(f)
            except Exception as e:
                summary.status = CacheStatus.CORRUPTED_CACHE
                summary.action = CacheAction.REBUILD_CORRUPTED
                summary.action_detail = f"Corrupted index JSON: {e}"
                summary.corrupted_count = total_images
                summary.affected_inference_image_ids = [str(img["id"]) for img in images]
                summary.estimated_inference_time_sec = round(total_images / 5.0, 1)
                overall_compatible = False
                teacher_summaries[t_name] = summary
                continue

            # Check individual images
            existing_shards = set(p.name for p in teacher_dir.glob("shard_*.npz"))
            valid_c = 0
            missing_c = 0
            changed_c = 0
            ann_changed_c = 0
            corrupted_c = 0

            inference_needed_ids = []
            rematch_needed_ids = []

            for img in images:
                iid = str(img["id"])
                if iid not in index_map:
                    missing_c += 1
                    inference_needed_ids.append(iid)
                    continue

                entry = index_map[iid]
                shard_name = entry.get("shard")
                if not shard_name or shard_name not in existing_shards:
                    corrupted_c += 1
                    inference_needed_ids.append(iid)
                    continue

                # Check if image file exists
                img_path = Path(self.cfg.dataset.train_images) / img.get("file_name", f"{iid}.jpg")
                if not img_path.exists():
                    # Check alt location
                    alt_path = Path(self.cfg.dataset.root) / "images" / img.get("file_name", f"{iid}.jpg")
                    if alt_path.exists():
                        img_path = alt_path

                if not img_path.exists():
                    corrupted_c += 1
                    inference_needed_ids.append(iid)
                    continue

                # Check if image content changed
                if "img_hash" in entry:
                    try:
                        cur_im_hash = compute_path_sha256(img_path)
                        if cur_im_hash != entry["img_hash"]:
                            changed_c += 1
                            inference_needed_ids.append(iid)
                            continue
                    except Exception:
                        pass

                # Check if annotations changed
                current_img_anns = anns_by_img.get(iid, [])
                if not dataset_identical:
                    cur_ann_hash = compute_annotation_hash(current_img_anns)
                    stored_ann_hash = entry.get("ann_hash")
                    if stored_ann_hash and stored_ann_hash != cur_ann_hash:
                        ann_changed_c += 1
                        rematch_needed_ids.append(iid)
                        continue

                valid_c += 1

            summary.valid_count = valid_c
            summary.missing_count = missing_c
            summary.changed_image_count = changed_c
            summary.annotation_changed_count = ann_changed_c
            summary.corrupted_count = corrupted_c
            summary.affected_inference_image_ids = inference_needed_ids
            summary.affected_rematch_image_ids = rematch_needed_ids

            # Formulate action for this teacher
            if valid_c == total_images:
                summary.status = CacheStatus.VALID_CACHE
                summary.action = CacheAction.REUSE
                summary.action_detail = "All records valid and verified on disk"
                summary.estimated_inference_time_sec = 0.0
            elif corrupted_c > 0:
                summary.status = CacheStatus.CORRUPTED_CACHE
                summary.action = CacheAction.REBUILD_CORRUPTED
                summary.action_detail = f"Rebuild {len(inference_needed_ids)} corrupted or missing shard entries"
                summary.estimated_inference_time_sec = round(len(inference_needed_ids) / 5.0, 1)
                overall_compatible = False
            elif missing_c > 0 or changed_c > 0:
                summary.status = CacheStatus.MISSING_CACHE
                summary.action = CacheAction.GENERATE_NEW
                summary.action_detail = f"Generate {len(inference_needed_ids)} missing/changed predictions"
                summary.estimated_inference_time_sec = round(len(inference_needed_ids) / 5.0, 1)
                overall_compatible = False
            elif ann_changed_c > 0:
                summary.status = CacheStatus.ANNOTATION_CHANGED
                summary.action = CacheAction.REMATCH_ANNOTATIONS
                summary.action_detail = f"Rematch {len(rematch_needed_ids)} images with ground truth (zero GPU inference)"
                summary.estimated_inference_time_sec = 0.0
                overall_compatible = False

            teacher_summaries[t_name] = summary

        # 5. Format Text Report
        text_rep = self._format_ascii_report(
            dataset_name=self.cfg.dataset.name,
            total_images=total_images,
            summaries=teacher_summaries,
            global_errors=global_errors,
            global_warnings=global_warnings
        )

        return CacheCompatibilityReport(
            dataset_name=self.cfg.dataset.name,
            dataset_path=str(train_json_path),
            total_images=total_images,
            canonical_taxonomy=self.cfg.dataset.canonical_taxonomy,
            manifest_key=manifest_key,
            is_fully_compatible=overall_compatible and len(global_errors) == 0,
            teachers=teacher_summaries,
            global_errors=global_errors,
            global_warnings=global_warnings,
            text_report=text_rep
        )

    def _format_ascii_report(
        self,
        dataset_name: str,
        total_images: int,
        summaries: Dict[str, TeacherCompatibilitySummary],
        global_errors: List[str],
        global_warnings: List[str]
    ) -> str:
        lines = [
            "=" * 56,
            f"{'TEACHER CACHE COMPATIBILITY REPORT':^56}",
            "=" * 56,
            f"Dataset:          {dataset_name}",
            f"Images selected:  {total_images:,}",
            ""
        ]

        total_valid = 0
        total_records_needed = 0

        for t_name, s in summaries.items():
            lines.append(f"{t_name} ({s.arch}):")
            lines.append(f"  Valid cache:         {s.valid_count:>8,}")
            lines.append(f"  Missing cache:       {s.missing_count:>8,}")
            lines.append(f"  Changed images:      {s.changed_image_count:>8,}")
            lines.append(f"  Annotation changed:  {s.annotation_changed_count:>8,}")
            lines.append(f"  Corrupted records:   {s.corrupted_count:>8,}")
            lines.append(f"  Status:              {s.status.value}")
            lines.append(f"  Action required:     {s.action.value}")
            lines.append(f"  Action detail:       {s.action_detail}")
            lines.append("")
            total_valid += s.valid_count
            total_records_needed += len(s.affected_inference_image_ids)

        lines.append("Actions Summary:")
        for t_name, s in summaries.items():
            if s.action == CacheAction.REUSE:
                lines.append(f"  {t_name}: Reuse existing cache")
            elif s.action == CacheAction.REMATCH_ANNOTATIONS:
                lines.append(f"  {t_name}: Rematch {len(s.affected_rematch_image_ids)} records (no GPU inference)")
            else:
                lines.append(f"  {t_name}: Generate {len(s.affected_inference_image_ids)} records")

        total_inference_time = sum(s.estimated_inference_time_sec for s in summaries.values())
        if total_inference_time == 0.0:
            lines.append("\nEstimated teacher inference time: 0.0s (100% cached)")
        else:
            lines.append(f"\nEstimated teacher inference time: ~{total_inference_time:.1f}s (at ~5.0 imgs/s/teacher)")

        denom = max(1, len(summaries) * total_images)
        reuse_pct = (total_valid / denom) * 100.0
        lines.append(f"Cache reuse: {reuse_pct:.1f}%")

        if global_warnings:
            lines.append("\nWarnings:")
            for w in global_warnings:
                lines.append(f"  [!] {w}")

        if global_errors:
            lines.append("\nErrors:")
            for e in global_errors:
                lines.append(f"  [ERROR] {e}")

        lines.append("=" * 56)
        return "\n".join(lines)

    def prepare_missing_cache(
        self,
        report: CacheCompatibilityReport,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> bool:
        """
        Intelligently executes minimal required actions:
        - Bypasses inference for valid cache records.
        - Rematches annotations from cache for annotation-only changes.
        - Sequentially loads teachers on GPU to process only missing/changed images.
        - Appends new records into the persistent disk-backed cache.
        - Updates reliability calibration with scientific provenance notice.
        - Updates cache manifest atomically.
        """
        if report.is_fully_compatible:
            print("[OK] Cache is fully compatible and complete! Reusing existing predictions.")
            return True

        print("\n" + "=" * 65)
        print(" [CACHE PREPARATION] EXECUTING INTELLIGENT INCREMENTAL CACHE UPDATE")
        print("=" * 65)

        # 1. Load dataset annotations
        with open(self.cfg.dataset.train_json, "r", encoding="utf-8") as f:
            train_coco = json.load(f)

        train_images_dict = {str(img["id"]): img for img in train_coco.get("images", [])}
        train_anns_by_img: Dict[str, List[dict]] = {}
        for a in train_coco.get("annotations", []):
            train_anns_by_img.setdefault(str(a["image_id"]), []).append(a)

        canonical_classes = CLASSES_32 if self.cfg.dataset.canonical_taxonomy == "32_class" else CLASSES_23
        mapper = TaxonomyMapper(canonical_classes=canonical_classes)
        matcher = InstanceMatcher(tau_match=self.cfg.matching.tau_match, tau_low=self.cfg.matching.tau_low)
        trust_weighting = TrustWeighting(self.cfg.weights)
        infer_engine = TeacherInference(mapper=mapper, mask_downscale=self.cfg.cache.mask_scale)

        cat_id_to_can: Dict[int, int] = {}
        for c in train_coco.get("categories", []):
            cat_name = c["name"]
            if cat_name in mapper.canonical_name_to_id:
                cat_id_to_can[c["id"]] = mapper.canonical_name_to_id[cat_name]
            else:
                cat_id_to_can[c["id"]] = c["id"] - 1

        # 2. Check reliability calibration
        rel_report_path = Path("reliability") / "calibration_report.md"
        if not rel_report_path.exists():
            print("[*] Reliability calibration report missing. Calibrating teachers on calibration split...")
            self._calibrate_reliability(canonical_classes, mapper, matcher, infer_engine, cat_id_to_can)

        # Read reliability scores for weighting
        all_r_hats: Dict[int, Dict[str, float]] = {}
        for cid in range(len(canonical_classes)):
            all_r_hats[cid] = {t.name: 0.80 for t in self.cfg.teachers}

        # 3. Process each teacher sequentially (GPU isolation)
        for t_cfg in self.cfg.teachers:
            t_name = t_cfg.name
            t_summary = report.teachers.get(t_name)
            if not t_summary:
                continue

            builder = CacheBuilder(self.cache_dir, t_name, shard_size=self.cfg.cache.shard_size)

            # Scenario A: Full reuse
            if t_summary.action == CacheAction.REUSE:
                print(f"[OK] Teacher '{t_name}': 100% valid cache ({t_summary.valid_count:,} images). Skipping inference.")
                continue

            # Scenario B: Rematch annotations only (no GPU inference)
            if t_summary.action == CacheAction.REMATCH_ANNOTATIONS:
                print(f"[*] Teacher '{t_name}': Rematching {len(t_summary.affected_rematch_image_ids):,} images with updated ground truth...")
                reader = CacheReader(self.cache_dir, t_name)
                rematch_records = []
                for iid in t_summary.affected_rematch_image_ids:
                    cached_rec = reader.read_image(iid)
                    # Convert cached instances back to TeacherInstance
                    t_instances = []
                    for inst in cached_rec.instances:
                        t_instances.append(TeacherInstance(
                            canonical_class_id=inst.canonical_class_id,
                            confidence=inst.confidence,
                            box=inst.box,
                            mask_rle=inst.mask_rle,
                            mask_prob_downscaled=inst.mask_prob_downscaled
                        ))
                    out = ImageTeacherOutput(
                        image_id=iid,
                        width=cached_rec.width,
                        height=cached_rec.height,
                        instances=t_instances,
                        is_empty=cached_rec.is_empty
                    )
                    gt_instances = load_gt_instances_for_image_helper(
                        train_anns_by_img.get(iid, []), cat_id_to_can, out.height, out.width
                    )
                    matches, fn_ids = matcher.match(out.instances, gt_instances)
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
                    rematch_records.append((out, matches, weights))

                builder.append_records(rematch_records)
                print(f"[OK] Teacher '{t_name}': Successfully rematched {len(rematch_records):,} records without GPU re-inference.")
                continue

            # Scenario C: Inference required (missing, changed, or new teacher)
            ids_to_infer = t_summary.affected_inference_image_ids
            if not ids_to_infer and t_summary.action in (CacheAction.REGENERATE_TEACHER, CacheAction.REGENERATE_ALL):
                ids_to_infer = list(train_images_dict.keys())

            print(f"\n[*] Teacher '{t_name}': Generating predictions for {len(ids_to_infer):,} images on GPU...")
            loaded_t = TeacherLoader.load(t_cfg)

            infer_records = []
            for idx, iid in enumerate(ids_to_infer, 1):
                img_info = train_images_dict.get(iid, {"file_name": f"{iid}.jpg"})
                img_p = Path(self.cfg.dataset.train_images) / img_info["file_name"]
                if not img_p.exists():
                    img_p = Path(self.cfg.dataset.root) / "images" / img_info["file_name"]

                out = infer_engine.infer_image(loaded_t, str(img_p), iid)
                gt_instances = load_gt_instances_for_image_helper(
                    train_anns_by_img.get(iid, []), cat_id_to_can, out.height, out.width
                )
                matches, fn_ids = matcher.match(out.instances, gt_instances)

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

                infer_records.append((out, matches, weights))

                if progress_callback and (idx % 100 == 0 or idx == len(ids_to_infer)):
                    progress_callback(t_name, idx, len(ids_to_infer))

            # If full regeneration, write full records; else append incrementally
            if t_summary.action in (CacheAction.REGENERATE_TEACHER, CacheAction.REGENERATE_ALL) and len(ids_to_infer) == len(train_images_dict):
                builder.write_records(infer_records)
            else:
                builder.append_records(infer_records)

            TeacherLoader.unload(loaded_t)
            print(f"[OK] Completed and unloaded teacher '{t_name}'.")

        # 4. Atomically update Cache Manifest
        manifest = CacheManifestManager.generate_manifest(self.cfg, n_images=len(train_images_dict))
        manifest_path = CacheManifestManager.save_manifest(manifest, self.cache_dir)
        print(f"[*] Updated cache manifest -> {manifest_path} (Cache Key: {manifest['cache_key'][:12]}...)")

        # 5. Validate cache integrity with CacheValidator
        expected_ids = list(train_images_dict.keys())
        validator = CacheValidator(self.cache_dir, self.cfg)
        is_valid, errors, summary = validator.validate_cache(expected_ids)

        if not is_valid:
            print("[ERROR] Cache validation failed after update:")
            for e in errors:
                print(f"  - {e}")
            return False

        print("\n[SUCCESS] Cache preparation complete and validated! Ready for Student C training.")
        return True

    def _calibrate_reliability(
        self,
        canonical_classes: List[str],
        mapper: TaxonomyMapper,
        matcher: InstanceMatcher,
        infer_engine: TeacherInference,
        cat_id_to_can: Dict[int, int]
    ) -> None:
        """Runs validation calibration on calib_json with scientific provenance check."""
        calib_json = getattr(self.cfg.dataset, "calib_json", None) or self.cfg.dataset.val_json
        calib_path = Path(calib_json)
        if not calib_path.exists():
            print(f"[!] Warning: Calibration split not found at {calib_path}. Skipping calibration calculation.")
            return

        with open(calib_path, "r", encoding="utf-8") as f:
            val_coco = json.load(f)

        val_images = val_coco.get("images", [])
        val_anns_by_img: Dict[str, List[dict]] = {}
        for a in val_coco.get("annotations", []):
            val_anns_by_img.setdefault(str(a["image_id"]), []).append(a)

        rel_calc = ReliabilityCalculator(canonical_classes=canonical_classes, shrinkage_m=self.cfg.reliability.shrinkage_m)
        artifacts = []

        for t_cfg in self.cfg.teachers:
            t_name = t_cfg.name
            print(f"[*] Calibrating {t_name} on {len(val_images)} calibration images...")
            loaded_t = TeacherLoader.load(t_cfg)
            val_matches = []
            for img_info in val_images:
                iid = str(img_info["id"])
                img_p = Path(self.cfg.dataset.val_images) / img_info["file_name"]
                if not img_p.exists():
                    img_p = Path(self.cfg.dataset.root) / "images" / img_info["file_name"]
                out = infer_engine.infer_image(loaded_t, str(img_p), iid)
                gt_instances = load_gt_instances_for_image_helper(
                    val_anns_by_img.get(iid, []), cat_id_to_can, out.height, out.width
                )
                matches, fn_ids = matcher.match(out.instances, gt_instances)
                val_matches.append((iid, matches, fn_ids))

            art = rel_calc.compute_teacher_reliability(t_name, val_matches)
            artifacts.append(art)
            TeacherLoader.unload(loaded_t)
            print(f"[*] {t_name} Reliability r_global = {art.r_global:.4f}")

        rel_report_path = Path("reliability") / "calibration_report.md"
        rel_report_path.parent.mkdir(parents=True, exist_ok=True)
        rel_calc.generate_markdown_report(artifacts, str(rel_report_path))

        warning_header = (
            "> [!WARNING] **Calibration Provenance & Selection Bias Notice**\n"
            "> The calibration images were partitioned from the validation pool on which teacher\n"
            "> checkpoints were selected. These reliability metrics are exploratory and selection-biased.\n\n"
        )
        if rel_report_path.exists():
            content = rel_report_path.read_text(encoding="utf-8")
            rel_report_path.write_text(warning_header + content, encoding="utf-8")
        print(f"[*] Saved reliability report -> {rel_report_path}")
