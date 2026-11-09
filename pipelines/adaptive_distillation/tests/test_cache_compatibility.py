"""
Unit tests for Cache Compatibility Manager, Intelligent Decision Engine,
and Incremental Cache Updating.
"""

import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from pipelines.adaptive_distillation.config import (
    DistillationPipelineConfig, TeacherConfig, DatasetConfig, CacheConfig
)
from pipelines.adaptive_distillation.cache.cache_compatibility_manager import (
    CacheCompatibilityManager, CacheStatus, CacheAction, compute_annotation_hash
)
from pipelines.adaptive_distillation.cache.cache_builder import CacheBuilder
from pipelines.adaptive_distillation.cache.cache_reader import CacheReader
from pipelines.adaptive_distillation.cache.cache_manifest import CacheManifestManager
from pipelines.adaptive_distillation.teachers.teacher_inference import ImageTeacherOutput, TeacherInstance
from pipelines.adaptive_distillation.matching.instance_matcher import MatchResult
from pipelines.adaptive_distillation.matching.trust_weighting import InstanceWeightResult


class TestCacheCompatibility(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_cache_compat_"))
        self.cache_dir = self.test_dir / "cache"
        self.cache_dir.mkdir(parents=True)
        self.data_dir = self.test_dir / "data"
        self.data_dir.mkdir(parents=True)
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(parents=True)

        # Create dummy image files
        self.image_ids = ["101", "102", "103"]
        self.images_coco = []
        for iid in self.image_ids:
            im_path = self.images_dir / f"{iid}.jpg"
            im_path.write_bytes(b"dummy_image_bytes_" + iid.encode())
            self.images_coco.append({
                "id": int(iid),
                "file_name": f"{iid}.jpg",
                "width": 640,
                "height": 480
            })

        # Create dummy annotations
        self.anns_coco = [
            {"id": 1, "image_id": 101, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
            {"id": 2, "image_id": 102, "category_id": 2, "bbox": [20, 20, 60, 60], "area": 3600, "iscrowd": 0},
            {"id": 3, "image_id": 103, "category_id": 3, "bbox": [30, 30, 70, 70], "area": 4900, "iscrowd": 0}
        ]

        self.train_json_path = self.data_dir / "train.json"
        with open(self.train_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "images": self.images_coco,
                "annotations": self.anns_coco,
                "categories": [
                    {"id": 1, "name": "wheel"},
                    {"id": 2, "name": "door"},
                    {"id": 3, "name": "hood"}
                ]
            }, f)

        # Create dummy teacher checkpoint files
        self.teacher_a_ckpt = self.test_dir / "teacher_a_model.pt"
        self.teacher_a_ckpt.write_bytes(b"teacher_a_weights_v1")

        self.teacher_b_ckpt = self.test_dir / "teacher_b_model.pt"
        self.teacher_b_ckpt.write_bytes(b"teacher_b_weights_v1")

        # Config
        self.cfg = DistillationPipelineConfig(
            experiment_name="test_experiment",
            output_dir=str(self.test_dir / "output"),
            teachers=[
                TeacherConfig(name="teacher_a", arch="mask2former", checkpoint_path=str(self.teacher_a_ckpt)),
                TeacherConfig(name="teacher_b", arch="mask2former", checkpoint_path=str(self.teacher_b_ckpt))
            ],
            dataset=DatasetConfig(
                name="test_dataset",
                root=str(self.data_dir),
                train_images=str(self.images_dir),
                val_images=str(self.images_dir),
                train_json=str(self.train_json_path),
                val_json=str(self.train_json_path),
                canonical_taxonomy="32_class",
                imgsz=640
            ),
            cache=CacheConfig(
                dir=str(self.cache_dir),
                shard_size=2
            )
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _build_valid_cache(self, teacher_name: str, image_ids: list):
        builder = CacheBuilder(self.cache_dir, teacher_name, shard_size=2)
        records = []
        for iid in image_ids:
            inst = TeacherInstance(
                instance_id=0,
                raw_class_id=1,
                raw_class_name="wheel",
                canonical_class_id=1,
                canonical_class_name="wheel",
                confidence=0.92,
                box=[10.0, 10.0, 60.0, 60.0],
                mask_rle={"size": [480, 640], "counts": "dummy_rle"},
                mask_prob_downscaled=np.zeros((16, 16), dtype=np.uint8)
            )
            out = ImageTeacherOutput(
                image_id=iid,
                file_name=f"{iid}.jpg",
                width=640,
                height=480,
                teacher_name=teacher_name,
                instances=[inst],
                is_empty=False
            )
            # Add matching and weighting
            m = MatchResult(
                teacher_instance_id=0,
                matched_gt_id=0,
                status="TP",
                iou=0.88,
                predicted_canonical_class=1,
                gt_canonical_class=1,
                quality_score=0.88
            )
            w = InstanceWeightResult(
                teacher_name=teacher_name,
                class_id=1,
                gate_g=1.0,
                share_s=0.5,
                instance_omega=0.88,
                final_weight_w=0.44,
                is_vetoed=False
            )
            records.append((out, [m], [w]))
        builder.write_records(records)

    def test_cache_missing_entirely_full_generation(self):
        """When cache directory is empty, status must be MISSING_CACHE and action REGENERATE_ALL."""
        mgr = CacheCompatibilityManager(self.cfg)
        rep = mgr.check_compatibility()

        self.assertFalse(rep.is_fully_compatible)
        for t_name in ["teacher_a", "teacher_b"]:
            self.assertIn(t_name, rep.teachers)
            s = rep.teachers[t_name]
            self.assertEqual(s.status, CacheStatus.MISSING_CACHE)
            self.assertEqual(s.action, CacheAction.REGENERATE_ALL)
            self.assertEqual(s.missing_count, 3)

    def test_all_cache_records_valid_reuse(self):
        """When all images are cached and manifest matches, status is VALID_CACHE and action is REUSE."""
        for t in ["teacher_a", "teacher_b"]:
            self._build_valid_cache(t, self.image_ids)

        # Generate valid manifest
        manifest = CacheManifestManager.generate_manifest(self.cfg, n_images=len(self.image_ids))
        CacheManifestManager.save_manifest(manifest, self.cache_dir)

        mgr = CacheCompatibilityManager(self.cfg)
        rep = mgr.check_compatibility()

        self.assertTrue(rep.is_fully_compatible)
        for t_name in ["teacher_a", "teacher_b"]:
            s = rep.teachers[t_name]
            self.assertEqual(s.status, CacheStatus.VALID_CACHE)
            self.assertEqual(s.action, CacheAction.REUSE)
            self.assertEqual(s.valid_count, 3)
            self.assertEqual(s.missing_count, 0)
            self.assertEqual(s.estimated_inference_time_sec, 0.0)

    def test_new_images_added_generate_new(self):
        """When dataset has a new image not present in index, action is GENERATE_NEW for that image."""
        for t in ["teacher_a", "teacher_b"]:
            self._build_valid_cache(t, ["101", "102"])  # Only 2 out of 3 cached

        manifest = CacheManifestManager.generate_manifest(self.cfg, n_images=2)
        CacheManifestManager.save_manifest(manifest, self.cache_dir)

        mgr = CacheCompatibilityManager(self.cfg)
        rep = mgr.check_compatibility()

        self.assertFalse(rep.is_fully_compatible)
        for t_name in ["teacher_a", "teacher_b"]:
            s = rep.teachers[t_name]
            self.assertEqual(s.status, CacheStatus.MISSING_CACHE)
            self.assertEqual(s.action, CacheAction.GENERATE_NEW)
            self.assertEqual(s.valid_count, 2)
            self.assertEqual(s.missing_count, 1)
            self.assertEqual(s.affected_inference_image_ids, ["103"])

    def test_one_teacher_changed_regenerate_only_that_teacher(self):
        """When Teacher A checkpoint changes, only Teacher A is invalidated; Teacher B is reused."""
        for t in ["teacher_a", "teacher_b"]:
            self._build_valid_cache(t, self.image_ids)

        manifest = CacheManifestManager.generate_manifest(self.cfg, n_images=3)
        CacheManifestManager.save_manifest(manifest, self.cache_dir)

        # Modify Teacher A checkpoint on disk
        self.teacher_a_ckpt.write_bytes(b"teacher_a_weights_v2_MODIFIED")

        mgr = CacheCompatibilityManager(self.cfg)
        rep = mgr.check_compatibility()

        self.assertFalse(rep.is_fully_compatible)
        # Teacher A must be TEACHER_CHANGED
        s_a = rep.teachers["teacher_a"]
        self.assertEqual(s_a.status, CacheStatus.TEACHER_CHANGED)
        self.assertEqual(s_a.action, CacheAction.REGENERATE_TEACHER)
        self.assertEqual(len(s_a.affected_inference_image_ids), 3)

        # Teacher B must remain VALID_CACHE and REUSE
        s_b = rep.teachers["teacher_b"]
        self.assertEqual(s_b.status, CacheStatus.VALID_CACHE)
        self.assertEqual(s_b.action, CacheAction.REUSE)
        self.assertEqual(s_b.valid_count, 3)

    def test_preprocessing_changed_invalidates_cache(self):
        """When image size changes, PREPROCESSING_CHANGED must be triggered."""
        for t in ["teacher_a", "teacher_b"]:
            self._build_valid_cache(t, self.image_ids)

        manifest = CacheManifestManager.generate_manifest(self.cfg, n_images=3)
        CacheManifestManager.save_manifest(manifest, self.cache_dir)

        # Change config imgsz
        self.cfg.dataset.imgsz = 800

        mgr = CacheCompatibilityManager(self.cfg)
        rep = mgr.check_compatibility()

        self.assertFalse(rep.is_fully_compatible)
        for t in ["teacher_a", "teacher_b"]:
            s = rep.teachers[t]
            self.assertEqual(s.status, CacheStatus.PREPROCESSING_CHANGED)
            self.assertEqual(s.action, CacheAction.REGENERATE_TEACHER)

    def test_cache_corrupted_rebuild(self):
        """When a shard file is deleted/corrupted, CORRUPTED_CACHE triggers REBUILD_CORRUPTED."""
        self._build_valid_cache("teacher_a", self.image_ids)
        manifest = CacheManifestManager.generate_manifest(self.cfg, n_images=3)
        CacheManifestManager.save_manifest(manifest, self.cache_dir)

        # Delete shard_00000.npz
        shard_0 = self.cache_dir / "teacher_a" / "shard_00000.npz"
        shard_0.unlink()

        mgr = CacheCompatibilityManager(self.cfg)
        rep = mgr.check_compatibility()

        s_a = rep.teachers["teacher_a"]
        self.assertEqual(s_a.status, CacheStatus.CORRUPTED_CACHE)
        self.assertEqual(s_a.action, CacheAction.REBUILD_CORRUPTED)
        self.assertGreater(s_a.corrupted_count, 0)

    def test_incremental_cache_append_and_read(self):
        """Tests that appending new records to an existing cache correctly indexes and reads all records."""
        builder = CacheBuilder(self.cache_dir, "teacher_a", shard_size=2)
        
        # 1. Write initial 2 images
        records_batch1 = []
        for iid in ["101", "102"]:
            inst = TeacherInstance(
                instance_id=0,
                raw_class_id=1,
                raw_class_name="wheel",
                canonical_class_id=1,
                canonical_class_name="wheel",
                confidence=0.9,
                box=[10, 10, 50, 50],
                mask_rle={"size": [480, 640], "counts": "rle"},
                mask_prob_downscaled=None
            )
            out = ImageTeacherOutput(
                image_id=iid,
                file_name=f"{iid}.jpg",
                width=640,
                height=480,
                teacher_name="teacher_a",
                instances=[inst],
                is_empty=False
            )
            records_batch1.append((out, [], []))
        builder.write_records(records_batch1)

        # Verify initial read
        reader1 = CacheReader(self.cache_dir, "teacher_a")
        self.assertEqual(len(reader1.index_map), 2)
        self.assertEqual(reader1.read_image("101").image_id, "101")
        self.assertEqual(reader1.read_image("102").image_id, "102")

        # 2. Incrementally append image 103
        records_batch2 = []
        inst = TeacherInstance(
            instance_id=0,
            raw_class_id=2,
            raw_class_name="door",
            canonical_class_id=2,
            canonical_class_name="door",
            confidence=0.85,
            box=[20, 20, 70, 70],
            mask_rle={"size": [480, 640], "counts": "rle2"},
            mask_prob_downscaled=None
        )
        out = ImageTeacherOutput(
            image_id="103",
            file_name="103.jpg",
            width=640,
            height=480,
            teacher_name="teacher_a",
            instances=[inst],
            is_empty=False
        )
        records_batch2.append((out, [], []))

        builder.append_records(records_batch2)

        # Verify that reader now sees all 3 images seamlessly!
        reader2 = CacheReader(self.cache_dir, "teacher_a")
        self.assertEqual(len(reader2.index_map), 3)
        self.assertEqual(reader2.read_image("101").image_id, "101")
        self.assertEqual(reader2.read_image("102").image_id, "102")
        rec3 = reader2.read_image("103")
        self.assertEqual(rec3.image_id, "103")
        self.assertEqual(len(rec3.instances), 1)
        self.assertEqual(rec3.instances[0].canonical_class_id, 2)


    def test_taxonomy_changed_incompatible(self):
        """When canonical taxonomy changes, TAXONOMY_CHANGED must be triggered."""
        for t in ["teacher_a", "teacher_b"]:
            self._build_valid_cache(t, self.image_ids)

        manifest = CacheManifestManager.generate_manifest(self.cfg, n_images=3)
        CacheManifestManager.save_manifest(manifest, self.cache_dir)

        # Change canonical taxonomy in config
        self.cfg.dataset.canonical_taxonomy = "23_class"

        mgr = CacheCompatibilityManager(self.cfg)
        rep = mgr.check_compatibility()

        self.assertFalse(rep.is_fully_compatible)
        for t in ["teacher_a", "teacher_b"]:
            s = rep.teachers[t]
            self.assertEqual(s.status, CacheStatus.TAXONOMY_CHANGED)
            self.assertEqual(s.action, CacheAction.REGENERATE_TEACHER)

    def test_annotation_changed_triggers_rematch(self):
        """When image file is unchanged but annotations in JSON change, ANNOTATION_CHANGED is flagged."""
        for t in ["teacher_a", "teacher_b"]:
            self._build_valid_cache(t, self.image_ids)

        manifest = CacheManifestManager.generate_manifest(self.cfg, n_images=3)
        CacheManifestManager.save_manifest(manifest, self.cache_dir)

        # Modify annotations in train.json without touching image files
        modified_anns = list(self.anns_coco)
        modified_anns[0]["bbox"] = [15, 15, 45, 45]  # slightly changed bbox
        with open(self.train_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "images": self.images_coco,
                "annotations": modified_anns,
                "categories": [
                    {"id": 1, "name": "wheel"},
                    {"id": 2, "name": "door"},
                    {"id": 3, "name": "hood"}
                ]
            }, f)

        # Set fake stored ann_hash in index for image 101 that differs from new hash
        for t in ["teacher_a", "teacher_b"]:
            idx_p = self.cache_dir / f"index_{t}.json"
            with open(idx_p, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            idx_data["101"]["ann_hash"] = "old_stale_hash_123"
            with open(idx_p, "w", encoding="utf-8") as f:
                json.dump(idx_data, f, indent=2)

        mgr = CacheCompatibilityManager(self.cfg)
        rep = mgr.check_compatibility()

        self.assertFalse(rep.is_fully_compatible)
        for t in ["teacher_a", "teacher_b"]:
            s = rep.teachers[t]
            self.assertEqual(s.status, CacheStatus.ANNOTATION_CHANGED)
            self.assertEqual(s.action, CacheAction.REMATCH_ANNOTATIONS)
            self.assertIn("101", s.affected_rematch_image_ids)


if __name__ == "__main__":
    unittest.main()
