"""
Unit tests for sharded cache building, streaming retrieval, and manifest validation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils

from pipelines.adaptive_distillation.config import DistillationPipelineConfig
from pipelines.adaptive_distillation.teachers.teacher_inference import ImageTeacherOutput, TeacherInstance
from pipelines.adaptive_distillation.cache.cache_builder import CacheBuilder
from pipelines.adaptive_distillation.cache.cache_reader import CacheReader
from pipelines.adaptive_distillation.cache.cache_manifest import CacheManifestManager


def dummy_rle(h=100, w=100):
    m = np.zeros((h, w), dtype=np.uint8)
    m[10:40, 20:60] = 1
    rle = mask_utils.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


class TestCacheSystem(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_cache_round_trip(self):
        builder = CacheBuilder(self.temp_dir, teacher_name="Teacher_A", shard_size=2)
        rle = dummy_rle()

        inst1 = TeacherInstance(
            instance_id=0, raw_class_id=0, raw_class_name="front_bumper",
            canonical_class_id=8, canonical_class_name="front_bumper",
            confidence=0.88, box=[20, 10, 60, 40], mask_rle=rle,
            mask_prob_downscaled=np.ones((25, 25), dtype=np.uint8) * 200
        )
        out1 = ImageTeacherOutput(
            image_id="img_101", file_name="car_101.jpg", width=100, height=100,
            teacher_name="Teacher_A", instances=[inst1], is_empty=False
        )

        out_empty = ImageTeacherOutput(
            image_id="img_102", file_name="car_102.jpg", width=100, height=100,
            teacher_name="Teacher_A", instances=[], is_empty=True
        )

        builder.write_records([(out1, None, None), (out_empty, None, None)])

        reader = CacheReader(self.temp_dir, teacher_name="Teacher_A")

        # 1. Read non-empty record
        rec1 = reader.read_image("img_101")
        self.assertEqual(rec1.image_id, "img_101")
        self.assertFalse(rec1.is_empty)
        self.assertEqual(len(rec1.instances), 1)
        self.assertEqual(rec1.instances[0].canonical_class_id, 8)
        self.assertAlmostEqual(rec1.instances[0].confidence, 0.88, places=3)
        self.assertEqual(rec1.instances[0].box, [20, 10, 60, 40])
        self.assertIsNotNone(rec1.instances[0].mask_prob_downscaled)

        # 2. Read explicit empty record
        rec2 = reader.read_image("img_102")
        self.assertEqual(rec2.image_id, "img_102")
        self.assertTrue(rec2.is_empty)
        self.assertEqual(len(rec2.instances), 0)

        # 3. Read missing image raises KeyError
        with self.assertRaises(KeyError):
            reader.read_image("missing_img_999")

    def test_cache_manifest_generation(self):
        cfg = DistillationPipelineConfig()
        cfg.cache.dir = self.temp_dir
        manifest = CacheManifestManager.generate_manifest(cfg, n_images=5)
        path = CacheManifestManager.save_manifest(manifest, self.temp_dir)
        self.assertTrue(path.exists())

        loaded = CacheManifestManager.load_manifest(self.temp_dir)
        self.assertEqual(loaded["schema_version"], "2.0")
        self.assertEqual(loaded["n_images"], 5)


if __name__ == "__main__":
    unittest.main()

