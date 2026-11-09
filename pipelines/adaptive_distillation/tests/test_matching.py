"""
Unit tests for Hungarian instance matching and error diagnosis.
"""

import unittest
import numpy as np
from pycocotools import mask as mask_utils

from pipelines.adaptive_distillation.matching.instance_matcher import (
    InstanceMatcher, GroundTruthInstance, MatchResult
)
from pipelines.adaptive_distillation.teachers.teacher_inference import TeacherInstance


def create_dummy_rle(h=100, w=100, x1=10, y1=10, x2=50, y2=50):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    rle = mask_utils.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


class TestInstanceMatcher(unittest.TestCase):

    def setUp(self):
        self.matcher = InstanceMatcher(tau_match=0.50, tau_low=0.10)

    def test_perfect_match_tp(self):
        rle = create_dummy_rle()
        ti = TeacherInstance(
            instance_id=0, raw_class_id=0, raw_class_name="front_bumper",
            canonical_class_id=8, canonical_class_name="front_bumper",
            confidence=0.9, box=[10, 10, 50, 50], mask_rle=rle
        )
        gt = GroundTruthInstance(
            gt_id=1, category_id=8, category_name="front_bumper",
            box=[10, 10, 50, 50], mask_rle=rle, area=1600
        )
        matches, fn_ids = self.matcher.match([ti], [gt])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].status, "TP")
        self.assertAlmostEqual(matches[0].iou, 1.0, places=3)
        self.assertEqual(len(fn_ids), 0)

    def test_class_mismatch_fp_class(self):
        rle = create_dummy_rle()
        ti = TeacherInstance(
            instance_id=0, raw_class_id=0, raw_class_name="front_bumper",
            canonical_class_id=8, canonical_class_name="front_bumper",
            confidence=0.9, box=[10, 10, 50, 50], mask_rle=rle
        )
        gt = GroundTruthInstance(
            gt_id=1, category_id=0, category_name="back_bumper",  # Different class!
            box=[10, 10, 50, 50], mask_rle=rle, area=1600
        )
        matches, fn_ids = self.matcher.match([ti], [gt])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].status, "FP_class")
        self.assertAlmostEqual(matches[0].iou, 1.0, places=3)

    def test_poor_localization_fp_loc(self):
        rle_pred = create_dummy_rle(x1=10, y1=10, x2=50, y2=50)
        rle_gt = create_dummy_rle(x1=25, y1=25, x2=65, y2=65)  # IoU ≈ 0.24, in [tau_low=0.10, tau_match=0.50)
        ti = TeacherInstance(
            instance_id=0, raw_class_id=8, raw_class_name="front_bumper",
            canonical_class_id=8, canonical_class_name="front_bumper",
            confidence=0.8, box=[10, 10, 50, 50], mask_rle=rle_pred
        )
        gt = GroundTruthInstance(
            gt_id=1, category_id=8, category_name="front_bumper",
            box=[25, 25, 65, 65], mask_rle=rle_gt, area=1600
        )
        matches, fn_ids = self.matcher.match([ti], [gt])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].status, "FP_loc")

    def test_unmatched_prediction_fp_unmatched(self):
        rle_pred = create_dummy_rle(x1=10, y1=10, x2=20, y2=20)
        rle_gt = create_dummy_rle(x1=80, y1=80, x2=90, y2=90)  # zero overlap
        ti = TeacherInstance(
            instance_id=0, raw_class_id=8, raw_class_name="front_bumper",
            canonical_class_id=8, canonical_class_name="front_bumper",
            confidence=0.8, box=[10, 10, 20, 20], mask_rle=rle_pred
        )
        gt = GroundTruthInstance(
            gt_id=1, category_id=8, category_name="front_bumper",
            box=[80, 80, 90, 90], mask_rle=rle_gt, area=100
        )
        matches, fn_ids = self.matcher.match([ti], [gt])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].status, "FP_unmatched")
        self.assertIn(1, fn_ids)  # GT instance 1 was missed (FN)

    def test_empty_predictions(self):
        gt = GroundTruthInstance(
            gt_id=1, category_id=8, category_name="front_bumper",
            box=[10, 10, 50, 50], mask_rle=create_dummy_rle(), area=1600
        )
        matches, fn_ids = self.matcher.match([], [gt])
        self.assertEqual(len(matches), 0)
        self.assertEqual(fn_ids, [1])

    def test_empty_ground_truth(self):
        ti = TeacherInstance(
            instance_id=0, raw_class_id=8, raw_class_name="front_bumper",
            canonical_class_id=8, canonical_class_name="front_bumper",
            confidence=0.8, box=[10, 10, 50, 50], mask_rle=create_dummy_rle()
        )
        matches, fn_ids = self.matcher.match([ti], [])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].status, "FP_unmatched")
        self.assertEqual(len(fn_ids), 0)


if __name__ == "__main__":
    unittest.main()
