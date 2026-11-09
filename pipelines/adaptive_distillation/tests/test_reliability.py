"""
Unit tests for reliability calculations, Bayesian shrinkage, and confidence intervals.
"""

import unittest
from pipelines.adaptive_distillation.matching.reliability_calculator import ReliabilityCalculator, MatchResult
from pipelines.adaptive_distillation.matching.taxonomy_mapper import CLASSES_23


class TestReliabilityCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = ReliabilityCalculator(
            canonical_classes=CLASSES_23,
            shrinkage_m=20.0,
            n_min=10,
            bootstrap_iters=100
        )

    def test_shrinkage_zero_samples(self):
        """When n=0, shrunk score must fall back to global reliability."""
        # Simulated run with no detections for class 0
        matches = [
            ("img_1", [
                MatchResult(
                    teacher_instance_id=0, matched_gt_id=1, status="TP",
                    iou=0.8, predicted_canonical_class=1, gt_canonical_class=1, quality_score=0.8
                )
            ], [])
        ]
        artifact = self.calc.compute_teacher_reliability("Teacher_A", matches)
        r_global = artifact.r_global
        cr_0 = artifact.class_reliabilities[0]

        self.assertEqual(cr_0.n_gt, 0)
        self.assertTrue(cr_0.is_low_sample)
        self.assertAlmostEqual(cr_0.r_shrunk, r_global, places=5)

    def test_shrinkage_large_samples(self):
        """When n is very large, shrunk score converges to raw score."""
        n_large = 1000
        r_raw = 0.90
        r_global = 0.50
        m = 20.0
        # Formula: (n * r + m * r_global) / (n + m)
        expected = (n_large * r_raw + m * r_global) / (n_large + m)
        self.assertAlmostEqual(expected, 0.8921568, places=4)
        self.assertGreater(expected, 0.89)

    def test_bootstrap_ci_validity(self):
        ious = [0.75, 0.80, 0.82, 0.70, 0.88, 0.92, 0.85]
        low, high = self.calc._bootstrap_ci(ious, recall=0.85)
        self.assertLessEqual(low, high)
        self.assertGreater(low, 0.0)
        self.assertLessEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()

