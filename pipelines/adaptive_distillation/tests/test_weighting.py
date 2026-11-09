"""
Unit tests for trust gating, relative teacher shares, and instance GT veto.
"""

import unittest
from pipelines.adaptive_distillation.config import TrustWeightingConfig
from pipelines.adaptive_distillation.matching.trust_weighting import TrustWeighting


class TestTrustWeighting(unittest.TestCase):

    def setUp(self):
        self.cfg = TrustWeightingConfig(
            mode="adaptive",
            gate_type="smooth",
            gate_margin=0.02,
            g_min=0.0,
            share_type="softmax",
            temperature=0.50,
            s_min=0.10,
            omega_min=0.10
        )
        self.weighter = TrustWeighting(self.cfg)

    def test_trust_gate_inactive_when_below_reference(self):
        """If best teacher is below r_ref + margin, gate g[c] must be 0."""
        r_teachers = {"Teacher_A": 0.45, "Teacher_B": 0.48}
        r_ref = 0.50  # student reference is higher
        g = self.weighter.compute_gate(class_id=0, r_hat_teachers=r_teachers, r_ref=r_ref)
        self.assertEqual(g, 0.0)

    def test_trust_gate_active_when_above_reference(self):
        """If best teacher beats r_ref + margin, gate g[c] must be > 0."""
        r_teachers = {"Teacher_A": 0.75, "Teacher_B": 0.60}
        r_ref = 0.50
        g = self.weighter.compute_gate(class_id=0, r_hat_teachers=r_teachers, r_ref=r_ref)
        self.assertGreater(g, 0.0)
        self.assertLessEqual(g, 1.0)

    def test_shares_sum_to_one(self):
        """Shares must sum to 1.0 across usable teachers."""
        r_teachers = {"Teacher_A": 0.80, "Teacher_B": 0.60}
        r_ref = 0.50
        shares = self.weighter.compute_shares(class_id=0, r_hat_teachers=r_teachers, r_ref=r_ref)
        tot = sum(shares.values())
        self.assertAlmostEqual(tot, 1.0, places=5)
        self.assertGreater(shares["Teacher_A"], shares["Teacher_B"])
        self.assertGreaterEqual(shares["Teacher_B"], 0.10)  # s_min floor

    def test_three_teachers_shares(self):
        """Supports N=3 teachers seamlessly."""
        r_teachers = {"A": 0.85, "B": 0.70, "C": 0.65}
        r_ref = 0.50
        shares = self.weighter.compute_shares(class_id=0, r_hat_teachers=r_teachers, r_ref=r_ref)
        self.assertEqual(len(shares), 3)
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=5)
        self.assertTrue(shares["A"] > shares["B"] > shares["C"])

    def test_asymmetry_rule_gt_veto(self):
        """GT mismatch or poor localization MUST zero out omega (veto)."""
        omega_fp_class = self.weighter.compute_instance_omega(status="FP_class", iou=0.85)
        self.assertEqual(omega_fp_class, 0.0)

        omega_fp_loc = self.weighter.compute_instance_omega(status="FP_loc", iou=0.30)
        self.assertEqual(omega_fp_loc, 0.0)

        omega_fp_unmatched = self.weighter.compute_instance_omega(status="FP_unmatched", iou=0.0)
        self.assertEqual(omega_fp_unmatched, 0.0)

    def test_asymmetry_rule_tp_cannot_boost_above_one(self):
        """TP instance quality must never exceed 1.0."""
        omega_perfect = self.weighter.compute_instance_omega(status="TP", iou=1.0)
        self.assertEqual(omega_perfect, 1.0)

        omega_good = self.weighter.compute_instance_omega(status="TP", iou=0.75)
        self.assertGreater(omega_good, 0.10)
        self.assertLessEqual(omega_good, 1.0)


if __name__ == "__main__":
    unittest.main()

