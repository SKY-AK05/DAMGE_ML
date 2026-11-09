"""
Unit tests for distillation loss numerical stability, zero-denominator guards,
and schedule computation.
"""

import unittest
import torch
import torch.nn as nn
from pycocotools import mask as mask_utils

from pipelines.adaptive_distillation.config import DistillationConfig
from pipelines.adaptive_distillation.student.losses import DistillationLoss
from pipelines.adaptive_distillation.student.schedules import DistillationScheduler
from pipelines.adaptive_distillation.cache.cache_reader import CachedInstanceRecord


class DummyStudent(nn.Module):
    """
    Minimal Mask2Former-style student for unit tests.
    Returns an object with .loss attribute matching HuggingFace API.
    """
    class _Out:
        def __init__(self, loss):
            self.loss = loss

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    def forward(self, pixel_values=None, mask_labels=None, class_labels=None, **kwargs):
        if pixel_values is None:
            # Allow old-style call during compatibility tests
            return self._Out(loss=torch.tensor(0.1, requires_grad=True))
        x = self.conv(pixel_values).mean()
        loss = x * 0.1 + x * 0.05
        return self._Out(loss=loss)


class TestDistillationLoss(unittest.TestCase):

    def setUp(self):
        self.cfg = DistillationConfig(
            enabled=True,
            lambda_schedule="warmup",
            lambda_max=1.0,
            warmup_epochs=10
        )
        self.loss_fn = DistillationLoss(self.cfg, num_classes=23)
        self.dummy_student = DummyStudent()

    def test_zero_denominator_stability(self):
        """When all instances are vetoed (weight=0), loss must remain finite (no NaN/Inf)."""
        pixel_values = torch.randn(2, 3, 64, 64)
        vetoed_inst = CachedInstanceRecord(
            canonical_class_id=0, confidence=0.8, box=[10, 10, 30, 30],
            mask_rle={"size": [64, 64], "counts": ""}, mask_prob_downscaled=None,
            omega=0.0, status="FP_class"  # Vetoed!
        )
        batch_teachers = [[vetoed_inst], [vetoed_inst]]

        loss_kd, eff_mass, n_dist, n_vetoed = self.loss_fn.compute_kd_loss(
            self.dummy_student, pixel_values, batch_teachers
        )

        self.assertFalse(torch.isnan(loss_kd))
        self.assertFalse(torch.isinf(loss_kd))
        self.assertEqual(eff_mass, 0.0)
        self.assertEqual(n_dist, 0)
        self.assertEqual(n_vetoed, 2)

    def test_empty_teacher_record(self):
        """Empty teacher detection records contribute 0 KD loss without crash."""
        pixel_values = torch.randn(2, 3, 64, 64)
        batch_teachers = [[], []]

        loss_kd, eff_mass, n_dist, n_vetoed = self.loss_fn.compute_kd_loss(
            self.dummy_student, pixel_values, batch_teachers
        )
        self.assertEqual(loss_kd.item(), 0.0)
        self.assertEqual(eff_mass, 0.0)

    def test_distillation_scheduler(self):
        scheduler = DistillationScheduler(self.cfg, total_epochs=50)
        # Epoch 0 warmup start
        self.assertEqual(scheduler.get_lambda(epoch=0), 0.1)
        # Epoch 9 warmup end
        self.assertAlmostEqual(scheduler.get_lambda(epoch=9), 1.0, places=5)
        # Epoch 15 constant phase
        self.assertEqual(scheduler.get_lambda(epoch=15), 1.0)


if __name__ == "__main__":
    unittest.main()

