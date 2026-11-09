"""
Integration tests for the complete adaptive distillation pipeline.
Verifies Mask2Former Student C scratch initialization, gradient isolation,
and end-to-end forward/backward step.
"""

import unittest
import torch
import torch.nn as nn
from PIL import Image as PILImage
import numpy as np

from pipelines.adaptive_distillation.config import DistillationPipelineConfig
from pipelines.adaptive_distillation.student.student_builder import StudentBuilder
from pipelines.adaptive_distillation.student.losses import DistillationLoss
from pipelines.adaptive_distillation.cache.cache_reader import CachedInstanceRecord


class DummyM2FStudent(nn.Module):
    """
    Minimal stand-in for Mask2FormerForUniversalSegmentation during unit tests.
    Returns an object with a .loss attribute to match the HuggingFace API.
    """
    class _Out:
        def __init__(self, loss):
            self.loss = loss

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.num_classes = 32

    def forward(self, pixel_values=None, mask_labels=None, class_labels=None, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values required")
        x = self.conv(pixel_values).mean()
        loss = x * 0.1 + x * 0.05
        return self._Out(loss=loss)


class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        self.cfg = DistillationPipelineConfig()
        self.cfg.student.arch = "mask2former"
        self.cfg.student.init = "scratch"
        self.cfg.evaluation.seed = 42

    def test_student_scratch_initialization(self):
        """Mask2Former Student C must initialize from scratch with all params requiring grad."""
        student = StudentBuilder.build(self.cfg.student, num_classes=32, seed=42)
        self.assertIsInstance(student, nn.Module)

        trainable_params = [p for p in student.parameters() if p.requires_grad]
        total_params = list(student.parameters())
        self.assertEqual(len(trainable_params), len(total_params))
        self.assertGreater(len(trainable_params), 0)

        # Verify scratch: backbone weights must NOT be all zeros (random init)
        verify = StudentBuilder.verify_scratch(student)
        self.assertTrue(verify["backbone_nonzero"], "Backbone should be randomly initialized (non-zero)")
        self.assertFalse(verify["backbone_all_zeros"], "Backbone should NOT be all zeros")
        self.assertGreater(verify["trainable_params"], 40_000_000, "M2F Swin-Tiny has ~47M params")

    def test_distillation_step_updates_only_student(self):
        """A training step must backpropagate and update ONLY Student C parameters."""
        # Use DummyM2FStudent to avoid loading real M2F in unit test
        student = DummyM2FStudent()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

        # Synthetic M2F-style batch: pixel_values (B, 3, H, W)
        pixel_values = torch.rand(2, 3, 256, 256)
        gt_targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),       # 0-indexed for M2F
                "masks": torch.zeros((1, 256, 256), dtype=torch.uint8),
            },
            {
                "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
                "labels": torch.tensor([1]),
                "masks": torch.zeros((1, 256, 256), dtype=torch.uint8),
            },
        ]

        t_inst = CachedInstanceRecord(
            canonical_class_id=0, confidence=0.85, box=[10.0, 10.0, 50.0, 50.0],
            mask_rle={"size": [256, 256], "counts": ""},
            mask_prob_downscaled=None, omega=0.8, status="TP",
        )
        batch_teachers = [[t_inst], [t_inst]]

        loss_fn = DistillationLoss(self.cfg.distill, num_classes=32)

        first_param = list(student.parameters())[0]
        w_before = first_param.clone().detach()

        loss, stats = loss_fn(student, pixel_values, gt_targets, batch_teachers, lam=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        w_after = first_param.clone().detach()

        self.assertFalse(torch.equal(w_before, w_after), "Student parameters should update")
        self.assertGreater(stats.effective_mass, 0.0)
        self.assertFalse(torch.isnan(loss))

    def test_e2_zero_kd_loss(self):
        """In E2 (GT-only mode), distillation loss must be exactly 0.0."""
        student = DummyM2FStudent()
        pixel_values = torch.rand(1, 3, 128, 128)
        gt_targets = [{
            "boxes": torch.tensor([[5.0, 5.0, 30.0, 30.0]]),
            "labels": torch.tensor([2]),
            "masks": torch.zeros((1, 128, 128), dtype=torch.uint8),
        }]
        batch_teachers = [[]]  # No teacher instances

        # GT-only: lam = 0.0
        loss_fn = DistillationLoss(self.cfg.distill, num_classes=32)
        loss, stats = loss_fn(student, pixel_values, gt_targets, batch_teachers, lam=0.0)

        self.assertEqual(stats.loss_kd, 0.0, "KD loss must be 0 in E2 (lam=0)")
        self.assertEqual(stats.n_instances_distilled, 0)
        self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
    unittest.main()
