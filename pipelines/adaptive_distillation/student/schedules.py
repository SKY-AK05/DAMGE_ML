"""
Distillation Schedule module.
Computes scheduled distillation strength lambda(t) across epochs and steps.
"""

import math
from pipelines.adaptive_distillation.config import DistillationConfig


class DistillationScheduler:
    """Manages lambda(t) scheduling across training epochs."""

    def __init__(self, config: DistillationConfig, total_epochs: int):
        self.config = config
        self.total_epochs = total_epochs

    def get_lambda(self, epoch: int) -> float:
        schedule = self.config.lambda_schedule.lower()
        lam_max = self.config.lambda_max
        warmup = max(1, self.config.warmup_epochs)

        if not self.config.enabled:
            return 0.0

        if schedule == "constant":
            return lam_max

        elif schedule == "warmup":
            if epoch < warmup:
                return lam_max * (float(epoch + 1) / float(warmup))
            return lam_max

        elif schedule in ("decay", "cosine"):
            if epoch < warmup:
                return lam_max * (float(epoch + 1) / float(warmup))
            # Cosine decay to 0.1 * lambda_max
            progress = (epoch - warmup) / max(1, self.total_epochs - warmup)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lam_max * (0.1 + 0.9 * cosine_factor)

        return lam_max

