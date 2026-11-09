"""
Distillation Trainer module — Mask2Former Student C.

Executes the training loop for Mask2Former Student C.
Guarantees that only Student C parameters are updated.
Teachers are strictly offline (never loaded into this process).

Key differences from MaskRCNN version:
  - Input to model is `pixel_values` (preprocessed by M2F processor), not raw images.
  - GT targets are passed as `mask_labels` + `class_labels`, not torchvision dicts.
  - `outputs.loss` replaces summing a loss dict.
  - Optimizer uses two param groups: backbone at 0.1× LR, head/decoder at full LR.
  - Scheduler: cosine with linear warmup over 10% of total steps.
"""

import os
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipelines.adaptive_distillation.config import DistillationPipelineConfig
from pipelines.adaptive_distillation.student.losses import DistillationLoss
from pipelines.adaptive_distillation.student.schedules import DistillationScheduler


class DistillationTrainer:
    """Trainer for Mask2Former Student C under offline multi-teacher guidance."""

    def __init__(
        self,
        student_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DistillationPipelineConfig,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model = student_model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = getattr(student_model, "processor", None)

        st_cfg = config.student

        # Two-group optimizer: backbone at 0.1× LR (even though scratch,
        # keeps consistent with normal M2F fine-tuning convention)
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "pixel_level_module.encoder" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        backbone_lr = st_cfg.lr0 * 0.1
        head_lr = st_cfg.lr0

        if st_cfg.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
                    {"params": head_params, "lr": head_lr, "name": "head"},
                ],
                weight_decay=st_cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                [
                    {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
                    {"params": head_params, "lr": head_lr, "name": "head"},
                ],
                momentum=st_cfg.momentum,
                weight_decay=st_cfg.weight_decay,
            )

        # Cosine LR schedule with linear warmup over 10% of total steps
        batches_per_epoch = max(1, len(train_loader))
        total_steps = batches_per_epoch * st_cfg.epochs
        warmup_steps = max(1, int(total_steps * 0.10))

        def _warmup_cosine(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_warmup_cosine
        )
        self._global_step = 0

        # Distillation loss & schedule
        self.loss_fn = DistillationLoss(
            config.distill,
            num_classes=32 if config.dataset.canonical_taxonomy == "32_class" else 23,
            tau=config.distill.temperature,
        )
        self.distill_scheduler = DistillationScheduler(config.distill, total_epochs=st_cfg.epochs)

        self.scaler = torch.amp.GradScaler("cuda", enabled=st_cfg.amp and self.device == "cuda")
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_map50 = -1.0
        self.best_epoch = 0

        print(f"[OK] M2F DistillationTrainer init:")
        print(f"     backbone params: {len(backbone_params)}  |  head params: {len(head_params)}")
        print(f"     backbone LR: {backbone_lr:.2e}  |  head LR: {head_lr:.2e}")
        print(f"     total steps: {total_steps}  |  warmup steps: {warmup_steps}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        lam = self.distill_scheduler.get_lambda(epoch)

        tot_loss = 0.0
        tot_gt_loss = 0.0
        tot_kd_loss = 0.0
        tot_eff_mass = 0.0
        tot_distilled_instances = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.student.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Mask2Former: use pixel_values if pre-processed by dataset, else preprocess here
            if "pixel_values" in batch:
                pixel_values = batch["pixel_values"].to(self.device)
            else:
                # Fallback: use raw images tensor; processor already applied in dataset
                pixel_values = batch["images"].to(self.device)

            gt_targets = batch["gts"]
            teacher_instances = batch["teacher_instances"]

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.config.student.amp and self.device == "cuda"):
                loss, stats = self.loss_fn(
                    self.model, pixel_values, gt_targets, teacher_instances, lam=lam
                )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[!] Warning: NaN/Inf in batch {batch_idx}. Skipping step.")
                continue

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            self._global_step += 1

            tot_loss += stats.loss_total
            tot_gt_loss += stats.loss_gt
            tot_kd_loss += stats.loss_kd
            tot_eff_mass += stats.effective_mass
            tot_distilled_instances += stats.n_instances_distilled
            n_batches += 1

            head_lr = self.optimizer.param_groups[1]["lr"]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "L_gt": f"{stats.loss_gt:.4f}",
                "L_kd": f"{stats.loss_kd:.4f}",
                "lam": f"{lam:.2f}",
                "R_kd/gt": f"{stats.kd_gt_ratio:.3f}",
                "eff_mass": f"{stats.effective_mass:.1f}",
                "head_lr": f"{head_lr:.2e}",
            })

        mean_gt = tot_gt_loss / max(n_batches, 1)
        mean_kd = tot_kd_loss / max(n_batches, 1)
        mean_ratio = float((lam * mean_kd) / (mean_gt + 1e-6))

        return {
            "epoch": epoch + 1,
            "loss_total": tot_loss / max(n_batches, 1),
            "loss_gt": mean_gt,
            "loss_kd": mean_kd,
            "effective_mass": tot_eff_mass / max(n_batches, 1),
            "kd_gt_ratio": mean_ratio,
            "instances_distilled": tot_distilled_instances,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        ckpt_path = self.output_dir / "last.pt"
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_map50": self.best_map50,
            "global_step": self._global_step,
            "config": self.config.to_dict(),
        }
        torch.save(payload, ckpt_path)

        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(payload, best_path)
            print(f"[*] Saved new best model checkpoint -> {best_path}")

    def resume_from_checkpoint(self, ckpt_path) -> int:
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint for resume not found at: {p}")
        payload = torch.load(p, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload and payload["optimizer_state_dict"]:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "best_map50" in payload:
            self.best_map50 = float(payload["best_map50"])
        if "global_step" in payload:
            self._global_step = int(payload["global_step"])
        start_epoch = int(payload.get("epoch", 0))
        print(f"[*] Resumed M2F student from {p} at epoch {start_epoch}")
        return start_epoch
