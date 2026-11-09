#!/usr/bin/env python3
"""
Stage B: Student C Training.
Trains Student C model from scratch using ground-truth supervised loss +
offline teacher guidance streamed from disk cache.
Teachers are frozen and NOT loaded in GPU memory during training.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.adaptive_distillation.config import DistillationPipelineConfig
from pipelines.adaptive_distillation.cache.cache_reader import CacheReader
from pipelines.adaptive_distillation.cache.cache_manifest import CacheManifestManager
from pipelines.adaptive_distillation.data.dataset_adapter import DistillationDataset, distillation_collate_fn
from pipelines.adaptive_distillation.data.transforms import AlignedTransforms
from pipelines.adaptive_distillation.student.student_builder import StudentBuilder
from pipelines.adaptive_distillation.student.distillation_trainer import DistillationTrainer
from pipelines.adaptive_distillation.evaluation.evaluator import PipelineEvaluator
from pipelines.adaptive_distillation.utils.reproducibility import set_seed, capture_environment_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Train Student C with Offline Multi-Teacher Distillation")
    parser.add_argument("--config", default="pipelines/adaptive_distillation/configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--mode", default=None, help="Override weight mode (adaptive, fixed, shuffled, inverse, uniform_in_gate)")
    parser.add_argument("--no_distill", action="store_true", help="Disable distillation (pure GT-only Student C baseline E2)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--teachers", nargs="+", default=None, help="Filter to specific teacher names (e.g. teacher_a_m2f)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DistillationPipelineConfig.from_yaml(args.config)

    # CLI Overrides
    if args.mode:
        cfg.weights.mode = args.mode
    if args.no_distill:
        cfg.distill.enabled = False
        cfg.experiment_name = "e2_gt_only_baseline"
    if args.epochs:
        cfg.student.epochs = args.epochs
    if args.batch_size:
        cfg.student.batch_size = args.batch_size
    if args.exp_name:
        cfg.experiment_name = args.exp_name
    if args.teachers:
        t_filter = [x.lower() for x in args.teachers]
        filtered = [t for t in cfg.teachers if t.name.lower() in t_filter or any(f in t.name.lower() for f in t_filter)]
        if filtered:
            cfg.teachers = filtered
            print(f"[*] Filtered teachers to: {[t.name for t in cfg.teachers]}")
        else:
            print(f"[!] Warning: None of specified teachers {args.teachers} matched. Using {cfg.teachers}")

    # Set seed for reproducibility
    set_seed(cfg.evaluation.seed)
    env_meta = capture_environment_metadata()
    print(f"\n[*] Environment: PyTorch {env_meta['torch_version']} | CUDA: {env_meta['cuda_available']} | GPU: {env_meta['gpu_name']}")
    print(f"[*] Starting Student C Training: Experiment '{cfg.experiment_name}'")
    print(f"[*] Distillation Enabled: {cfg.distill.enabled} | Weight Mode: {cfg.weights.mode}")

    cache_dir = Path(cfg.cache.dir)

    # Validate cache manifest if distillation is enabled
    cache_readers = {}
    if cfg.distill.enabled:
        is_valid, errs = CacheManifestManager.validate_manifest_against_config(cache_dir, cfg)
        if not is_valid:
            print("[ERROR] Cache manifest validation failed:")
            for e in errs:
                print(f"  - {e}")
            print("[!] Please run prepare_teacher_cache.py before training Student C.")
            sys.exit(1)

        for t in cfg.teachers:
            cache_readers[t.name] = CacheReader(cache_dir, t.name)
        print(f"[OK] Cache readers initialized for {list(cache_readers.keys())}. Teachers are offline.")

    # Build Student C from scratch
    student_model = StudentBuilder.build(
        cfg.student,
        num_classes=23 if cfg.dataset.canonical_taxonomy == "23_class" else 32,
        seed=cfg.evaluation.seed
    )
    print(f"[OK] Student C ({cfg.student.arch}) initialized from scratch. No pretrained weights.")

    # Datasets and loaders
    train_transforms = AlignedTransforms(
        level=cfg.augmentation.level,
        hflip_prob=cfg.augmentation.hflip_prob,
        target_size=cfg.dataset.imgsz
    )
    train_dataset = DistillationDataset(
        images_dir=cfg.dataset.train_images,
        coco_json_path=cfg.dataset.train_json,
        cache_readers=cache_readers,
        transforms=train_transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.student.batch_size,
        shuffle=True,
        num_workers=cfg.student.workers,
        collate_fn=distillation_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    val_dataset = DistillationDataset(
        images_dir=cfg.dataset.val_images,
        coco_json_path=cfg.dataset.val_json,
        transforms=AlignedTransforms(level="L0", target_size=cfg.dataset.imgsz)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.student.batch_size,
        shuffle=False,
        num_workers=cfg.student.workers,
        collate_fn=distillation_collate_fn
    )

    trainer = DistillationTrainer(
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg
    )

    print(f"[*] Training on {len(train_dataset)} images for {cfg.student.epochs} epochs...")
    for epoch in range(cfg.student.epochs):
        stats = trainer.train_epoch(epoch)
        print(
            f"Epoch {stats['epoch']}/{cfg.student.epochs} | "
            f"Loss: {stats['loss_total']:.4f} (GT: {stats['loss_gt']:.4f}, KD: {stats['loss_kd']:.4f}) | "
            f"Eff Mass: {stats['effective_mass']:.2f}"
        )
        trainer.save_checkpoint(stats['epoch'], is_best=(epoch == cfg.student.epochs - 1))

    print(f"\n[SUCCESS] Student C Training finished for experiment '{cfg.experiment_name}'!")


if __name__ == "__main__":
    main()

