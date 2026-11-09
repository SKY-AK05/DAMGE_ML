#!/usr/bin/env python3
"""
run_training.py -- Unified Interactive End-to-End Adaptive Multi-Teacher Distillation Workflow.

Integrates teacher discovery from runs_comparison/, dataset selection, scratch Student C configuration,
automated cache compatibility checking, intelligent incremental cache generation, reliability validation,
GPU-isolated training, and automatic evaluation.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.adaptive_distillation.config import (
    DistillationPipelineConfig, TeacherConfig, DatasetConfig, StudentConfig,
    TrustWeightingConfig, DistillationConfig, AugmentationConfig
)
from pipelines.adaptive_distillation.teachers.teacher_scanner import TeacherScanner, DiscoveredTeacher
from pipelines.adaptive_distillation.teachers.teacher_loader import TeacherLoader, compute_path_sha256
from pipelines.adaptive_distillation.cache.cache_compatibility_manager import (
    CacheCompatibilityManager, CacheCompatibilityReport, CacheStatus, CacheAction
)
from pipelines.adaptive_distillation.cache.cache_reader import CacheReader
from pipelines.adaptive_distillation.data.dataset_adapter import DistillationDataset, distillation_collate_fn
from pipelines.adaptive_distillation.data.transforms import AlignedTransforms
from pipelines.adaptive_distillation.matching.teacher_fusion import TeacherFusion
from pipelines.adaptive_distillation.student.student_builder import StudentBuilder
from pipelines.adaptive_distillation.student.distillation_trainer import DistillationTrainer
from pipelines.adaptive_distillation.evaluation.evaluator import PipelineEvaluator
from pipelines.adaptive_distillation.utils.reproducibility import set_seed, capture_environment_metadata


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "pipelines" / "adaptive_distillation" / "configs" / "default.yaml"


def prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (KeyboardInterrupt, EOFError):
        print("\n[!] Exiting.")
        sys.exit(0)


def prompt_default(label: str, default: Any) -> str:
    val = prompt(f"  {label} [default: {default}]: ")
    return val if val else str(default)


class DistillationWorkflow:
    """Manages the full lifecycle of adaptive multi-teacher distillation experiments."""

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        if self.config_path.exists():
            self.cfg = DistillationPipelineConfig.from_yaml(self.config_path)
        else:
            self.cfg = DistillationPipelineConfig()
            self.cfg.save_yaml(self.config_path)

        self.scanner = TeacherScanner()
        self.compat_manager = CacheCompatibilityManager(self.cfg)
        self.last_report: Optional[CacheCompatibilityReport] = None

    def save_config(self) -> None:
        self.cfg.save_yaml(self.config_path)
        # Refresh compatibility manager
        self.compat_manager = CacheCompatibilityManager(self.cfg)

    # -------------------------------------------------------------
    # Step 1: Select Teachers
    # -------------------------------------------------------------
    def step_select_teachers(self) -> None:
        print("\n" + "=" * 75)
        print(" [STEP 1] SELECT CANDIDATE TEACHER MODELS (from runs_comparison/)")
        print("=" * 75)
        target_tax = self.cfg.dataset.canonical_taxonomy
        candidates = self.scanner.scan(target_taxonomy=target_tax)
        if not candidates:
            print("[!] No candidate teacher models found in runs_comparison/.")
            return

        self.scanner.display_candidates(candidates, target_taxonomy=target_tax)

        print("\nCurrent Teacher Team:")
        for t in self.cfg.teachers:
            print(f"  * {t.name}: {t.arch} ({t.checkpoint_path}) [Run: {t.run_id}]")

        print("\nChoose an action:")
        print("  1) Re-select teachers interactively")
        print("  2) Keep current teacher team")
        print("  3) Add a custom checkpoint path")

        choice = prompt("Enter choice [1-3] (default: 2): ") or "2"
        if choice == "1":
            new_team = self.scanner.interactive_select(target_taxonomy=target_tax)
            if new_team:
                self.cfg.teachers = new_team
                self.save_config()
                print(f"[OK] Saved {len(new_team)} teachers to configuration.")
        elif choice == "3":
            custom_path = prompt("Enter path to checkpoint weights directory or file: ")
            if custom_path and Path(custom_path).exists():
                p = Path(custom_path)
                arch = "mask2former" if (p / "model.safetensors").exists() or p.is_dir() else "yolo"
                idx = len(self.cfg.teachers) + 1
                new_t = TeacherConfig(
                    name=f"Teacher_{chr(64 + idx)}",
                    arch=arch,
                    checkpoint_path=str(p.resolve()),
                    run_id=p.name,
                    class_map=target_tax
                )
                self.cfg.teachers.append(new_t)
                self.save_config()
                print(f"[OK] Added {new_t.name}: {new_t.arch} ({new_t.checkpoint_path})")
            else:
                print(f"[!] Path does not exist: {custom_path}")

    # -------------------------------------------------------------
    # Step 2: Select Dataset
    # -------------------------------------------------------------
    def step_select_dataset(self) -> None:
        print("\n" + "=" * 75)
        print(" [STEP 2] SELECT DATASET & ANNOTATIONS")
        print("=" * 75)
        print("Available datasets in repository:")
        print("  1) RAW_DATASET2 (Native 32-class car-parts, 6,516 images: 5,212 train / 652 calib / 652 val) [Recommended]")
        print("  2) combined_carparts (Legacy 23-class car-parts dataset)")
        print("  3) Custom dataset path")

        choice = prompt("Select dataset [1-3] (default: 1): ") or "1"
        if choice == "1":
            self.cfg.dataset = DatasetConfig(
                name="RAW_DATASET2",
                root="/home/orchvate/CAR_TRAIN1/RAW_DATASET2",
                train_images="/home/orchvate/CAR_TRAIN1/RAW_DATASET2/images",
                val_images="/home/orchvate/CAR_TRAIN1/RAW_DATASET2/images",
                test_images="/home/orchvate/CAR_TRAIN1/EVA/GROUND/images",
                train_json="/home/orchvate/CAR_TRAIN1/RAW_DATASET2/annotations/train_5212.json",
                calib_json="/home/orchvate/CAR_TRAIN1/RAW_DATASET2/annotations/calib_652.json",
                val_json="/home/orchvate/CAR_TRAIN1/RAW_DATASET2/annotations/val_652.json",
                test_json="/home/orchvate/CAR_TRAIN1/EVA/GROUND/coco_annotations.json",
                canonical_taxonomy="32_class",
                imgsz=640
            )
            self.save_config()
            print("[OK] Selected RAW_DATASET2.")

        elif choice == "2":
            comb_dir = PROJECT_ROOT / "datasets" / "combined_carparts"
            self.cfg.dataset = DatasetConfig(
                name="combined_carparts",
                root=str(comb_dir),
                train_images=str(comb_dir / "train"),
                val_images=str(comb_dir / "valid"),
                test_images=str(comb_dir / "test"),
                train_json=str(comb_dir / "train" / "_annotations.coco.json"),
                val_json=str(comb_dir / "valid" / "_annotations.coco.json"),
                test_json=str(comb_dir / "test" / "_annotations.coco.json"),
                canonical_taxonomy="23_class",
                imgsz=640
            )
            self.save_config()
            print("[OK] Selected combined_carparts (23 classes).")

        elif choice == "3":
            root_d = prompt("Enter dataset root directory: ")
            train_j = prompt("Enter train annotations JSON path: ")
            val_j = prompt("Enter val annotations JSON path: ")
            img_d = prompt("Enter images directory: ")
            tax = prompt("Enter taxonomy [32_class / 23_class] (default: 32_class): ") or "32_class"
            self.cfg.dataset = DatasetConfig(
                name=Path(root_d).name if root_d else "custom_dataset",
                root=root_d,
                train_images=img_d,
                val_images=img_d,
                test_images=img_d,
                train_json=train_j,
                val_json=val_j,
                test_json=val_j,
                canonical_taxonomy=tax,
                imgsz=640
            )
            self.save_config()
            print(f"[OK] Custom dataset configured: {self.cfg.dataset.name}")

        # Display dataset summary
        try:
            with open(self.cfg.dataset.train_json, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            n_imgs = len(cdata.get("images", []))
            n_anns = len(cdata.get("annotations", []))
            n_cats = len(cdata.get("categories", []))
            print(f"\n[*] Dataset Summary for {self.cfg.dataset.name}:")
            print(f"    - Training Images:       {n_imgs:,}")
            print(f"    - Total Annotations:     {n_anns:,}")
            print(f"    - Distinct Categories:   {n_cats}")
            print(f"    - Canonical Taxonomy:    {self.cfg.dataset.canonical_taxonomy}")
        except Exception as e:
            print(f"[!] Could not load dataset summary: {e}")

    # -------------------------------------------------------------
    # Step 3: Configure Student C
    # -------------------------------------------------------------
    def step_configure_student(self) -> None:
        print("\n" + "=" * 75)
        print(" [STEP 3] CONFIGURE STUDENT C ARCHITECTURE")
        print("=" * 75)
        print("Student Architecture: Mask2Former (Swin-Tiny, 47.4M params)")
        print("  * Architecture:    mask2former (Swin-Tiny Transformer)")
        print("  * Init:            STRICT SCRATCH (Mask2FormerConfig-only, no from_pretrained() on model)")
        print("  * Pretrained:      NONE — backbone, pixel decoder, and transformer decoder all random")
        print("  * Classes:         32 foreground classes (native RAW_DATASET2 taxonomy)")
        print("  * Queries:         100 object queries (Mask2Former standard)")
        print()
        print("  [NOTE] Only Mask2Former is supported for Student C.")
        print("         Teacher A and Teacher B are both Mask2Former models.")
        print("         Mask R-CNN and YOLO are not compatible with this research design.")

        self.cfg.student.arch = "mask2former"
        self.cfg.student.init = "scratch"
        self.save_config()

        print(f"\n[OK] Configured Student C:")
        print(f"    - Architecture:       mask2former (Swin-Tiny)")
        print(f"    - Initialization:     STRICT SCRATCH (zero pretrained weight leakage)")

    # -------------------------------------------------------------
    # Step 4: Configure Training Experiment
    # -------------------------------------------------------------
    def step_configure_training(self) -> None:
        print("\n" + "=" * 75)
        print(" [STEP 4] CONFIGURE TRAINING EXPERIMENT & HYPERPARAMETERS")
        print("=" * 75)

        print("Select Training Distillation Mode:")
        print("  1) adaptive  -- Ground-Truth-Guided Adaptive Multi-Teacher KD (Dynamic trust weighting) [Recommended]")
        print("  2) fixed     -- Fixed Multi-Teacher KD (Uniform 50/50 weighting)")
        print("  3) single    -- Single-Teacher KD (Distill from Teacher_A only)")
        print("  4) gt_only   -- Pure Ground-Truth Supervised Baseline E2 (No teacher guidance)")

        m_choice = prompt("Select mode [1-4] (default: 1): ") or "1"
        mode_map = {"1": "adaptive", "2": "fixed", "3": "fixed", "4": "gt_only"}
        sel_mode = mode_map.get(m_choice, "adaptive")

        if sel_mode == "gt_only":
            self.cfg.distill.enabled = False
            self.cfg.weights.mode = "fixed"
            default_exp = "e2_gt_only_scratch"
        elif m_choice == "3":
            self.cfg.distill.enabled = True
            self.cfg.weights.mode = "fixed"
            print("Select single teacher to distill from:")
            print("  1) Teacher A (run_20261031_150325, mAP50 0.7972) -> Experiment E3")
            print("  2) Teacher B (run_20261022_005324, mAP50 0.7577) -> Experiment E4")
            t_choice = prompt("Select teacher [1-2] (default: 1): ") or "1"
            if t_choice == "2":
                chosen_t = next((t for t in self.cfg.teachers if "teacher_b" in t.name.lower() or "b" in t.name.lower()), self.cfg.teachers[-1])
                default_exp = "e4_singleteacher_b"
            else:
                chosen_t = next((t for t in self.cfg.teachers if "teacher_a" in t.name.lower() or "a" in t.name.lower()), self.cfg.teachers[0])
                default_exp = "e3_singleteacher_a"
            self.cfg.teachers = [chosen_t]
            print(f"[OK] Isolated single teacher to: {chosen_t.name} ({chosen_t.arch})")
        elif sel_mode == "fixed":
            self.cfg.distill.enabled = True
            self.cfg.weights.mode = "fixed"
            default_exp = "e5_fixed_multiteacher"
        else:
            self.cfg.distill.enabled = True
            self.cfg.weights.mode = "adaptive"
            default_exp = "e6_adaptive_multiteacher"

        exp_name = prompt_default("Experiment Name", default_exp)
        epochs = int(prompt_default("Number of Epochs", self.cfg.student.epochs or 50))
        batch_size = int(prompt_default("Batch Size", self.cfg.student.batch_size or 8))
        lr = float(prompt_default("Learning Rate (lr0)", self.cfg.student.lr0 or 0.005))
        optimizer = prompt_default("Optimizer (adamw/sgd)", self.cfg.student.optimizer or "adamw")
        workers = int(prompt_default("DataLoader Workers", self.cfg.student.workers or 4))
        temp = float(prompt_default("Distillation Temperature (tau)", self.cfg.distill.temperature or 2.0))

        self.cfg.experiment_name = exp_name
        self.cfg.student.epochs = epochs
        self.cfg.student.batch_size = batch_size
        self.cfg.student.lr0 = lr
        self.cfg.student.optimizer = optimizer
        self.cfg.student.workers = workers
        self.cfg.distill.temperature = temp

        self.save_config()

        print("\nResolved Training Configuration:")
        print(f"  * Experiment:    {self.cfg.experiment_name}")
        print(f"  * Mode:          {self.cfg.weights.mode} (Distill Enabled: {self.cfg.distill.enabled})")
        print(f"  * Epochs:        {self.cfg.student.epochs}")
        print(f"  * Batch Size:    {self.cfg.student.batch_size}")
        print(f"  * LR:            {self.cfg.student.lr0}")
        print(f"  * Optimizer:     {self.cfg.student.optimizer}")
        print(f"  * Temperature:   {self.cfg.distill.temperature}")
        print(f"  * Workers:       {self.cfg.student.workers}")
        print(f"  * Seed:          {self.cfg.evaluation.seed}")

    # -------------------------------------------------------------
    # Step 5: Check Cache Compatibility
    # -------------------------------------------------------------
    def step_check_cache(self) -> CacheCompatibilityReport:
        print("\n" + "=" * 75)
        print(" [STEP 5] CHECKING TEACHER CACHE COMPATIBILITY")
        print("=" * 75)
        self.compat_manager = CacheCompatibilityManager(self.cfg)
        rep = self.compat_manager.check_compatibility()
        self.last_report = rep
        print(rep.text_report)
        return rep

    # -------------------------------------------------------------
    # Step 6: Prepare Missing Cache
    # -------------------------------------------------------------
    def step_prepare_cache(self) -> bool:
        print("\n" + "=" * 75)
        print(" [STEP 6] PREPARING MISSING OR INCOMPLETE CACHE")
        print("=" * 75)
        if not self.last_report:
            self.last_report = self.compat_manager.check_compatibility()

        if self.last_report.is_fully_compatible:
            print("[OK] Cache is already 100% valid and verified on disk. No GPU re-inference required!")
            return True

        print("Required Cache Actions:")
        for t_name, s in self.last_report.teachers.items():
            print(f"  - {t_name}: {s.action.value} ({s.action_detail})")

        confirm = prompt("\nExecute cache preparation now? [Y/N] (default: Y): ") or "Y"
        if confirm.upper() != "Y":
            print("[!] Cache preparation cancelled.")
            return False

        success = self.compat_manager.prepare_missing_cache(self.last_report)
        if success:
            # Refresh report
            self.last_report = self.compat_manager.check_compatibility()
        return success

    # -------------------------------------------------------------
    # Step 7: Validate Teacher Reliability
    # -------------------------------------------------------------
    def step_validate_reliability(self) -> None:
        print("\n" + "=" * 75)
        print(" [STEP 7] VALIDATE TEACHER RELIABILITY CALIBRATION")
        print("=" * 75)
        rel_path = PROJECT_ROOT / "reliability" / "calibration_report.md"
        if not rel_path.exists():
            print("[!] Calibration report not found. Running calibration check...")
            self.compat_manager.check_compatibility()

        if rel_path.exists():
            lines = rel_path.read_text(encoding="utf-8").splitlines()
            for line in lines[:35]:
                print(line)
            if len(lines) > 35:
                print(f"  ... [{len(lines) - 35} more lines in {rel_path}]")
        else:
            print("[!] Calibration report still missing.")

        print("\n[!] Asymmetry Rule Notice:")
        print("    GT matches can veto teacher guidance (setting omega=0.0),")
        print("    but can never artificially inflate teacher trust beyond the calibration ceiling.")

    # -------------------------------------------------------------
    # Step 8: Initialize Student C
    # -------------------------------------------------------------
    def step_initialize_student(self) -> torch.nn.Module:
        print("\n" + "=" * 75)
        print(" [STEP 8] INITIALIZE STUDENT C FROM SCRATCH")
        print("=" * 75)
        num_classes = 23 if self.cfg.dataset.canonical_taxonomy == "23_class" else 32
        student_model = StudentBuilder.build(
            self.cfg.student,
            num_classes=num_classes,
            seed=self.cfg.evaluation.seed
        )

        all_params = list(student_model.parameters())
        trainable_params = [p for p in all_params if p.requires_grad]
        tot_numel = sum(p.numel() for p in all_params)

        print(f"[OK] Model successfully initialized: {self.cfg.student.arch}")
        print(f"     Total parameter tensors:     {len(all_params)}")
        print(f"     Trainable parameter tensors: {len(trainable_params)} (100% require_grad=True)")
        print(f"     Total parameter count:       {tot_numel:,}")
        print(f"     Pretrained weights:          None (Strict scratch initialization)")
        return student_model

    # -------------------------------------------------------------
    # Step 9: Start Training
    # -------------------------------------------------------------
    def step_start_training(self) -> None:
        print("\n" + "=" * 75)
        print(" [STEP 9] PRE-FLIGHT VALIDATION & STUDENT C TRAINING")
        print("=" * 75)

        # 1. Verify cache compatibility
        if not self.last_report or not self.last_report.is_fully_compatible:
            print("[*] Re-checking cache compatibility...")
            self.last_report = self.compat_manager.check_compatibility()

        if self.cfg.distill.enabled and not self.last_report.is_fully_compatible:
            print("[!] Cache is not fully compatible! Resolving missing cache records first...")
            success = self.step_prepare_cache()
            if not success:
                print("[ERROR] Cannot proceed to training without complete cache.")
                return

        # 2. Check GPU environment
        env_meta = capture_environment_metadata()
        print(f"\nEnvironment Pre-Flight:")
        print(f"  * PyTorch:       {env_meta['torch_version']}")
        print(f"  * CUDA:          {env_meta['cuda_available']}")
        print(f"  * GPU:           {env_meta['gpu_name']}")
        print(f"  * VRAM:          {env_meta.get('gpu_memory_gb', 'N/A')} GB")

        # 3. Pre-flight summary
        print("\nPre-Flight Training Summary:")
        print(f"  * Experiment:    {self.cfg.experiment_name}")
        print(f"  * Mode:          {self.cfg.weights.mode} (Distill: {self.cfg.distill.enabled})")
        print(f"  * Teachers:      {[t.name for t in self.cfg.teachers]}")
        print(f"  * Dataset:       {self.cfg.dataset.name} ({self.cfg.dataset.canonical_taxonomy})")
        print(f"  * Student:       {self.cfg.student.arch} (Init: {self.cfg.student.init})")
        print(f"  * Epochs:        {self.cfg.student.epochs}")
        print(f"  * Batch Size:    {self.cfg.student.batch_size}")
        print(f"  * Cache Dir:     {self.cfg.cache.dir}")
        print(f"  * Memory Guard:  Teachers 100% offline (Zero teacher weights in VRAM)")

        confirm = prompt("\nStart training? [Y/N] (default: Y): ") or "Y"
        if confirm.upper() != "Y":
            print("[!] Training aborted by user.")
            return

        # Set seed
        set_seed(self.cfg.evaluation.seed)

        # Build Student C
        student_model = self.step_initialize_student()

        # Initialize Cache Readers
        cache_readers = {}
        if self.cfg.distill.enabled:
            for t in self.cfg.teachers:
                cache_readers[t.name] = CacheReader(self.cfg.cache.dir, t.name)
            print(f"[OK] Initialized cache readers for {list(cache_readers.keys())}. Teachers are offline.")

        # Build Datasets and DataLoaders
        train_transforms = AlignedTransforms(
            level=self.cfg.augmentation.level,
            hflip_prob=self.cfg.augmentation.hflip_prob,
            target_size=self.cfg.dataset.imgsz
        )
        fusion_engine = TeacherFusion(self.cfg.fusion) if self.cfg.distill.enabled else None
        processor = getattr(student_model, "processor", None)

        train_dataset = DistillationDataset(
            images_dir=self.cfg.dataset.train_images,
            coco_json_path=self.cfg.dataset.train_json,
            cache_readers=cache_readers,
            transforms=train_transforms,
            processor=processor,
            fusion_engine=fusion_engine,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.student.batch_size,
            shuffle=True,
            num_workers=self.cfg.student.workers,
            collate_fn=distillation_collate_fn,
            pin_memory=torch.cuda.is_available()
        )

        val_dataset = DistillationDataset(
            images_dir=self.cfg.dataset.val_images,
            coco_json_path=self.cfg.dataset.val_json,
            transforms=AlignedTransforms(level="L0", target_size=self.cfg.dataset.imgsz),
            processor=processor,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg.student.batch_size,
            shuffle=False,
            num_workers=self.cfg.student.workers,
            collate_fn=distillation_collate_fn
        )

        trainer = DistillationTrainer(
            student_model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.cfg
        )

        # Check for checkpoint resume
        ckpt_last = Path(self.cfg.output_dir) / self.cfg.experiment_name / "last.pt"
        start_epoch = 0
        if ckpt_last.exists():
            resume_choice = prompt(f"Found existing checkpoint at {ckpt_last}. Resume training? [Y/N] (default: N): ") or "N"
            if resume_choice.upper() == "Y":
                start_epoch = trainer.resume_from_checkpoint(ckpt_last)

        print(f"\n[*] Starting training loop from epoch {start_epoch + 1} to {self.cfg.student.epochs}...")
        for epoch in range(start_epoch, self.cfg.student.epochs):
            stats = trainer.train_epoch(epoch)
            print(
                f"Epoch {stats['epoch']}/{self.cfg.student.epochs} | "
                f"Loss: {stats['loss_total']:.4f} (GT: {stats['loss_gt']:.4f}, KD: {stats['loss_kd']:.4f}) | "
                f"Eff Mass: {stats['effective_mass']:.2f} | R_kd/gt: {stats['kd_gt_ratio']:.3f}"
            )
            trainer.save_checkpoint(stats['epoch'], is_best=(epoch == self.cfg.student.epochs - 1))

        print(f"\n[SUCCESS] Training finished for experiment '{self.cfg.experiment_name}'!")

        # Run automatic post-training evaluation
        eval_choice = prompt("Run automatic evaluation on validation split now? [Y/N] (default: Y): ") or "Y"
        if eval_choice.upper() == "Y":
            self.step_evaluate()

    # -------------------------------------------------------------
    # Step 10: Evaluate and Generate Report
    # -------------------------------------------------------------
    def step_evaluate(self) -> None:
        print("\n" + "=" * 75)
        print(" [STEP 10] EVALUATE STUDENT C ON VALIDATION SPLIT")
        print("=" * 75)
        ckpt_path = Path(self.cfg.output_dir) / self.cfg.experiment_name / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = Path(self.cfg.output_dir) / self.cfg.experiment_name / "last.pt"

        if not ckpt_path.exists():
            print(f"[!] No checkpoint found to evaluate at: {ckpt_path}")
            return

        val_json = self.cfg.dataset.val_json
        canonical_classes = CLASSES_32 if self.cfg.dataset.canonical_taxonomy == "32_class" else CLASSES_23

        print(f"[*] Evaluating checkpoint: {ckpt_path}")
        print(f"[*] Split: val ({val_json})")
        print(f"[*] Preserving EVA/GROUND untouched as final test benchmark.")

        evaluator = PipelineEvaluator(
            val_json_path=val_json,
            class_names=canonical_classes,
            config=self.cfg.evaluation
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = torch.load(ckpt_path, map_location=device)
        model = StudentBuilder.build(self.cfg.student, num_classes=len(canonical_classes))
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        model.eval()

        with open(val_json, "r", encoding="utf-8") as f:
            v_coco = json.load(f)
        val_images = v_coco.get("images", [])

        import torchvision.transforms.functional as F
        from PIL import Image

        predictions = []
        print(f"[*] Running inference on {len(val_images)} validation images...")
        with torch.no_grad():
            for img_info in val_images:
                iid = img_info["id"]
                fname = img_info["file_name"]
                img_p = Path(self.cfg.dataset.val_images) / fname
                if not img_p.exists():
                    img_p = Path(self.cfg.dataset.root) / "images" / fname
                if not img_p.exists():
                    continue

                pil_img = Image.open(img_p).convert("RGB")
                w, h = pil_img.size
                t_img = F.to_tensor(pil_img).to(device)

                preds = model([t_img])[0]
                bxs = preds["boxes"].cpu().numpy()
                lbls = preds["labels"].cpu().numpy()
                scs = preds["scores"].cpu().numpy()
                msks = preds["masks"].squeeze(1).cpu().numpy() if "masks" in preds else None

                from pycocotools import mask as mask_utils
                for bi, box in enumerate(bxs):
                    score = float(scs[bi])
                    if score < self.cfg.evaluation.conf:
                        continue
                    cid = int(lbls[bi])
                    # Mask R-CNN 1-based indexing
                    if cid == 0:
                        continue
                    x1, y1, x2, y2 = box
                    bw = max(0.0, float(x2 - x1))
                    bh = max(0.0, float(y2 - y1))
                    rle = None
                    if msks is not None:
                        bin_m = (msks[bi] > 0.5).astype(np.uint8)
                        rle = mask_utils.encode(np.asfortranarray(bin_m))
                        rle["counts"] = rle["counts"].decode("utf-8")

                    predictions.append({
                        "image_id": iid,
                        "category_id": cid,
                        "bbox": [float(x1), float(y1), bw, bh],
                        "score": score,
                        "segmentation": rle
                    })

        metrics = evaluator.evaluate_predictions(predictions)
        print("\n" + "=" * 55)
        print(" VALIDATION EVALUATION RESULTS")
        print("=" * 55)
        print(f"  Mask mAP50:     {metrics['mask_mAP50']:.4f}")
        print(f"  Mask mAP50-95:  {metrics['mask_mAP50_95']:.4f}")
        print(f"  Box mAP50:      {metrics['box_mAP50']:.4f}")
        print(f"  Box mAP50-95:   {metrics['box_mAP50_95']:.4f}")
        print("=" * 55)

        # Save report
        out_rep_dir = Path(self.cfg.output_dir) / self.cfg.experiment_name
        out_rep_dir.mkdir(parents=True, exist_ok=True)
        report_file = out_rep_dir / "evaluation_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# Evaluation Report: {self.cfg.experiment_name}\n\n")
            f.write(f"- Checkpoint: `{ckpt_path}`\n")
            f.write(f"- Dataset: `{self.cfg.dataset.name}` (Split: `val`)\n")
            f.write(f"- Mask mAP50: **{metrics['mask_mAP50']:.4f}**\n")
            f.write(f"- Mask mAP50-95: **{metrics['mask_mAP50_95']:.4f}**\n")
            f.write(f"- Box mAP50: **{metrics['box_mAP50']:.4f}**\n")
            f.write(f"- Box mAP50-95: **{metrics['box_mAP50_95']:.4f}**\n\n")
            f.write("## Per-Class AP50\n\n| Class | AP50 |\n|---|---:|\n")
            for cname, ap in metrics.get("per_class", {}).items():
                f.write(f"| {cname} | {ap:.4f} |\n")
        print(f"[*] Saved evaluation report -> {report_file}")

    # -------------------------------------------------------------
    # Option A: Run Complete End-to-End Workflow
    # -------------------------------------------------------------
    def run_auto_end_to_end(self) -> None:
        print("\n" + "=" * 75)
        print(" STARTING COMPLETE END-TO-END ADAPTIVE DISTILLATION WORKFLOW")
        print("=" * 75)
        # Step 5: Cache compatibility
        rep = self.step_check_cache()
        if not rep.is_fully_compatible:
            # Step 6: Prepare missing cache
            ok = self.step_prepare_cache()
            if not ok:
                print("[ERROR] Cache preparation failed. Halting workflow.")
                return
        # Step 7: Reliability validation
        self.step_validate_reliability()
        # Step 9: Start training
        self.step_start_training()

    # -------------------------------------------------------------
    # Option B: Run Baseline E2 (Ground-Truth-Only Scratch Student)
    # -------------------------------------------------------------
    def run_baseline_e2(self) -> None:
        print("\n" + "=" * 75)
        print(" [BASELINE E2] LAUNCH GROUND-TRUTH-ONLY SCRATCH STUDENT TRAINING")
        print("=" * 75)
        print("Experiment Profile:")
        print("  * Experiment Name:   e2_gt_only_scratch")
        print("  * Distillation:      DISABLED (L_kd = 0.0, 0 teachers loaded)")
        print("  * Loss Objective:    L_total = L_GT")
        print(f"  * Student Arch:      {self.cfg.student.arch} (Strict scratch, weights=None)")
        print(f"  * Dataset:           {self.cfg.dataset.name} ({self.cfg.dataset.canonical_taxonomy})")
        print(f"  * Train Images:      5,212 images (train_5212.json)")
        print(f"  * Validation Images: 652 images (val_652.json)")
        print(f"  * Benchmark Test:    EVA/GROUND (110 images) [Preserved Untouched]")
        print("-" * 75)

        epochs_val = int(prompt_default("Number of Epochs", self.cfg.student.epochs or 50))
        batch_val = int(prompt_default("Batch Size", self.cfg.student.batch_size or 8))

        self.cfg.experiment_name = "e2_gt_only_scratch"
        self.cfg.distill.enabled = False
        self.cfg.weights.mode = "fixed"
        self.cfg.student.epochs = epochs_val
        self.cfg.student.batch_size = batch_val
        self.save_config()

        print(f"\n[OK] Configuration saved. Starting Baseline E2 training ({epochs_val} epochs, batch {batch_val})...")
        self.step_start_training()

    def display_main_menu(self) -> None:
        print("\n" + "=" * 75)
        print(" GROUND-TRUTH-GUIDED ADAPTIVE MULTI-TEACHER DISTILLATION SYSTEM")
        print("=" * 75)
        print(f" Current Experiment: {self.cfg.experiment_name} | Mode: {self.cfg.weights.mode}")
        print(f" Teachers:           {[t.name for t in self.cfg.teachers]}")
        print(f" Dataset:            {self.cfg.dataset.name} ({self.cfg.dataset.canonical_taxonomy})")
        print(f" Student:            {self.cfg.student.arch} (scratch)")
        print("-" * 75)
        print("  1)  Select Teachers                -- Choose teacher models from runs_comparison/")
        print("  2)  Select Dataset                 -- Select dataset, split, taxonomy & annotations")
        print("  3)  Configure Student C            -- Choose scratch student architecture")
        print("  4)  Configure Training             -- Set epochs, batch size, mode, lr, temp")
        print("  5)  Check Cache Compatibility      -- Validate existing cache against current config")
        print("  6)  Prepare Missing Cache          -- Incrementally build missing/changed predictions")
        print("  7)  Validate Teacher Reliability   -- View exploratory calibration scores & CI")
        print("  8)  Initialize Student C           -- Verify scratch initialization & parameters")
        print("  9)  Start Student C Training       -- Train student with offline cache guidance")
        print("  10) Evaluate on Validation Split   -- Evaluate checkpoints & generate markdown report")
        print("  B)  Run Baseline E2 (GT-Only)      -- Pure ground-truth scratch student training (No teachers)")
        print("  A)  Run All End-to-End             -- Automated check -> prepare -> train -> evaluate")
        print("  0)  Exit")
        print("=" * 75)

    def run_interactive(self) -> None:
        while True:
            self.display_main_menu()
            choice = prompt("Select an option [0-10, A, B]: ").strip()
            if choice == "0":
                print("Exiting.")
                break
            elif choice == "1":
                self.step_select_teachers()
            elif choice == "2":
                self.step_select_dataset()
            elif choice == "3":
                self.step_configure_student()
            elif choice == "4":
                self.step_configure_training()
            elif choice == "5":
                self.step_check_cache()
            elif choice == "6":
                self.step_prepare_cache()
            elif choice == "7":
                self.step_validate_reliability()
            elif choice == "8":
                self.step_initialize_student()
            elif choice == "9":
                self.step_start_training()
            elif choice == "10":
                self.step_evaluate()
            elif choice.upper() == "B":
                self.run_baseline_e2()
            elif choice.upper() == "A":
                self.run_auto_end_to_end()
            else:
                print(f"[!] Unknown option: {choice}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Multi-Teacher Distillation Training Workflow")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML")
    parser.add_argument("--auto", action="store_true", help="Non-interactive end-to-end execution")
    parser.add_argument("--check_cache", action="store_true", help="Only run cache compatibility check and exit")
    parser.add_argument("--prepare_cache", action="store_true", help="Only run incremental cache preparation and exit")
    parser.add_argument("--mode", choices=["adaptive", "fixed", "single", "gt_only"], default=None, help="Override distillation mode")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--teachers", nargs="+", default=None, help="Filter to specific teacher names (e.g. teacher_a_m2f)")

    args = parser.parse_args()
    workflow = DistillationWorkflow(args.config)

    # CLI Overrides
    if args.teachers:
        t_filter = [x.lower() for x in args.teachers]
        filtered = [t for t in workflow.cfg.teachers if t.name.lower() in t_filter or any(f in t.name.lower() for f in t_filter)]
        if filtered:
            workflow.cfg.teachers = filtered
            print(f"[*] Filtered workflow teachers to: {[t.name for t in workflow.cfg.teachers]}")
    if args.mode:
        if args.mode == "gt_only":
            workflow.cfg.distill.enabled = False
            workflow.cfg.weights.mode = "fixed"
        else:
            workflow.cfg.distill.enabled = True
            workflow.cfg.weights.mode = args.mode
    if args.epochs:
        workflow.cfg.student.epochs = args.epochs
    if args.batch_size:
        workflow.cfg.student.batch_size = args.batch_size
    if args.exp_name:
        workflow.cfg.experiment_name = args.exp_name

    workflow.save_config()

    if args.check_cache:
        rep = workflow.step_check_cache()
        sys.exit(0 if rep.is_fully_compatible else 1)
    elif args.prepare_cache:
        ok = workflow.step_prepare_cache()
        sys.exit(0 if ok else 1)
    elif args.auto:
        workflow.run_auto_end_to_end()
    else:
        workflow.run_interactive()


if __name__ == "__main__":
    main()

