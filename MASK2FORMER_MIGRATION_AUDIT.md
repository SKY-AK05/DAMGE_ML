# Mask2Former Migration and Project Integrity Audit Report

**Date:** 2026-11-10  
**Branch:** `mask2former-distillation-migration`  
**Host Environment:** Local Development Machine (Windows 11, Intel/AMD x86_64, 16 GB RAM, CPU-Only / PyTorch 2.7.0+cpu, Python 3.13.2)  
**Authoritative Training Environment:** Azure GPU VM (Ubuntu, NVIDIA Attached GPU, CUDA Enabled, Authoritative Dataset & Checkpoints)

---

## 1. Executive Summary

This audit assesses the state of the `CAR_TRAIN1` (repository `DAMGE_ML`) codebase following transfer from the Azure VM to the local development machine. The objective is to establish an end-to-end, **Mask2Former-only** adaptive multi-teacher knowledge distillation pipeline across Teacher A, Teacher B, and Student C, eliminating any Mask R-CNN dependencies from the active distillation flow, and preparing the system for experiments **E3**, **E4**, **E5**, and **E6**.

---

## 2. Repository Structure and Inventory

```
DAMGE_ML/
├── .git/                                # Git repository on branch mask2former-distillation-migration
├── carparts-seg/                        # Legacy dataset labels (YOLO format)
├── docs/                                # Technical specifications and historical project logs
├── EVA/                                 # Final benchmark directory
│   ├── GROUND/                          # SEALED: coco_annotations.json (110 evaluation images)
│   └── mask2former_eval/                # Mask2Former standalone evaluation scripts
├── pipelines/
│   └── adaptive_distillation/           # Primary distillation pipeline
│       ├── cache/                       # Cache building, reading, manifests, and compatibility
│       ├── configs/                     # default.yaml and experiment configuration definitions
│       ├── data/                        # Dataset adapter, collators, transforms
│       ├── evaluation/                  # Metrics calculation and statistical hypothesis testing
│       ├── matching/                    # Hungarian matching, trust weighting, teacher fusion
│       ├── scripts/                     # Cache prep, student training, evaluation CLI tools
│       ├── student/                     # Student builder, distillation trainer, losses, schedules
│       ├── teachers/                    # Teacher scanner, loader, inference
│       ├── tests/                       # Unit and integration test suite (22 tests)
│       └── utils/                       # Checkpointing, logging, reproducibility
├── RAW_DATASET2/
│   ├── annotations/                     # train_5212.json, calib_652.json, val_652.json (32 classes)
│   ├── images/                          # 6 sample images (code-only dev; 6,516 images reside on VM)
│   └── unified_categories.yaml          # Canonical 32-class taxonomy definition
├── reliability/
│   └── calibration_report.md            # Calibration report for Teacher A (0.8146) & Teacher B (0.7781)
├── scripts/
│   ├── training/                        # train_mask2former.py, azure_train.py, legacy train_maskrcnn.py
│   └── evaluation/                      # unified_evaluator.py, unified_logger.py
└── teacher_cache/                       # Persistent sharded teacher predictions
    ├── cache_manifest.json              # Schema 2.0 manifest (key: 89a823761a77...)
    ├── index_teacher_a_m2f.json         # Index for 5,212 training images (Teacher A)
    ├── index_teacher_b_m2f.json         # Index for 5,212 training images (Teacher B)
    ├── teacher_a_m2f/                   # 11 shards (shard_00000.npz – shard_00010.npz)
    └── teacher_b_m2f/                   # 11 shards (shard_00000.npz – shard_00010.npz)
```

---

## 3. Current Architecture vs. Required Architecture

| Component | Required Architecture | Current State | Audit Finding |
|---|---|---|---|
| **Teacher A** | Mask2Former (Swin-Tiny) | Mask2Former Swin-Tiny (`run_20261031_150325`, mAP50=0.7972) | **Compatible & Preserved** |
| **Teacher B** | Mask2Former (Swin-Tiny) | Mask2Former Swin-Tiny (`run_20261022_005324`, mAP50=0.7577) | **Compatible & Preserved** |
| **Student C** | Mask2Former (Swin-Tiny) initialized strictly from scratch | Mask2Former builder exists in `student_builder.py`; legacy docs/orchestrator refer to Mask R-CNN | **Migration Required**: Ensure entire active distillation pipeline (trainer, matcher, loss, configs) is strictly Mask2Former-native |
| **Taxonomy** | 32 Canonical Classes | 32 Classes (`RAW_DATASET2/unified_categories.yaml`) | **Compatible & Preserved** |
| **Distillation Matching** | Query-independent Hungarian matching (Teacher $\to$ Student/GT) | Class-agnostic Hungarian matcher exists in `instance_matcher.py`, but multi-teacher fusion was disconnected from batch streaming | **Critical Gap**: Connect multi-teacher fusion and student query matching properly |
| **Absence Supervision** | Teacher absence $\ne$ Background | `losses.py` passed pseudo-targets into standard HuggingFace loss, penalizing unassigned student queries as `no_object` | **Critical Defect**: Must isolate positive teacher distillation loss without negative background penalty on unpredicted queries |

---

## 4. Existing Working Components

1. **Teacher Cache System (`teacher_cache/`)**:
   - 11 `.npz` shards each for Teacher A and Teacher B covering all 5,212 training images are present and readable.
   - Cache schema 2.0 contains bounding boxes, canonical classes (0–31), confidences, reliability weights ($\omega$), match statuses (`TP`, `FP_loc`, etc.), RLE masks, and $1/4$-scale downscaled masks.
2. **Reliability Calibration (`reliability/calibration_report.md`)**:
   - Exploratory reliability scores computed on held-out 652 images:
     - Teacher A: $r_{\text{global}} = 0.8146$
     - Teacher B: $r_{\text{global}} = 0.7781$
   - Selection bias notice is appropriately documented (derived from the historical validation pool).
3. **Canonical Annotations (`RAW_DATASET2/annotations/`)**:
   - Split manifests `train_5212.json`, `calib_652.json`, and `val_652.json` exist locally and match the 32-class taxonomy.
4. **Scratch Mask2Former Student Builder (`student_builder.py`)**:
   - Correctly loads `Mask2FormerConfig` from `facebook/mask2former-swin-tiny-coco-instance` with random weight initialization (`Mask2FormerForUniversalSegmentation(config)` without `.from_pretrained()`).
   - Verified that all 47.4M parameters require gradient and have non-zero random weights.
5. **Baseline Test Suite (`pipelines/adaptive_distillation/tests/`)**:
   - 22/22 unit and integration tests currently pass on CPU.

---

## 5. Mask R-CNN Specific Code & Dependencies Audit

The following code elements relate to Mask R-CNN and must either be preserved as legacy without polluting the active pipeline or refactored:

1. **`pipelines/adaptive_distillation/teachers/teacher_loader.py`**:
   - Lines 108–125: Loading logic for `maskrcnn_resnet50_fpn_v2`.
   - *Status*: Remove or guard against selection in active distillation; distillation teachers are strictly Mask2Former Swin-Tiny.
2. **`pipelines/adaptive_distillation/teachers/teacher_scanner.py`**:
   - Mentions `maskrcnn` as a detectable architecture.
   - *Status*: Filter scanner to Mask2Former models for active experiments.
3. **`pipelines/adaptive_distillation/student/losses.py`**:
   - Comments and legacy 1-based class shift checks (`labels_t = labels_t - 1 if min >= 1`) from Mask R-CNN conventions.
   - *Status*: Standardize on explicit 0-indexed canonical classes (0..31) for Mask2Former.
4. **`scripts/training/train_maskrcnn.py` & `orchestrator.py`**:
   - Root project orchestrator contains historical training options for YOLO, Mask R-CNN, and Fast R-CNN.
   - *Status*: Do not break root orchestrator, but ensure distillation experiments E3–E6 run strictly through the Mask2Former adaptive distillation pipeline.

---

## 6. Identified Missing Assets (Local Development vs. Azure VM)

As specified in the transfer protocol, the local machine contains code and metadata only:
- **Missing Full Image Sets**: `RAW_DATASET2/images` contains 6 sample images. The remaining 6,510 images reside exclusively on the Azure VM.
- **Missing Model Weight Checkpoints**: Checkpoints for `run_20261031_150325` and `run_20261022_005324` reside on the Azure VM filesystem.
- **Hardware Absence**: Local machine has no CUDA GPU (`torch.cuda.is_available() == False`).
- **Sealed Benchmark**: `EVA/GROUND/coco_annotations.json` is intact and sealed; images reside on Azure VM.

---

## 7. Potential Compatibility & Architectural Problems

1. **Hardcoded Machine-Specific Absolute Paths**:
   - Files like `default.yaml` and `config.py` hardcoded paths such as `/home/orchvate/CAR_TRAIN1/...`.
   - *Remedy*: Introduce portable relative/environment-based path resolution (`CAR_PROJECT_ROOT` fallback to local repo root).
2. **HuggingFace Offline Execution / Network Dependency**:
   - `StudentBuilder.build()` called `Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")` over the network.
   - *Remedy*: Cache configuration locally or allow local config instantiation so tests and offline runs never fail due to HF Hub connectivity.
3. **Multi-Teacher Fusion Disconnection in Training Pipeline**:
   - `DistillationDataset` concatenated teacher instances across teachers instead of calling `TeacherFusion.fuse_image_instances()`.
   - In E5 and E6, duplicate teacher detections must be clustered and fused, solo detections penalized by $\kappa = 0.8$, and conflicting predictions arbitrated or pruned.
4. **Teacher Absence Background Penalty**:
   - Hugging Face's default `Mask2FormerLoss` penalizes unassigned student queries with `no_object` background loss.
   - In knowledge distillation, teacher absence must NOT be treated as negative supervision.
   - *Remedy*: Construct custom Mask2Former distillation criterion that calculates mask, dice, and class supervision only on matched teacher instances, zeroing out unassigned query penalties.

---

## 8. Proposed Migration Plan

1. **Phase 2 — Architecture Migration & Portability**:
   - Update `config.py` and experiment configurations to support portable paths across Windows local dev and Azure Linux VM.
   - Create dedicated YAML experiment configurations for **E3 (Single Teacher A)**, **E4 (Single Teacher B)**, **E5 (Fixed Multi-Teacher)**, and **E6 (Adaptive Multi-Teacher)**.
   - Ensure `StudentBuilder` can operate completely offline with fallback to local cached architecture configuration.
2. **Phase 3 — Mask2Former-Compatible Distillation Implementation**:
   - Implement teacher-to-student Hungarian instance matching without query-index assumptions.
   - Formulate teacher distillation loss with strict absence masking (no background penalties on unassigned queries).
   - Integrate `TeacherFusion` into `DistillationDataset` and `DistillationTrainer` for E5 (fixed equal weighting) and E6 (adaptive reliability weighting).
   - Enforce ground-truth quality veto and asymmetry rules.
3. **Phase 4 — Cache and Calibration Compatibility**:
   - Validate that existing Tier 2 `.npz` cache shards and manifests are 100% compatible with Student C Mask2Former input requirements.
   - Implement caching adapter if needed for fast spatial mask decompression.
4. **Phase 5 — Comprehensive CPU Testing Suite**:
   - Build automated tests using synthetic fixtures for:
     - Scratch initialization of Student C (Mask2Former Swin-Tiny).
     - Forward pass, GT loss, backward pass with gradient clipping.
     - Teacher cache reader and instance deserialization.
     - Hungarian matching between teacher targets and student predictions.
     - E3 / E4 single-teacher isolation.
     - E5 fixed multi-teacher fusion (WBF + probability averaging).
     - E6 adaptive fusion with reliability weights and GT veto.
     - Checkpoint save and resume round-trip.
5. **Phase 6 — Final Documentation, Git Review & Azure VM Readiness**:
   - Produce `MASK2FORMER_ARCHITECTURE.md`, `MASK2FORMER_DISTILLATION_DESIGN.md`, and Azure VM execution runbooks.

