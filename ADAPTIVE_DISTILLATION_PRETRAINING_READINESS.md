# Adaptive Multi-Teacher Distillation: Pre-Training Readiness Report

**Document Name:** `ADAPTIVE_DISTILLATION_PRETRAINING_READINESS.md`  
**Date & Time:** 2026-11-09T09:40:00Z  
**Hardware Verified:** NVIDIA A10-12Q vGPU (11.81 GB VRAM, Driver 570.211.01, CUDA 12.8), 18 CPU Cores, 216 GB System RAM  
**Dataset Verified:** `/home/orchvate/CAR_TRAIN1/RAW_DATASET2` (6,516 images, 32 classes)  
**Teachers Verified:**
- **Teacher A:** `run_20261031_150325` — Mask2Former Swin-Tiny (32 classes, Best Val Mask mAP50: **0.7972**)
- **Teacher B:** `run_20261022_005324` — Mask2Former Swin-Tiny (32 classes, Best Val Mask mAP50: **0.7577**)
**Student C Verified:** Mask R-CNN (`maskrcnn_resnet50_fpn_v2`), strictly initialized from scratch (`weights=None`, 33 classes)

---

## 1. Executive Summary & Readiness Scorecard

All five requested pre-training corrections have been implemented, tested on GPU, and verified against regressions.

| # | Task / Area | Previous Audit | Readiness Status | Empirical Verification Summary |
|---|---|:---:|:---:|---|
| 1 | **Persisted Dataset Split** | PARTIAL | **PASS** | 5,212 train / 652 calib / 652 val persisted to JSON; 0 overlap; EVA/GROUND provenance documented. |
| 2 | **KD Instance-Weight Scaling** | PARTIAL | **PASS** | Scaled normalized KD loss by mean instance weight $\bar{\omega}$; verified across $[0.0, 1.0]$; $R_{\text{KD/GT}}$ logged. |
| 3 | **Configuration Lock** | PARTIAL | **PASS** | `default.yaml` locked to 32 classes, Mask R-CNN scratch, Teachers A & B, batch size 2. |
| 4 | **Experiment Evaluation** | PARTIAL | **PASS** | E0–E6 and C1–C5 execution paths verified; teacher evaluation path supported; COCO category indexing fixed. |
| 5 | **Final Smoke Test** | PARTIAL | **PASS** | 6 sub-tests passed on GPU; GT-only ($R_{\text{KD/GT}}=0.0$) and Adaptive KD ($R_{\text{KD/GT}}\approx 0.44$) verified. |
| 6 | **Automated Tests** | PASS | **PASS** | 22/22 unit tests passing in 11.72s; CLI contracts verified intact. |

---

## 2. Detailed Task Verification & Evidence

### Task 1: Persist the Dataset Split & Document Provenance
- **Status:** **PASS**
- **Action Taken:**
  Divided the 6,516 images of `/home/orchvate/CAR_TRAIN1/RAW_DATASET2/annotations/instances_unified.json` into three mutually exclusive, reproducible COCO format files:
  1. `train_5212.json`: 5,212 training images, 24,771 annotations (for offline cache generation & student training).
  2. `calib_652.json`: 652 calibration images, 3,011 annotations (independent split for Phase A teacher reliability $r[k, c]$).
  3. `val_652.json`: 652 validation images, 3,018 annotations (for Student C checkpoint selection and early stopping).
- **Mutual Exclusivity & Coverage Verification:**
  ```text
  Total images in instances_unified.json: 6,516
  Train images: 5,212 | Calib images: 652 | Val images: 652
  Overlap Train & Calib: 0
  Overlap Train & Val:   0
  Overlap Calib & Val:   0
  Union size: 6,516 (Matches total: True)
  ```
- **Metadata Persistence:**
  Created `/home/orchvate/CAR_TRAIN1/RAW_DATASET2/annotations/split_metadata.json` containing split parameters (random seed 42), SHA-256 hashes for all three split files, image ID manifests, and annotation counts.
- **`EVA/GROUND` Provenance Verification:**
  - File: `/home/orchvate/CAR_TRAIN1/EVA/GROUND/annotations.xml` and `coco_annotations.json` (110 images, 2,774 annotations, 32 classes).
  - Provenance: Exported from CVAT annotation task *"Vishnu Mukherjee"* (Task created 2026-06-14, updated 2026-08-07, dumped 2026-09-13).
  - Verified: **0 filename and 0 image hash overlap** with `RAW_DATASET2`.
  - Policy: Strictly preserved as the final untouched evaluation benchmark. No training, validation, or hyperparameter tuning touches `EVA/GROUND`.

---

### Task 2: KD Instance-Weight Scaling & Ratio Logging
- **Status:** **PASS**
- **Problem Identified in Audit:**
  In `losses.py`, `raw_kd_loss` was normalized by the number of active images, but was not scaled by the teacher confidence/quality weights $\omega$, causing teacher guidance to contribute with full pseudo-label weight regardless of trust.
- **Mathematical Correction Implemented:**
  1. Compute the mean instance weight:
     $$\bar{\omega} = \frac{\text{effective\_mass}}{\max(N_{\text{distilled}}, 1)} \in [0, 1.0]$$
  2. Scale the normalized KD loss exactly once:
     $$\mathcal{L}_{\text{KD}} = \left( \frac{\mathcal{L}_{\text{raw}}}{\max(N_{\text{active\_images}}, 1)} \right) \times \bar{\omega}$$
  3. Log the distillation-to-supervision ratio:
     $$R_{\text{KD/GT}} = \frac{\lambda(t) \cdot \mathcal{L}_{\text{KD}}}{\mathcal{L}_{\text{GT}} + \epsilon}$$
- **Empirical Weight Ladder Verification (on real GPU):**

| Instance Weight ($\omega$) | Status | Total Loss | GT Loss ($\mathcal{L}_{\text{GT}}$) | KD Loss ($\mathcal{L}_{\text{KD}}$) | Eff Mass ($\sum \omega$) | $R_{\text{KD/GT}}$ | Mean Grad Norm |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.00** (Vetoed) | `FP_class` | 5.7860 | 5.7860 | **0.0000** | 0.00 | **0.0000** | 3.0380 |
| **0.15** (Low Trust) | `TP` | 6.6495 | 5.7840 | **0.8655** | 0.15 | **0.1496** | 3.2719 |
| **0.50** (Medium Trust) | `TP` | 8.6665 | 5.7841 | **2.8824** | 0.50 | **0.4983** | 4.0744 |
| **1.00** (Full Trust) | `TP` | 11.5566 | 5.7855 | **5.7711** | 1.00 | **0.9975** | 5.7237 |

- **Observations:**
  - Vetoed instances ($\omega=0.0$) produce strictly 0.0 KD loss and zero distortion to supervised gradients.
  - KD loss scales linearly with teacher trust.
  - $R_{\text{KD/GT}}$ scales smoothly from 0.00 to 0.99 without numerical instability or gradient explosion.

---

### Task 3: Lock the Initial Experiment Configuration
- **Status:** **PASS**
- **Action Taken:**
  Updated `pipelines/adaptive_distillation/configs/default.yaml` and `config.py` to freeze the initial configuration:
  - **Canonical Taxonomy:** `32_class` (32 car parts, matching both teachers and `RAW_DATASET2`).
  - **Student Model:** `maskrcnn` (`maskrcnn_resnet50_fpn_v2`), scratch initialized (`weights=None`).
  - **Teachers:**
    - Teacher A: `run_20261031_150325` (`mask2former`, mAP50: 0.7972)
    - Teacher B: `run_20261022_005324` (`mask2former`, mAP50: 0.7577)
  - **Batch Size:** `2` (empirically verified to consume 7.10 GB VRAM, providing 4.7 GB headroom on 12GB GPU).
  - **Training Budget:** 50 epochs, AdamW optimizer (`lr0=0.005`, `weight_decay=0.0005`), AMP enabled.
  - **Existing Repositories:** Untouched.

---

### Task 4: Complete Experiment Evaluation
- **Status:** **PASS**
- **Action Taken:**
  1. Updated `evaluate_experiments.py` to support evaluating both teacher directories (Mask2Former) and student checkpoint files (`.pt`).
  2. Fixed 1-based COCO category ID indexing:
     Mask R-CNN foreground classes (1..32) and COCO category IDs (1..32) now align 1:1, preventing category index offsets during AP computation.
  3. Verified E0 evaluation path on validation split:
     Ran standalone evaluation on Teacher A (`run_20261031_150325`) with `evaluate_experiments.py --checkpoint ... --split val --exp_id E0 --max_images 20`. Inference executed cleanly in 3.1 seconds, generating standard COCO bbox and segm metrics.
  4. Verified uniform evaluation protocol across E0–E6 and C1–C5.

---

### Task 5: Final Smoke Test
- **Status:** **PASS**
- **Test Executed:** `pipelines/adaptive_distillation/scripts/pretraining_smoke_test.py` on NVIDIA A10 GPU.
- **Results across 6 Sub-Tests:**
  1. *Persisted Splits Verification:* Confirmed 5,212 train, 652 calib, 652 val images with zero overlap.
  2. *Offline Cache Preparation & Manifest Validation:* Built 50-image cache for Teacher A and Teacher B, unloaded teachers to 8.1 MB VRAM, validated manifest key `0f1d3f4d7fc19918...`.
  3. *Student Scratch Initialization:* Built Mask R-CNN from scratch; verified 46,047,253 / 46,047,253 trainable parameters.
  4. *GT-Only Training (3 real batches, $\lambda=0.0$):*
     - Batch 1: Total Loss = 5.4133 | GT Loss = 5.4133 | KD Loss = 0.0000 | $R_{\text{KD/GT}} = 0.0000$ | Peak VRAM = 3,668.7 MB | Latency = 0.715s
     - Batch 2: Total Loss = 4.8646 | GT Loss = 4.8646 | KD Loss = 0.0000 | $R_{\text{KD/GT}} = 0.0000$ | Peak VRAM = 3,841.8 MB | Latency = 0.431s
     - Batch 3: Total Loss = 3.6144 | GT Loss = 3.6144 | KD Loss = 0.0000 | $R_{\text{KD/GT}} = 0.0000$ | Peak VRAM = 3,842.7 MB | Latency = 0.180s
  5. *Adaptive Multi-Teacher KD Training (3 real batches, $\lambda=1.0$):*
     - Batch 1: Total Loss = 2.8290 | GT Loss = 1.9681 | KD Loss = 0.8610 | Eff Mass = 12.28 | **$R_{\text{KD/GT}} = 0.4375$** | Peak VRAM = 7,088.1 MB | Latency = 0.725s
     - Batch 2: Total Loss = 3.3754 | GT Loss = 2.3772 | KD Loss = 0.9982 | Eff Mass = 12.72 | **$R_{\text{KD/GT}} = 0.4199$** | Peak VRAM = 7,090.5 MB | Latency = 0.787s
     - Batch 3: Total Loss = 3.8990 | GT Loss = 2.6643 | KD Loss = 1.2347 | Eff Mass = 24.06 | **$R_{\text{KD/GT}} = 0.4634$** | Peak VRAM = 7,100.5 MB | Latency = 0.754s
  6. *Teacher Isolation Check:* Confirmed exactly 225 parameter tensors updated (100% Student C). Zero teacher parameters loaded or updated.

---

## 3. Ratio Analysis: $R_{\text{KD/GT}}$ Before vs After Correction

Prior to the mean instance-weight scaling fix, the single-step KD loss was 22.14 vs GT loss 3.98 ($R_{\text{KD/GT}} \approx 5.56$), meaning teacher predictions were exerting over 5 times the gradient magnitude of ground truth annotations.

With the mean instance-weight scaling fix in place:
$$R_{\text{KD/GT}} \in [0.42, 0.46]$$

This establishes the desired relationship:
- Supervised ground truth annotations ($\mathcal{L}_{\text{GT}}$) remain the primary anchor signal.
- Multi-teacher distillation ($\mathcal{L}_{\text{KD}}$) provides calibrated, non-overwhelming regularization.

---

## 4. Execution Plan (Awaiting User Approval)

The pipeline is in a fully validated, stable state. Full training has **NOT** been started.

When you approve proceeding to experimental execution, the sequence will run as follows:

```bash
# Step 1: Generate Offline 32-Class Teacher Cache (Phase A: ~35-40 min)
python3 pipelines/adaptive_distillation/scripts/prepare_teacher_cache.py \
  --config pipelines/adaptive_distillation/configs/default.yaml

# Step 2: Run Baseline Experiment E2 (GT-Only Scratch Student C)
python3 pipelines/adaptive_distillation/scripts/train_student.py \
  --no_distill \
  --exp_name e2_gt_only_scratch

# Step 3: Run Fixed Multi-Teacher KD Experiment E5 (Uniform 50/50)
python3 pipelines/adaptive_distillation/scripts/train_student.py \
  --mode fixed \
  --exp_name e5_fixed_multiteacher

# Step 4: Run Adaptive Multi-Teacher KD Experiment E6
python3 pipelines/adaptive_distillation/scripts/train_student.py \
  --mode adaptive \
  --exp_name e6_adaptive_multiteacher

# Step 5: Evaluate on EVA/GROUND Benchmark and Generate Final Comparison Report
python3 pipelines/adaptive_distillation/scripts/evaluate_experiments.py \
  --split test \
  --exp_id E6
```

*All pre-training readiness criteria are satisfied. Awaiting your approval to begin Phase A cache preparation.*

