# GPU Readiness and Hardware Environment Report

**Date:** 2026-11-10  
**Project:** Mask2Former Distillation Migration (`DAMGE_ML` / `CAR_TRAIN1`)  
**Prepared For:** Azure VM Execution of Experiments E3–E6  

---

## 1. Local Development Machine Assessment

| Metric | Measured Value | Minimum Training Requirement | Assessment |
|---|---|---|---|
| **Operating System** | Microsoft Windows 11 Home / Pro | Ubuntu 20.04 / 22.04 LTS (Azure VM) | Development & Testing Only |
| **Python Version** | 3.13.2 (64-bit AMD64) | 3.10 – 3.12 Recommended | Fully Compatible |
| **PyTorch Version** | 2.7.0+cpu | 2.1+ with CUDA 11.8 or 12.1+ | Local: CPU Only; Azure VM: CUDA Enabled |
| **Transformers Version** | 5.10.2 | 4.30+ | Fully Compatible |
| **Torchvision Version** | 0.22.0 | 0.16+ | Fully Compatible |
| **Local CUDA / GPU** | **None (CPU Only, 0 GPUs)** | **NVIDIA Attached GPU (>= 12 GB VRAM)** | **Training Blocked Locally; Must Run on Azure VM** |
| **System RAM** | 16,409,232 KB (~16.0 GB, ~1.4 GB free) | 16 GB System Memory | Sufficient for CPU Unit Tests |
| **Disk Capacity (C:)** | ~11.9 GB Free | >= 30 GB SSD for full dataset & cache | Sufficient for Source Code Only |

> [!CAUTION] **Strict Local Training Prohibition**  
> Because the local workstation lacks an NVIDIA CUDA GPU and contains only a code-only snapshot (without the 6,510 training images or teacher checkpoint weights), **full training must NEVER be launched locally**. All 52 automated tests executed locally utilize lightweight synthetic fixtures.

---

## 2. Authoritative Azure VM Training Environment

The Azure VM holds the true authoritative environment:
- **VM Hardware**: Standard_NV12ads_A10_v5 or equivalent NVIDIA A10 / V100 GPU (12GB–24GB VRAM).
- **Authoritative Dataset**: `RAW_DATASET2` (6,516 images across `train_5212.json`, `calib_652.json`, `val_652.json`).
- **Authoritative Checkpoints**:
  - Teacher A: `runs_comparison/run_20261031_150325/mask2former/best_model`
  - Teacher B: `runs_comparison/run_20261022_005324/mask2former/best_model`
- **Authoritative Cache**: `teacher_cache/` (11 shards per teacher, validated Schema 2.0).
- **Sealed Benchmark**: `EVA/GROUND/coco_annotations.json` (110 images).

---

## 3. VRAM and Compute Budget Breakdown (Azure GPU Training)

### 3.1 Memory Isolation (The Zero-Teacher-VRAM Contract)
In conventional multi-teacher distillation, keeping Teacher A (Swin-Tiny, 47.4M params), Teacher B (47.4M params), and Student C (47.4M params) simultaneously in GPU memory would require $> 142\text{M}$ active model parameters plus dual inference activations, exceeding a 12GB VRAM boundary and causing Out-Of-Memory (OOM) crashes.

**Our Implemented Safeguard:**
- Teacher inference was executed 100% offline in Phase A.
- Teacher predictions (masks, classes, bounding boxes, reliability weights) are streamed directly from pre-sharded `.npz` files via memory-mapped IO.
- **Zero teacher parameters reside in GPU VRAM during Student C training.**
- GPU memory is dedicated 100% to Student C:
  - Model weights: ~190 MB (FP32) / ~95 MB (AMP FP16).
  - Optimizer states (AdamW 2 moments): ~380 MB.
  - Activation memory (Batch size 4, $640 \times 640$, 100 queries): ~4.2 GB.
  - Gradient accumulation buffer (Steps = 4): ~150 MB.
  - Total Estimated Training Footprint: **~5.5 GB VRAM** (comfortably within 12GB–24GB GPU budget).

---

## 4. Workload Tier Separation

```
+---------------------------------------------------------------------------------------+
| TIER 1: CPU-Compatible Unit Tests (RUNS LOCALLY & ON CI)                              |
| - Fast execution (< 15 seconds)                                                       |
| - Uses SyntheticM2FStudent and synthetic bounding boxes/masks                         |
| - Verifies: Student C scratch init, Hungarian bipartite matching, GT veto asymmetry, |
|   multi-teacher fusion, checkpoint saving/loading, manifest schema                    |
| - Command: python -m unittest pipelines/adaptive_distillation/tests/test_mask2former_migration.py |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| TIER 2: GPU Integration Smoke Tests (RUNS ON AZURE VM BEFORE TRAINING)                |
| - Executes 1 epoch with 10 real images on Azure GPU VM                                |
| - Verifies: CUDA memory allocation, AMP GradScaler stability, real cache decompression |
| - Command: python pipelines/adaptive_distillation/scripts/train_student.py \          |
|              --config pipelines/adaptive_distillation/configs/e6_adaptive_multiteacher.yaml \
|              --epochs 1 --batch_size 2                                                |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| TIER 3: Full Production GPU Distillation Experiments (AZURE VM ONLY)                  |
| - E3: Single Teacher A (50 epochs)                                                    |
| - E4: Single Teacher B (50 epochs)                                                    |
| - E5: Fixed Multi-Teacher (50 epochs)                                                 |
| - E6: Adaptive Multi-Teacher (50 epochs)                                              |
| - Execution via interactive orchestrator or direct CLI scripts                        |
+---------------------------------------------------------------------------------------+
```

---

## 5. Verification Checklist Before Azure VM Run

- [x] Student C is verified as Mask2Former Swin-Tiny.
- [x] All 47.4M parameters initialized randomly from scratch.
- [x] Teacher absence is masked out (zero gradient on unassigned queries).
- [x] Multi-teacher fusion engine is fully integrated into `DistillationDataset`.
- [x] Path portability resolution tested across Windows and Linux path conventions.
- [x] All 52 automated tests pass with 0 failures on CPU.
- [x] Zero large binaries, dataset images, or secret credentials staged in Git.

