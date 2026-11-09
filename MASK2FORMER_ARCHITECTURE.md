# Mask2Former Pure Multi-Teacher Distillation Architecture Specification

**Status:** APPROVED ARCHITECTURAL SPECIFICATION  
**Author:** Deep Learning Research & Systems Engineering  
**Target:** Mask2Former Swin-Tiny (Teachers A & B, Student C)  
**Taxonomy:** 32 Canonical Classes (`RAW_DATASET2`)  

---

## 1. System Overview

The objective is to train a high-accuracy, scratch-initialized **Student C** model using the **Mask2Former Swin-Tiny** architecture, supervised simultaneously by verified **Ground Truth (GT)** labels and knowledge distilled from two existing, independently trained Mask2Former Swin-Tiny teachers:
- **Teacher A (`run_20261031_150325`)**: Validation Mask mAP50 = 0.7972, Exploratory Reliability $r_{\text{global}} = 0.8146$.
- **Teacher B (`run_20261022_005324`)**: Validation Mask mAP50 = 0.7577, Exploratory Reliability $r_{\text{global}} = 0.7781$.

```
+---------------------------------------------------------------------------------------+
|                                    OFFLINE PHASE A                                    |
|                                                                                       |
|   +------------------------------------+     +------------------------------------+   |
|   |             TEACHER A              |     |             TEACHER B              |   |
|   |        Mask2Former Swin-Tiny       |     |        Mask2Former Swin-Tiny       |   |
|   |         (run_20261031_150325)      |     |         (run_20261022_005324)      |   |
|   +-----------------+------------------+     +-----------------+------------------+   |
|                     |                                          |                      |
|                     v                                          v                      |
|           [ Hungarian GT Match ]                     [ Hungarian GT Match ]           |
|           [ Reliability Gating ]                     [ Reliability Gating ]           |
|                     |                                          |                      |
|                     v                                          v                      |
|   +-----------------+------------------+     +-----------------+------------------+   |
|   |      Shard Cache: teacher_a_m2f    |     |      Shard Cache: teacher_b_m2f    |   |
|   |         11 shards / 5,212 imgs     |     |         11 shards / 5,212 imgs     |   |
|   +-----------------+------------------+     +-----------------+------------------+   |
+---------------------|------------------------------------------|----------------------+
                      |                                          |
======================|==========================================|=======================
                      |                                          |
+---------------------v------------------------------------------v----------------------+
|                           ONLINE TRAINING PHASE B (ZERO TEACHER VRAM)                 |
|                                                                                       |
|   +-------------------------------------------------------------------------------+   |
|   |                    Multi-Teacher Fusion & Arbitration Engine                  |   |
|   |           - Mask IoU Clustering (tau_cluster = 0.60)                          |   |
|   |           - Weighted Box Fusion & Mask Probability Merging                    |   |
|   |           - GT Quality Veto & Solo Detection Penalty (kappa = 0.80)           |   |
|   |           - Conflict Arbitration (GT Agreement or Disagreement Margin >= 2.0) |   |
|   +---------------------------------------+---------------------------------------+   |
|                                           | Fused Pseudo-Targets                      |
|                                           v                                           |
|       +-----------------------------------+-----------------------------------+       |
|       |                                                                       |       |
|       |                    STUDENT C (Mask2Former Swin-Tiny)                  |       |
|       |                         SCRATCH INITIALIZATION                        |       |
|       |                                                                       |       |
|       |  Input: Preprocessed Pixel Values (B, 3, 640, 640)                    |       |
|       |  Backbone: Swin-Tiny (Randomly Initialized, 4 stages)                 |       |
|       |  Pixel Decoder: Multi-scale Deformable Attention (Random)             |       |
|       |  Transformer Decoder: 9 Masked-Attention Layers, 100 Queries (Random) |       |
|       |  Heads: Classification (32 classes) + Mask MLP (Random)               |       |
|       +-----------------------------------+-----------------------------------+       |
|                                           | Query Predictions                         |
|                                           v                                           |
|                       +-------------------+-------------------+                       |
|                       |  Hungarian Matcher (Instance-Level)   |                       |
|                       +---------+-------------------+---------+                       |
|                                 |                   |                                 |
|                                 v                   v                                 |
|                         [ Supervised GT Loss ] [ Distillation KD Loss ]               |
|                         - Cross-Entropy        - Matched CE on Teacher Class          |
|                         - Focal Mask Loss      - Matched BCE on Teacher Mask          |
|                         - Dice Loss            - Matched Dice Loss                    |
|                                                - Weighted by omega                    |
|                                                - NO UNASSIGNED BACKGROUND PENALTY     |
|                                 \                   /                                 |
|                                  \                 /                                  |
|                                   v               v                                   |
|                          Total Loss: L_tot = L_GT + lambda(t) * L_KD                  |
+---------------------------------------------------------------------------------------+
```

---

## 2. Model Architecture Compatibility Matrix

To guarantee strict parity and prevent structural divergence:

| Property | Teacher A | Teacher B | Student C | Parity Status |
|---|---|---|---|---|
| **Base Model Architecture** | Mask2Former | Mask2Former | Mask2Former | **Identical** |
| **Backbone** | Swin-Tiny (`swin_tiny_patch4_window7_224`) | Swin-Tiny | Swin-Tiny | **Identical** |
| **Backbone Embedding Dim** | 96 | 96 | 96 | **Identical** |
| **Backbone Depths** | `[2, 2, 6, 2]` | `[2, 2, 6, 2]` | `[2, 2, 6, 2]` | **Identical** |
| **Backbone Num Heads** | `[3, 6, 12, 24]` | `[3, 6, 12, 24]` | `[3, 6, 12, 24]` | **Identical** |
| **Pixel Decoder** | MSDeformAttn | MSDeformAttn | MSDeformAttn | **Identical** |
| **Decoder Hidden Dims** | 256 | 256 | 256 | **Identical** |
| **Number of Object Queries** | 100 | 100 | 100 | **Identical** |
| **Decoder Layers** | 9 | 9 | 9 | **Identical** |
| **Number of Classes** | 32 foreground classes | 32 foreground classes | 32 foreground classes | **Identical** |
| **No-Object / Background Index** | Class index 32 (internally handled) | Class index 32 | Class index 32 | **Identical** |
| **Input Resolution** | $640 \times 640$ | $640 \times 640$ | $640 \times 640$ | **Identical** |
| **Mask Output Resolution** | $160 \times 160$ ($1/4$ resolution) | $160 \times 160$ | $160 \times 160$ | **Identical** |
| **Normalization** | ImageNet Mean/Std | ImageNet Mean/Std | ImageNet Mean/Std | **Identical** |
| **Weight Provenance** | Pretrained $\to$ Fine-tuned | Pretrained $\to$ Fine-tuned | **Strict Scratch (Random)** | **Verified** |

---

## 3. Strict Scratch Initialization Protocol

Student C must NOT inherit weights from Teacher A, Teacher B, ImageNet, or COCO.

### 3.1 Mechanism
```python
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

# 1. Load configuration specification only (NO weights loaded)
config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
config.num_labels = 32
config.id2label = {i: name for i, name in enumerate(class_names)}
config.label2id = {name: i for i, name in enumerate(class_names)}

# 2. Instantiate model directly from config
# PyTorch initializes all layers (Swin backbone, MSDeformAttn, Transformer decoder, heads)
# with random Gaussian / truncated normal distributions under a deterministic seed.
torch.manual_seed(seed)
student_model = Mask2FormerForUniversalSegmentation(config)
```

### 3.2 Scratch Verification Gate
Before training or evaluation:
1. `sum(p.numel() for p in student.parameters() if p.requires_grad) == 47,440,784`.
2. Backbone weights have non-zero variance ($\sigma > 0$) and $\max(|W|) > 0$.
3. Checkpoint state dictionary contains 0 keys from teacher checkpoints.
4. Classification head, pixel decoder, and Swin patch embedding layers all exhibit random weight initialization statistics.

---

## 4. Query Permutation Invariance & Bipartite Matching

Because DETR and Mask2Former transformer decoder queries are permutation-invariant:
- Query index $i$ in Teacher A has **no semantic correspondence** to query index $i$ in Teacher B or Student C.
- Query order varies dynamically across training iterations and between models.
- **Direct element-wise loss across query indices $i$ is strictly forbidden.**
- All teacher-to-student and teacher-to-GT knowledge transfer is conducted in **instance coordinate space** via **bipartite Hungarian matching** on bounding boxes and mask IoU.

