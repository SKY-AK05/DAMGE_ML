# Ground-Truth-Guided Adaptive Multi-Teacher Distillation
## Full Design and Implementation Plan — v2.0

**Status:** Design proposal (upgraded, implementation-ready draft)
**Version:** 2.0
**Date:** 2026-11-08
**Supersedes:** v1.0 (2026-10-31)
**Primary goal:** Train a new student model (Student C) from scratch using verified ground-truth (GT) annotations plus selectively weighted knowledge from two existing trained models (Teacher A, Teacher B), with teacher outputs precomputed to disk so the training loop is cheap, bounded in memory, and reproducible.

---

## 0. Document Control

### 0.1 What changed from v1.0 and why

| # | v1.0 gap | v2.0 upgrade | Where |
|---|---|---|---|
| 1 | Weight rule `w = r / Σr` always sums to 1, so two poor teachers get the same total influence as two excellent ones | Split weighting into **relative share** (who to trust more) and an **absolute trust gate** (whether to distill at all), with a floor/ceiling | §8 |
| 2 | Per-example GT agreement is inflated when teachers were trained on the same images | Principle: **GT can veto a teacher, but agreement with GT never boosts it beyond its validation reliability** | §5, §7, §8 |
| 3 | Missed GT objects and "teacher says background" were not handled at the loss level | Explicit **ignore-mask policy**: teacher absence is never a negative target; unmatched teacher detections are quarantined, not copied | §9, §10 |
| 4 | Cache design assumed dense outputs, which are large and break under geometric augmentation | **Three cache tiers** (instance pseudo-labels → soft class distributions → dense logits). Tier 1/2 are augmentation-safe because they are geometry-transformable instances | §6, §11 |
| 5 | Weighting formulas were only sketched | Concrete formulas for shrinkage, calibration, matching, instance weights, fusion, and loss reduction | §7–§10 |
| 6 | Success criteria were qualitative | **Numeric go/no-go gates**, pre-registered hypotheses, and statistical testing rules | §2, §13, §16 |
| 7 | Controls were missing | Added **shuffled-weight, uniform-weight, and oracle-style controls** to prove the reliability signal itself matters | §13 |
| 8 | No handling of disagreement between teachers | **Teacher fusion and disagreement policy** with GT-arbitrated cases | §9 |
| 9 | No config, test list, or interfaces | Added config schema, interface contracts, unit/integration test list, and CLI layout | §15, App. C–D |
| 10 | Unlabeled data not considered | Optional **Phase C** (unlabeled-image distillation) where distillation is usually most valuable | §4, §17 |
| 11 | Static distillation strength | **Loss schedule** (warm-up and optional decay) as an ablation | §10 |
| 12 | Open questions listed with no consequence | Appendix B now maps each answer to the **design decision it changes** | App. B |

### 0.2 Assumptions made by this version

v1.0 refers to "Run 1/Run 2 checkpoints", "mask mAP50", and "segmentation", so v2.0 assumes **instance segmentation** with class labels, boxes, and masks as the default concrete example. Everything is written so that it degrades gracefully to detection (drop masks) or classification (treat each image as one "instance"). Anything that depends on an unanswered question from Appendix B is marked **[DECISION]**.

---

## 1. Executive Summary

Two already-trained models, **Teacher A** (e.g., Run 1) and **Teacher B** (e.g., Run 2), help train a new model, **Student C**, that starts from scratch.

The core idea is **not to trust either teacher unconditionally**. Both teachers see the same images. Their predictions are compared with verified GT, and the evidence is turned into reliability scores that set how much each teacher influences the student. Teacher outputs and reliability metadata are computed once, stored on disk, and streamed per batch during student training.

The student learns from two sources:

1. **Ground-truth supervision** (primary and always active).
2. **Teacher guidance**, scaled by measured reliability, and silenced where the evidence says it is unreliable.

This is **offline multi-teacher knowledge distillation with ground-truth-guided, validation-calibrated adaptive weighting.**

### 1.1 Five design principles (the "constitution")

1. **GT is the anchor.** The GT loss is never turned off and never down-weighted by teacher presence.
2. **GT can veto, never boost.** Where a teacher contradicts GT on a labeled example, its guidance for that object is removed. Where it agrees, its influence is capped by its *held-out* reliability, not by how well it fits images it may have memorized.
3. **Absence is not evidence.** A teacher not predicting an object is never converted into a "this is background" training target.
4. **Relative trust and absolute trust are different questions.** "A is better than B here" and "A is good enough to learn from here" are decided separately.
5. **Nothing is claimed without a controlled comparison.** Every component must beat its simpler ablation on held-out data, across seeds, or it is removed.

### 1.2 Why this design?

- Teachers can be strong on some classes or examples and weak on others.
- Running large teachers at every student step increases compute and complicates memory management.
- Precomputing separates teacher inference from student optimization.
- Explicit baselines and controls are required to learn whether teacher guidance helps or just adds complexity.

### 1.3 Important limitations (stated up front)

- Teacher agreement with GT on a **training** image is evidence about that image only. If the teacher was trained on it, the evidence is inflated by memorization. Use held-out validation data for general reliability and keep the test set untouched until final evaluation.
- Distillation does **not** guarantee a student better than its teachers, and may not beat the GT-only student at all. The plan includes the experiments that decide this honestly.
- Two teachers can share a blind spot; agreement is not proof of correctness.

---

## 2. Goals, Non-Goals, and Success Criteria

### 2.1 Goals

1. Keep Teacher A and Teacher B frozen during student training.
2. Have both teachers process exactly the same dataset examples and preprocessing as the student pipeline.
3. Compare teacher outputs against verified annotations where available.
4. Produce reliability weights that vary by class and, where justified, by output type and by individual instance.
5. Cache teacher outputs and metadata on disk before student training.
6. Train Student C from scratch with a loss combining GT supervision and weighted teacher guidance.
7. Never load the whole cache into RAM; read only the current mini-batch.
8. Evaluate with controlled baselines, controls, multiple seeds, and a fixed held-out test set.
9. Make the pipeline reproducible, auditable, and safe against stale or misaligned cache data.
10. **(New)** Make every adaptive component individually switchable so it can be ablated.

### 2.2 Non-goals for the first version

- Updating teacher parameters during student training.
- Requiring the student to exceed both teachers.
- Transferring arbitrary internal features across different architectures.
- Replacing GT with teacher predictions.
- Online mutual learning where all models change together.
- Claiming success from training loss or training accuracy.
- Using the test set for any decision.

### 2.3 Pre-registered hypotheses

Write these down **before** running experiments, so results cannot be rationalized afterwards.

| ID | Hypothesis | Evidence that supports it | Evidence that refutes it |
|---|---|---|---|
| H1 | Distillation from a single teacher improves the scratch student over GT-only | E3 or E4 > E2 on val and test, beyond seed noise | E3/E4 ≤ E2 |
| H2 | Using both teachers beats either alone | E5 > max(E3, E4) | E5 ≤ max(E3, E4) |
| H3 | Adaptive weights beat fixed weights | E6 > E5 | E6 ≤ E5 |
| H4 | The **reliability signal itself** is what helps, not just extra regularization | E6 > E6-shuffled and E6 > E6-uniform | E6 ≈ shuffled/uniform |
| H5 | Gains are concentrated in classes where one teacher is much stronger | Per-class gain correlates with teacher reliability gap | No such correlation |

### 2.4 Numeric go/no-go gates

Fill in thresholds **[DECISION]** after measuring GT-only seed variance in Milestone 3.

| Gate | Rule |
|---|---|
| Method adopted | E6 beats E2 by more than `max(2 × seed_std, δ_min)` on the primary metric, on validation **and** test, with a paired-bootstrap 95% CI excluding 0 |
| Adaptive weighting kept | E6 beats E5 by the same margin; otherwise ship fixed weights (simpler) |
| No harmful regressions | No class with ≥ `n_min` test instances loses more than `δ_class` vs. E2 (suggested: 3 points AP50) |
| Cost acceptable | Total extra wall-clock (cache build + training overhead) ≤ budget agreed in Milestone 1 |

### 2.5 Primary and secondary metrics

For segmentation: mask mAP50 and mask mAP50–95 (primary), per-class AP/IoU, precision, recall, and rare/difficult-class performance. **Choose one primary metric before experiments and never change it afterward.** Use the same evaluation implementation, thresholds, NMS settings, and confidence cut-offs for every run.

---

## 3. Terminology

| Term | Meaning | Why it matters |
|---|---|---|
| Input image `x` | Image supplied to all models | Keeps comparison fair and aligns teacher outputs with student input |
| Ground truth `y` | Verified annotation (labels, boxes, masks) | The trusted supervised target |
| Teacher A / B | Existing trained, frozen models | Sources of learned predictions |
| Student C | New model from scratch | The only model whose parameters are updated |
| Teacher instance | One predicted object (class distribution, box, mask, confidence) | The unit of caching in Tier 1/2 |
| Reliability score `r` | Estimate of how useful a teacher is for a class / output type | Prevents equal trust in unequal teachers |
| Trust gate `g` | Absolute 0–1 factor: is this teacher good enough to learn from? | Fixes the "always sums to 1" problem |
| Share `s` | Relative split of influence between teachers | Answers "who is better here?" |
| Instance weight `ω` | Per-instance factor from GT comparison (veto/quality) | Handles individual errors |
| Veto | GT contradicts a teacher instance → its weight becomes 0 | Implements "GT can veto" |
| Ignore region | Pixels/anchors excluded from distillation loss | Implements "absence is not evidence" |
| Distillation loss | Loss comparing student output to teacher target | Transfers teacher behavior |
| Cache tier | Representation level of stored teacher output (1/2/3) | Trades fidelity vs. storage and augmentation safety |
| Validation set | Held-out labeled data for calibration, tuning, checkpoint selection | Estimates generalization |
| Test set | Final evaluation only | Less biased final estimate |

**Ground truth is the verified annotation, not the image.** The image is the question; the annotation is the answer key.

---

## 4. High-Level Architecture

```text
                     ┌────────────────────────── PHASE A (once) ──────────────────────────┐
  Images + GT ─────► │ Teacher A (frozen) ─► predictions ─┐                                │
  (train/val)        │ Teacher B (frozen) ─► predictions ─┤                                │
                     │                                    ▼                                │
                     │   GT matching ► metrics ► validation reliability r[k,c] (+shrink)   │
                     │                    │                                                │
                     │                    ▼                                                │
                     │   per-instance ω, trust gate g, shares s ► disk cache (sharded)     │
                     └──────────────────────────────────────────────────────────────────────┘
                                                      │
                     ┌────────────────────────── PHASE B (per run) ───────────────────────┐
                     │ batch(images, GT) + cache.read(ids) ► aligned augmentation          │
                     │   ► Student C forward ► L_GT + λ(t)·Σ_k w_k·L_distill_k ► backward  │
                     │   (teachers NOT loaded; only Student C updates)                     │
                     └──────────────────────────────────────────────────────────────────────┘
                                                      │
                     ┌────────────────────── PHASE C (optional) ───────────────────────────┐
                     │ Unlabeled images: teachers produce fused pseudo-labels with          │
                     │ class-level reliability only (no GT veto available)                  │
                     └──────────────────────────────────────────────────────────────────────┘
```

### Phase A — Teacher preparation and cache generation

1. Load one teacher checkpoint at a time; record its SHA-256.
2. Freeze, set to inference mode, disable gradients.
3. Read images in bounded batches using deterministic preprocessing.
4. Generate predictions (with a **low confidence floor** so reliability can be calibrated across thresholds).
5. Match predictions to GT (§7.3).
6. Compute metrics, calibrate, and compute reliability signals.
7. Write predictions, metadata, and weights to sharded disk storage.
8. Validate the cache (§6.5).
9. Unload the teacher before the next one.

### Phase B — Student training

1. Initialize Student C from scratch (seeded).
2. Load a batch of images and GT.
3. Read the matching cached records for exactly those image IDs.
4. Apply the **same** geometric transform to images, GT, and cached instances (§11).
5. Forward pass through the student.
6. Compute `L_GT`.
7. Compute per-teacher distillation losses with ignore masks.
8. Apply cached weights; combine with the scheduled `λ(t)`.
9. Backpropagate through the student only; update.
10. Validate periodically; checkpoint on the validation metric.

### Phase C (optional) — Unlabeled data

If extra unlabeled images exist, distillation is typically most valuable there, because GT cannot cover them. Use class-level reliability only (no per-example GT veto is possible), higher confidence thresholds, and fused pseudo-labels (§9). Evaluate it as a separate experiment (E8).

### Why split the phases?

Teacher inference is expensive and unnecessary to repeat when teachers and preprocessing are fixed. Splitting reduces GPU memory during student training and lets you re-run many student experiments against one cache.

---

## 5. Data Splits and Leakage Prevention

Define splits **before** generating any metric.

- **Training set:** trains the student; GT available for per-example comparison. If teachers were also trained on it, their behavior here is optimistic.
- **Validation set:** held out from teacher fitting whenever possible. Used for reliability, calibration, hyperparameters, checkpoint choice.
- **Test set:** never used for any decision. Final comparisons only.

### 5.1 Teacher data provenance table (fill in during Milestone 1)

| Teacher | Trained on which split(s)? | Validation set unseen by it? | Implication |
|---|---|---|---|
| A | [DECISION: e.g., train] | yes/no | If "no", val-based reliability is also inflated → create a fresh held-out split |
| B | [DECISION: e.g., train + val?] | yes/no | Same |

If either teacher saw the validation set, it cannot be used to estimate that teacher's generalization. Carve a **calibration split** the teachers never saw, even if small, or fall back to confidence-calibrated signals with explicit caveats.

### 5.2 Two kinds of reliability evidence (do not confuse them)

1. **Per-example evidence (veto only).** On a labeled training image, compare the teacher prediction with that image's GT. A *contradiction* is strong, reliable evidence the teacher is wrong there. An *agreement* is weak evidence of general reliability if the teacher memorized the image.
2. **General reliability (held-out).** Class-level and output-type reliability from unseen data. This sets the ceiling on how much a teacher can influence the student.

### 5.3 The asymmetry rule — "GT can veto, never boost"

```text
instance weight  = class_reliability_heldout × quality_from_GT_match
                   where quality_from_GT_match ∈ [0, 1] only REDUCES influence
                   (mismatch → 0; good match → capped at 1, never above class reliability)
```

This prevents memorized training images from making a teacher look better than it really is, while still removing guidance that GT shows to be wrong.

### 5.4 If no suitable validation data exists

Create a clean split before tuning any adaptive weighting. Never quietly reuse the test set. If the dataset is tiny, use K-fold evaluation for the **student** experiments and report variance, while keeping teacher reliability estimated only from data the teachers never trained on.

---
## 6. Phase A: Teacher Inference and Cache Preparation

### 6.1 Teacher inference procedure

For each teacher:

1. Load the checkpoint; confirm architecture, class mapping, and input size; record SHA-256.
2. Switch to inference mode; disable gradients.
3. Use deterministic, versioned preprocessing (same resize/letterbox/normalization the student will see at canonical resolution).
4. Run inference with a **low confidence threshold and permissive NMS** (e.g., conf ≥ 0.05) so calibration curves cover the full confidence range. Apply stricter thresholds later at *use* time, not at cache time. **[DECISION: exact values]**
5. Process in batches sized to memory; write shards incrementally.
6. Log failed image IDs; never silently skip.
7. Release model and accelerator memory.
8. Optionally run with test-time augmentation (TTA, e.g., flip) as a separate, documented cache variant. Do not mix TTA and non-TTA records.

**Why sequential?** Loading both teachers at once raises peak memory for no benefit. Run A → save → unload → run B.

### 6.2 Cache tiers (choose the smallest tier that supports the planned loss)

| Tier | Stored per teacher instance | Approx. size | Supports | Augmentation-safe? |
|---|---|---|---|---|
| **1 — Instance pseudo-labels** | class, confidence, box, mask (RLE or polygon), metadata | Very small (KBs/image) | Hard/soft pseudo-label distillation through the **existing GT-style loss** | **Yes** (instances can be geometrically transformed) |
| **2 — Soft instance targets** | Tier 1 + top-k class probability vector, mask probability at reduced resolution (e.g., 1/4), mask-quality score | Small–medium | Soft-label KD on class and mask, calibrated confidences | **Yes**, with the same transforms applied to the reduced-res mask |
| **3 — Dense logits** | Per-scale head outputs / per-pixel logits | Large (MBs/image) | Logit/feature-style KD for **same-architecture** teacher→student | **No** for random geometry; only with deterministic inputs |

**Recommendation:** Build Tier 1 first (cheap, works across architectures, supports augmentation). Add Tier 2 fields when you want soft targets. Only build Tier 3 if an ablation shows Tier 1/2 leaves accuracy on the table **and** teacher/student architectures are compatible.

### 6.3 What every record contains

- Stable image ID and relative path; image width/height; preprocessing version.
- Teacher name, checkpoint SHA-256, class-map version, cache-schema version, generation timestamp, config hash.
- Per instance: class ID, class probabilities (Tier 2), confidence, box, mask, mask-quality score.
- GT-match results (if labeled): matched GT index, IoU, class-correct flag, status ∈ {`TP`, `FP_class`, `FP_loc`, `FP_unmatched`}.
- Instance weight components: `class_reliability`, `quality`, final `omega` (§8).
- Per image: list of **missed GT objects** (`FN`) for the teacher, stored so they are never lost from scoring.

### 6.4 Cache format and layout

Prefer a format with incremental writes and random access: sharded files (e.g., Parquet/Arrow or `.npz`/`.pt`) with an index mapping `image_id → (shard, offset)`. Keep metadata (small, frequently scanned) separate from bulky arrays (masks). Store masks as **RLE or polygons** in Tier 1/2; avoid raw full-resolution float masks.

```text
distillation_project/
├── configs/
│   ├── teacher_a.yaml
│   ├── teacher_b.yaml
│   ├── reliability.yaml
│   └── student.yaml
├── manifests/
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   ├── calib_manifest.csv          # NEW: data neither teacher saw (if available)
│   └── test_manifest.csv
├── teacher_cache/
│   ├── cache_manifest.json         # provenance + hashes (see below)
│   ├── index_teacher_a.parquet     # image_id → shard, offset, n_instances
│   ├── index_teacher_b.parquet
│   ├── teacher_a/shard_00000.* ...
│   └── teacher_b/shard_00000.* ...
├── reliability/
│   ├── matches_teacher_a.parquet   # per-instance GT matches
│   ├── matches_teacher_b.parquet
│   ├── validation_metrics.json
│   ├── calibration_maps.json       # confidence → expected quality (per teacher/class)
│   ├── class_reliability.json      # r, n, CI, shrunk r_hat, gate g, share s
│   └── calibration_report.md
├── src/                            # see Appendix C for module/CLI layout
├── checkpoints/
├── logs/
└── reports/
```

**`cache_manifest.json` (required fields):**

```json
{
  "schema_version": "2.0",
  "created_utc": "2026-11-08T12:00:00Z",
  "dataset_manifest_sha256": "…",
  "class_map_version": "v3",
  "preprocess": {"resize": 640, "letterbox": true, "normalize": "0-1", "version": "p1"},
  "teachers": {
    "A": {"checkpoint_sha256": "…", "arch": "…", "tier": 2, "conf_floor": 0.05, "nms_iou": 0.7, "tta": false},
    "B": {"checkpoint_sha256": "…", "arch": "…", "tier": 2, "conf_floor": 0.05, "nms_iou": 0.7, "tta": false}
  },
  "n_images": 0,
  "failed_image_ids": [],
  "cache_key": "sha256(all of the above)"
}
```

**Why a manifest?** It proves which checkpoint, preprocessing, label map, and image set produced the cache. A mismatch on load must be a hard error, not a warning.

### 6.5 Cache validity checks (hard gates before training)

- Every required image has a record for each teacher (or an explicit "no detections" record — distinguish *empty* from *missing*).
- Image IDs match exactly; never rely on directory order.
- Class IDs/names use one mapping (explicit conversion table if teachers differ).
- Mask dimensions agree with recorded image size.
- No NaN/Inf; boxes inside image bounds (after clipping); masks non-degenerate.
- Checkpoint hashes in the manifest equal the hashes of the files being used.
- `cache_key` recomputed from current configs equals the stored key.
- **Visual spot check:** overlay teacher instances on ≥ 50 random images per teacher and review.
- **Round-trip test:** write → read → compare for a random sample.

Invalidate and regenerate the affected cache whenever a checkpoint, image, class map, or relevant preprocessing changes.

### 6.6 Storage estimator (run before full generation)

```text
bytes_per_image ≈ n_instances × (box + class/prob + conf + mask_bytes) + metadata
total ≈ n_images × bytes_per_image × n_teachers × (1 + overhead≈0.1)
```

Measure `n_instances` and `mask_bytes` on a 200-image sample and extrapolate. For example, with 10,000 images, 2 teachers, ~12 instances/image, ~2 KB per RLE mask → ≈ 0.5 GB (Tier 1/2). A dense Tier 3 cache at 1 MB/image/teacher would be ≈ 20 GB. These numbers are illustrative; measure yours.

---

## 7. Phase A: Measuring Teacher Reliability

Reliability must not be a single global accuracy number when teachers have different strengths.

### 7.1 Reliability levels (start simple)

1. **Global** — overall quality on held-out data (fallback and shrinkage target).
2. **Class-level** — `r[k, c]` per teacher `k`, class `c`. **Default starting point.**
3. **Output-type** — separate scores for classification, localization (box), and mask quality: `r_cls`, `r_box`, `r_mask`.
4. **Instance-level** — `ω` from GT match plus calibrated confidence (§7.5).

### 7.2 Choosing the reliability score (must match what is being distilled)

| Distilled signal | Suggested reliability score | Not suitable |
|---|---|---|
| Class label/probability | Class precision at the operating confidence, or calibrated accuracy | mAP alone (mixes localization), raw confidence |
| Box | Mean IoU of matched TPs, weighted by recall | Class accuracy |
| Mask | Mean mask IoU / Dice of matched TPs, weighted by recall | Box IoU |
| Overall instance | `F-β` or `AP50` per class (with sample counts) | Single threshold-free number without counts |

**Recall must be included.** A score computed only over matched predictions rewards a teacher for predicting only easy objects. Use a *recall-aware* score, e.g.:

```text
r_mask[k,c] = recall[k,c] × mean_mask_IoU_of_TPs[k,c]        (or F1-like combination)
```

### 7.3 Matching procedure (explicit and shared by all teachers)

For each image with GT:

1. Consider teacher instances with confidence ≥ the cache's conf floor.
2. Compute the pairwise **mask IoU** (box IoU if no masks) between every teacher instance and every GT instance.
3. Solve a one-to-one assignment (Hungarian) maximizing IoU, subject to `IoU ≥ τ_match` (default 0.5 **[DECISION]**), allowing class-agnostic matching so class errors are detectable.
4. Label each teacher instance:
   - `TP`: matched, class correct.
   - `FP_class`: matched, wrong class (localized correctly, labeled wrongly).
   - `FP_loc`: same class, best IoU in `[τ_low, τ_match)` (poor localization).
   - `FP_unmatched`: no GT overlap (either a true false positive **or a missing GT label**).
5. Every unmatched GT object is recorded as `FN` for that teacher.

Use the **same matcher, thresholds, and tie-breaking** for both teachers and for the final evaluation code, so reliability and reported metrics are consistent.

### 7.4 General (held-out) reliability

Compute `r[k, c]` on validation/calibration data. Store sample count `n[k,c]` and a confidence interval (bootstrap over **images**, not instances, to respect within-image correlation).

**Shrinkage (required for small classes):**

```text
r_hat[k,c] = ( n[k,c] · r[k,c] + m · r_global[k] ) / ( n[k,c] + m )
```

where `m` is a pseudo-count (start with 20 **[DECISION]**, tune on validation). Classes with `n < n_min` use `r_global` entirely and are flagged in the report.

**Why smooth?** A class with 3 validation examples must not receive an extreme weight from one lucky or unlucky image.

### 7.5 Calibrated per-instance reliability

Raw confidence is not correctness. On validation data, fit a monotone map (isotonic regression or Platt/temperature scaling) per teacher (and per class group if data allows):

```text
q_hat[k](conf) = P(instance is TP with IoU ≥ τ | confidence)
```

Report calibration error (ECE) per teacher. If a teacher is badly calibrated, either apply the map, or exclude raw confidence from the weighting. Because teachers are usually **overconfident on their own training images**, do not assume the validation calibration transfers perfectly to the training split; this is why per-example GT veto (§5.3) is still applied on labeled images.

### 7.6 Reliability dashboard (required report)

`reliability/calibration_report.md` must show, per teacher and class: `n`, precision, recall, mean IoU, `r`, `r_hat`, CI, calibration error, and a flag for low-sample classes. Plot reliability diagrams.

---

## 8. Adaptive Teacher Weighting

### 8.1 Concept

For teacher `k`, image `i`, instance/class `c`, a non-negative **loss weight** `w[k,i,c]` scales that teacher's distillation loss. It is *not* the teacher's network parameter weight and does not directly modify the student's parameters; it scales a loss term that shapes the student's gradients.

### 8.2 Decomposition (replaces v1's `r / Σr`)

```text
w[k,i,c] =  g[c]  ×  s[k,c]  ×  ω[k,i,c]
            └─┬─┘    └──┬──┘    └───┬───┘
        absolute    relative     per-instance
        trust gate  share        GT veto/quality
```

**(a) Absolute trust gate `g[c] ∈ [0, 1]`** — "is any teacher good enough to learn from for this class?"

```text
r_best[c] = max_k r_hat[k,c]
g[c] = clip( (r_best[c] − r_ref[c]) / (r_max − r_ref[c]), g_min, 1 )   if r_best[c] ≥ r_ref[c]
g[c] = 0                                                                otherwise
```

`r_ref[c]` is the quality of a **GT-only student** on class `c` (measured in Milestone 3, validation set), optionally plus a margin. This makes the gate *relative to what the student can already do*, which is more meaningful than an arbitrary absolute threshold. Until E2 exists, use a provisional fixed threshold and record it as provisional.

**(b) Relative share `s[k,c]`** — "who should we listen to more?"

```text
s[k,c] = softmax_k( r_hat[k,c] / T )          # T = temperature
```

or proportional: `s[k,c] = r_hat[k,c] / Σ_j r_hat[j,c]`. With `T → ∞` shares become uniform; with `T → 0` the best teacher dominates. Tune `T` (and the choice proportional vs. softmax) on validation data. Add a **floor** `s_min` (e.g., 0.1) when both teachers are usable, to preserve diversity, and set `s[k,c] = 0` for any teacher below `r_ref[c]` individually.

**(c) Instance weight `ω[k,i,c]`** — GT-based veto/quality, from §7.3 labels (labeled images only):

| Match status | `ω` | Rationale |
|---|---|---|
| `TP` | `clip((IoU − τ_low)/(1 − τ_low), ω_min, 1)` | Better overlap → more trust, but never above class-level reliability (asymmetry rule) |
| `FP_class` | 0 | GT contradicts the class: veto classification distillation for this instance (box/mask distillation may stay if localization was good — configurable) |
| `FP_loc` | 0 or small | Poor localization: veto mask/box distillation |
| `FP_unmatched` | 0 for distillation, **quarantined** | May be a true error *or* a missing GT label (see §8.6) |
| `FN` (missed GT) | n/a | No target exists; handled by ignore policy (§10.3) |
| Unlabeled image (Phase C) | `q_hat[k](conf)` | No GT available → calibrated confidence only |

### 8.3 Optional output-type split

Replace the single `w` with `w_cls`, `w_box`, `w_mask`, each built from the corresponding `r_cls`, `r_box`, `r_mask`. A teacher can be a good classifier but a poor mask producer; this prevents copying its masks when only its labels are trustworthy.

### 8.4 Threshold ("IF–ELSE") behavior, formalized

```python
# per class c, per teacher k  (computed once in Phase A, stored in class_reliability.json)
usable[k, c] = (r_hat[k, c] >= r_ref[c] + margin) and (n[k, c] >= n_min)

# per instance (computed once in Phase A for labeled images)
omega[k, i, j] = 0.0 if status in {"FP_class", "FP_loc", "FP_unmatched"} else quality_from_iou

weight = gate[c] * share[k, c] * omega[k, i, j] if usable[k, c] else 0.0

# training loop (Phase B) ALWAYS keeps GT active:
total = L_gt + lam(t) * sum(weight_k * L_distill_k for k in teachers)
```

Hard thresholds create discontinuities; prefer the **smooth gate** in §8.2 and keep thresholds only for `usable` and for veto. Compare "hard gate" vs. "smooth gate" as an ablation (A2 in §13.4).

### 8.5 If both teachers are unreliable for a class

`g[c] = 0` → both distillation weights are zero for that class → the student trains from GT alone for it. Do not force imitation of an unreliable teacher.

### 8.6 Handling `FP_unmatched` (possible missing GT labels)

A confident, consistent teacher detection with no GT match is either a true false positive or an **unlabeled real object**. Do **not** copy it into the target, and do **not** let it silently penalize the teacher without review.

- Quarantine these instances in a review file ordered by confidence and by **A-and-B agreement** (both teachers detect it, high confidence → high probability of missing GT).
- Review a sample manually; if the rate of missing GT is material, correct the labels *through the normal annotation process* and regenerate reliability artifacts.
- Report two versions of teacher precision: with and without quarantined instances, to expose the sensitivity.

### 8.7 When both teachers are reliable

Combine through the fusion policy (§9). Agreement is useful but not proof; two models can share a blind spot — GT remains active.

### 8.8 Misinterpretation guard

Teacher reliability weights are **loss weights**, not multipliers applied to the student's parameters. The student's parameters change only via gradients of the combined objective.

---

## 9. Teacher Fusion and Disagreement Policy

When both teachers detect objects in the same image, the student needs a consistent set of targets rather than duplicates.

### 9.1 Cross-teacher instance clustering

1. For each image, cluster A's and B's instances by class-agnostic mask IoU ≥ `τ_cluster` (default 0.6 **[DECISION]**), using the same one-to-one matcher as §7.3 between the two teachers.
2. Each cluster has 1 member (only one teacher saw it) or 2 members (both saw it).

### 9.2 Rules per cluster

| Situation | Rule |
|---|---|
| Both agree on class, masks overlap well | **Fuse**: class distribution = weight-averaged; mask = weight-averaged probability then threshold; box = weighted box fusion. Weights = per-teacher `w` |
| Both detect, **classes disagree**, GT available | GT arbitrates: keep the teacher whose class equals GT class; veto the other (via `ω`) |
| Both detect, classes disagree, **no GT** (Phase C) | Use the higher-`w` teacher only if `w_max / (w_min + ε) ≥ ρ`; otherwise discard the cluster (don't teach the student to guess) |
| Only one teacher detects | Use that teacher's instance with its own `w`; down-weight by a **solo penalty** `κ ∈ [0.5, 1]` if the other teacher has high class reliability there and *missed* it (disagreement signal) |
| Neither teacher detects a GT object | `FN` — no distillation target; GT loss covers it |

### 9.3 Mask fusion detail

```text
P_fused(x) = Σ_k w_k · P_k(x) / Σ_k w_k       (P_k = mask probability, aligned to the same grid)
mask = P_fused ≥ 0.5
```

Keep the fused result **soft** (Tier 2) when the loss supports it; threshold only for hard-label losses.

### 9.4 Diversity tracking

Log the fraction of clusters with 1 vs. 2 members and the class-disagreement rate by class. If teachers almost never disagree, the second teacher adds little; if they disagree often, the fusion policy matters more. This is an input to deciding whether H2 is plausible before spending compute.

---

## 10. Phase B: Student Training

### 10.1 Student initialization

From scratch. Record seed, architecture, initializer, optimizer, schedule, precision, and config hash. Without these, a baseline and a distillation run might differ for reasons unrelated to teacher guidance.

### 10.2 Training objective

```text
L_total = L_GT  +  λ(t) · Σ_{k∈{A,B}}  [ Σ_{instances j} w[k,i,j] · ℓ_distill( student, target_k,j ) ] / N_norm
```

- `L_GT`: normal supervised loss (class, box, mask terms).
- `ℓ_distill`: per-instance distillation term (below).
- `w[k,i,j]`: cached weight from §8 (already includes gate, share, and instance factor).
- `λ(t)`: scheduled overall strength.
- `N_norm`: normalization (see §10.5), chosen so `λ` has a stable meaning across batches.

### 10.3 Per-tier distillation losses

**Tier 1 (hard pseudo-label as extra targets).** Add the fused teacher instances to the target set of the student's normal detection/segmentation loss **as a second, weighted target stream**, each with weight `w`. Implementation pattern: compute the standard loss on the GT stream (weight 1) and on the teacher-instance stream (weight `λ·w`). Where a teacher instance and a GT instance coincide, drop the teacher copy (GT already supervises that object) unless using it to supply a soft class label.

**Tier 2 (soft targets).**
- Class: `KL( softmax(z_s/τ) ‖ p_teacher^τ ) · τ²` on matched student predictions (τ = distillation temperature, start 2 **[DECISION]**), or BCE with soft labels if the head is sigmoid-based.
- Mask: soft BCE or Dice between student mask logits and fused teacher mask probability, restricted to the instance region.
- Box: L1/GIoU to the teacher box, weighted by `w_box`.

**Tier 3 (dense logits, same architecture only).** Per-scale KL/MSE on logits, masked by ignore regions.

### 10.4 Ignore-mask policy ("absence is not evidence")

- Never create a "background" target from a teacher's lack of detection.
- In regions covered by a **vetoed** instance (§8.2) or by a quarantined `FP_unmatched`, exclude those pixels/anchors from distillation (but **not** from GT loss).
- For GT objects the teacher missed (`FN`), distillation contributes nothing; the GT loss alone supervises them.
- If using dense (Tier 3) losses, multiply the loss by a binary mask that is 0 outside trusted regions.

### 10.5 Loss reduction (the detail v1 left implicit)

Specify before coding:
- **Per-instance weights**, reduced as a weighted mean: `Σ w·ℓ / max(Σ w, ε_N)`, or divided by a fixed constant if you want gradient magnitude to scale with how much trusted guidance exists. Choose one and keep it fixed across experiments.
- Prevent a batch with zero trusted teacher instances from producing NaNs (guard the denominator).
- Report **effective distillation mass** `Σ w` per epoch, per class. If it is near zero for most classes, distillation is effectively off and E6 ≈ E2.
- Scale `λ` after comparing the **gradient norm** contributed by GT vs. distillation in the first 100 steps; extreme ratios (> 10× either way) usually indicate a scaling error.

### 10.6 Distillation schedule (ablation A5)

Start with constant `λ`. Then test:
- **Warm-up:** `λ` ramps 0 → `λ_max` over the first `E_w` epochs (avoids teacher noise dominating an untrained student).
- **Decay:** `λ` decays toward a small value late in training so the student specializes on GT and corrects teacher errors.
- Choose among them **on validation only**.

### 10.7 Training-step pseudocode (Tier 1/2)

```python
for step, batch in enumerate(train_loader):
    images, gt, image_ids = batch

    ta = cache_a.read(image_ids)          # instances + cached weights
    tb = cache_b.read(image_ids)
    t_fused = fuse(ta, tb)                # §9, can itself be precomputed in Phase A

    images, gt, t_fused = aligned_augment(images, gt, t_fused)   # §11

    out = student(images)

    loss_gt = supervised_loss(out, gt)
    loss_kd, kd_stats = distill_loss(out, t_fused, tau=cfg.tau,
                                     ignore=t_fused.ignore_masks)  # per-instance w inside

    loss = loss_gt + lam_schedule(step) * loss_kd

    optimizer.zero_grad(set_to_none=True)
    loss.backward()                        # only student parameters exist in the graph
    optimizer.step()

    log(loss_gt, loss_kd, kd_stats.effective_mass, kd_stats.per_class_mass)
```

Precomputing the fused targets in Phase A is allowed **only if** fusion weights are fixed; if you change `T`, `s_min`, or thresholds, regenerate or recompute them from the stored per-teacher records (hence store raw per-teacher records, not only the fused result).

### 10.8 Freeze the teachers

Ideally the teacher networks are not loaded at all during student training. If a checkpoint changes, the cache and reliability artifacts must be regenerated.

---
## 11. Augmentation and Alignment

Cached predictions are tied to the exact image and preprocessing used at teacher inference.

### 11.1 Why Tier 1/2 helps here

Because Tier 1/2 stores **instances** (boxes, RLE/polygon masks, reduced-res mask probabilities), the same affine/geometric transform applied to the image can be applied to those instances, exactly as for GT annotations. Dense Tier 3 logits cannot be warped faithfully for arbitrary geometry.

### 11.2 Augmentation ladder (introduce in this order, test each)

| Level | Augmentation | Teacher-target handling | Notes |
|---|---|---|---|
| L0 | None (deterministic resize/normalize) | Use cache directly | First correctness prototype |
| L1 | Photometric only (color jitter, blur, noise) | Targets from the **clean** image (this is a standard "weak teacher / strong student" setup) | Evaluate; it is a changed learning setup, not automatically harmless |
| L2 | Flips, scale, translate, rotate, crop | Apply identical geometric transform to GT **and** cached instances; recompute boxes from transformed masks; drop instances that fall mostly outside | Re-check class/mask/box consistency |
| L3 | Mosaic / mixup / copy-paste | Compose instances like GT; ignore-masks must be composed too; mixup requires blending soft targets with matching coefficients | Highest risk of silent misalignment |

### 11.3 Alignment tests (required before enabling any level)

- **Golden test:** for a fixed seed and image, draw GT and teacher instances on the augmented image; assert overlays line up (automated IoU between transformed GT and a teacher instance known to match it must stay ≥ pre-transform IoU − small tolerance).
- **Round-trip test:** flip twice returns the original.
- **Visual dump:** save 100 augmented batches with overlays for manual review.

### 11.4 Why this matters

If an image is flipped but the cached mask is not, the student receives contradictory targets with no runtime error. Such bugs can silently erase the gains of distillation.

---

## 12. Memory and Storage Strategy

Caching removes repeated teacher inference from the loop but does not make storage free.

### 12.1 Keep the cache on disk and stream it

- Shard files; index by image ID; read only the current batch's records.
- Use `DataLoader` workers that open shards lazily (avoid pickling large arrays to workers).
- Memory-map arrays where supported; optionally keep a small bounded LRU cache in RAM.
- Keep metadata (weights, statuses) in a compact table that fits RAM; keep bulky masks on disk.
- Shuffle at the shard level plus a buffer, or use an index-based sampler so random access cost is measured, not assumed.

### 12.2 Measure, don't guess

Record for a realistic batch: cache read time per batch, peak RAM, peak GPU memory, dataloader throughput (images/s), and GPU utilization. If cache reads starve the GPU, increase workers, pack shards larger, or switch to a more compact encoding.

### 12.3 Reducing cache size (only after measuring the cost to accuracy)

- Store only what the chosen loss uses.
- Lower-resolution mask probabilities (e.g., 1/4) and float16/uint8 quantization — verify numerically.
- RLE/polygon instead of raw masks.
- Drop instances below a use-time confidence floor from the *training* cache while keeping the full low-confidence set in the *calibration* artifacts.
- Compression (zstd/lz4) after benchmarking decode speed.

### 12.4 Why not store only final hard answers?

Hard labels are compact but discard uncertainty and class alternatives. Ablation A1 (Tier 1 vs. 2 vs. 3) decides whether the extra fidelity pays for its storage.

---

## 13. Evaluation Plan and Required Baselines

### 13.1 Core experiments

| ID | Student training signal | Question answered |
|---|---|---|
| E0 | Teacher A (existing checkpoint) | How good is A? |
| E1 | Teacher B (existing checkpoint) | How good is B? |
| E2 | GT-only scratch student | What can the student learn without teachers? **(the baseline everything is compared to)** |
| E3 | GT + A (fixed `λ`) | Does A help? (H1) |
| E4 | GT + B (fixed `λ`) | Does B help? (H1) |
| E5 | GT + A+B, fixed equal weights (fused) | Do two teachers help? (H2) |
| E6 | GT + A+B, adaptive `w = g·s·ω` | Does adaptive weighting help? (H3) |

### 13.2 Controls (new — these make the result interpretable)

| ID | Control | What it rules out |
|---|---|---|
| C1 | **Shuffled reliability:** randomly permute `r_hat` across classes (and `ω` across instances within teacher) while keeping the same weight distribution | Gains caused merely by "some regularization/noise in the loss", not by reliability information (H4) |
| C2 | **Uniform within gate:** `s = 0.5`, `ω = 1`, keep `g` | Whether sharing/instance weights add anything beyond gating |
| C3 | **Inverse reliability:** weight the *worse* teacher more | A sanity check; should hurt or at least not help. If it helps equally, the weights carry no signal |
| C4 | **Teacher-only reference:** fused A+B ensemble evaluated directly (no student) | Upper-ish reference for what the combined teacher knowledge is worth |
| C5 | **Longer GT-only training / equal compute:** E2 with the same wall-clock or epochs as E6 (and, separately, same number of loss terms if relevant) | Gains caused by extra optimization budget, not by teacher information |

### 13.3 Statistical protocol

- **Seeds:** at least 3 per configuration for E2, E5, E6 and controls (5 if affordable). Report mean ± std.
- **Estimate noise first:** run E2 with 3+ seeds in Milestone 3; use its std to set `δ_min` (§2.4).
- **Confidence intervals:** paired bootstrap over **images** (same resampled images for both runs being compared) for the primary metric; report the 95% CI of the difference.
- **Multiple comparisons:** with several variants, use pre-registered primary comparisons (H1–H5); treat the rest as exploratory and label them so.
- **No test-set peeking:** hyperparameters and checkpoints chosen on validation; test evaluated once per final configuration.

### 13.4 Ablations (each answers "does this component earn its complexity?")

| ID | Ablation | Compare |
|---|---|---|
| A1 | Cache tier 1 vs. 2 vs. 3 | Accuracy vs. storage/throughput |
| A2 | Hard threshold gate vs. smooth gate | §8.2 vs. §8.4 |
| A3 | Class-level only vs. + instance `ω` | Value of GT veto |
| A4 | Proportional vs. softmax shares; temperature `T` | §8.2(b) |
| A5 | `λ` constant vs. warm-up vs. decay | §10.6 |
| A6 | Single `w` vs. output-type `w_cls/w_box/w_mask` | §8.3 |
| A7 | Augmentation level L0 → L3 | §11.2 |
| A8 | Shrinkage `m` ∈ {0, 10, 20, 50} | §7.4 |
| A9 | With vs. without solo-penalty `κ` and cluster-disagreement rules | §9.2 |

Run ablations on **validation**; carry only the best configuration to the single final test evaluation.

### 13.5 Fair comparison rules

- Same train/val/test splits and class mapping.
- Same student architecture and comparable optimization budget (and explicit C5 for equal compute).
- Identical evaluation code, IoU thresholds, NMS, confidence cut-offs, and post-processing.
- No tuning on the test set.
- Record seeds and full config for every run.
- Track training time, peak GPU memory, peak RAM, cache size, and throughput.
- Report per-class results and sample counts, not only one aggregate.

### 13.6 Analysis beyond the headline number

- **Per-class delta vs. E2** plotted against the teacher reliability gap (tests H5).
- **Error analysis:** inspect the 50 worst regressions vs. E2 and the 50 largest gains; categorize (class confusion, mask boundary, missed small objects, false positives).
- **Teacher-error inheritance:** how often does the student reproduce a *known* teacher error (a vetoed instance)? A good method should reproduce it less often than a naive pseudo-labeling baseline.
- **Calibration of the student:** ECE for E2 vs. E6.

### 13.7 Example report table

| Run | Seeds | Mask mAP50 (mean ± std) | Mask mAP50–95 | Precision | Recall | Δ vs. E2 [95% CI] | Peak GPU | Cache size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 Teacher A | 1 | — | — | — | — | — | — | — |
| E1 Teacher B | 1 | — | — | — | — | — | — | — |
| E2 GT-only | 3+ | — | — | — | — | 0 | — | N/A |
| E3 + A | 3 | — | — | — | — | — | — | — |
| E4 + B | 3 | — | — | — | — | — | — | — |
| E5 + A+B fixed | 3+ | — | — | — | — | — | — | — |
| E6 + A+B adaptive | 3+ | — | — | — | — | — | — | — |
| C1 shuffled | 3 | — | — | — | — | — | — | — |
| C2 uniform-in-gate | 3 | — | — | — | — | — | — | — |
| C3 inverse | 3 | — | — | — | — | — | — | — |
| C4 teacher ensemble | 1 | — | — | — | — | — | — | — |

Fill with measured results only. Never infer results from training loss.

---

## 14. Failure Modes and Safeguards

| Risk | Likelihood | Impact | Why it matters | Safeguard |
|---|---|---|---|---|
| Teacher overfitting to train images | High | High | Train-set agreement overstates reliability | Held-out reliability; "GT can veto, never boost"; provenance table (§5.1) |
| Teachers saw the validation set | Medium | High | Reliability estimates biased | Fresh calibration split; document otherwise |
| Bad / missing GT labels | Medium | High | Correct teachers penalized; true objects treated as FPs | Quarantine `FP_unmatched`; A∧B-agreement review; audit sample (§8.6) |
| Shared teacher errors | Medium | Medium | Agreement ≠ correctness | GT always active; evaluate on held-out data |
| Class imbalance | High | Medium | Aggregates hide weak rare classes | Per-class metrics, shrinkage, `n_min`, report counts |
| No-prediction cases | High | Medium | Matched-only scoring rewards omission | Recall-aware score; FN recorded; absence ≠ negative |
| Stale cache | Medium | High | Outputs inconsistent with current checkpoint/labels | Hashes + `cache_key`; hard fail on mismatch |
| Augmentation misalignment | Medium | High | Contradictory targets, silent | Golden tests; ladder L0→L3 |
| Excessive distillation | Medium | Medium | Student copies teacher mistakes | Tune `λ`; schedule; GT anchor; veto |
| Trust gate set incorrectly | Medium | Medium | Wasted or harmful guidance | Gate relative to GT-only student; provisional flag; ablation A2 |
| Weights carry no signal | Medium | Medium | Complexity with no benefit | Controls C1/C3; remove component if no gain |
| Loss-scale mismatch | High | Medium | One term dominates | Gradient-norm check in first 100 steps (§10.5) |
| RAM/disk exhaustion | Medium | Medium | Large caches | Tiering, sharding, estimator, streaming |
| Dataloader bottleneck | Medium | Medium | Cache reads starve GPU | Benchmark; worker tuning; shard packing |
| Class mapping differences | Medium | High | Wrong class merged | Explicit conversion table; assertions in cache validation |
| Metric mismatch | Medium | Medium | Confidence mistaken for correctness | Calibration; metric chosen per distilled signal (§7.2) |
| Evaluation inconsistency | Low | High | Runs not comparable | One shared evaluation module; config hash in report |
| Test-set leakage | Low | High | Optimistic final result | Test touched once; access logged |
| Seed noise mistaken for gains | High | Medium | False positives for method | ≥ 3 seeds; bootstrap CIs; `δ_min` from E2 variance |
| Hidden compute cost | Medium | Medium | Cache build + overhead exceeds benefit | Track total cost; gate in §2.4 |

---

## 15. Reproducibility, Logging, and Configuration

### 15.1 Record for every experiment

- Teacher checkpoint IDs/hashes and cache key.
- Student architecture, initialization, seed.
- Dataset manifest hash and split IDs; class-map version.
- Preprocessing and augmentation level.
- Cache schema/tier and generation settings (conf floor, NMS, TTA).
- Reliability definitions, `n`, CIs, shrinkage `m`, gate rule, `T`, `s_min`, `κ`.
- GT and distillation loss definitions, `λ` schedule, `τ`.
- Optimizer, LR schedule, batch size, precision, seed.
- Training/validation curves; per-class metrics; effective distillation mass.
- Cache errors and missing records.
- Runtime, peak RAM, peak accelerator memory, disk usage; git commit hash and environment (library versions).

**Why:** If a result improves, these records establish why. If it degrades, they help isolate data, teacher quality, weights, alignment, loss scale, or optimization.

### 15.2 Example experiment config (`configs/student.yaml`)

```yaml
experiment:
  name: e6_adaptive_s0
  seed: 0
  baseline_ref: e2_gt_only        # for reporting deltas
splits:
  train: manifests/train_manifest.csv
  val:   manifests/val_manifest.csv
  test:  manifests/test_manifest.csv   # evaluated only by `final_eval`
student:
  arch: <your_student_arch>
  init: scratch
  imgsz: 640
  epochs: 200
  optimizer: {name: sgd, lr0: 0.01, momentum: 0.937, weight_decay: 0.0005}
  amp: true
cache:
  dir: teacher_cache/
  tier: 2
  require_key: <cache_key>        # hard-fail if the manifest key differs
distill:
  enabled: true
  teachers: [A, B]
  lambda: {schedule: warmup, max: 1.0, warmup_epochs: 10, decay: none}
  temperature: 2.0
  loss_reduction: weighted_mean   # see §10.5
  ignore_policy: {vetoed: true, fp_unmatched: true, teacher_absence: true}
weights:
  mode: adaptive                  # adaptive | fixed | shuffled | inverse | uniform_in_gate
  gate: {type: smooth, ref: gt_only_student_val, margin: 0.02, g_min: 0.0}
  share: {type: softmax, temperature: 0.5, s_min: 0.1}
  instance: {enabled: true, tau_match: 0.5, tau_low: 0.1, omega_min: 0.1}
  shrinkage_m: 20
  n_min: 10
  output_type_split: false
fusion:
  tau_cluster: 0.6
  solo_penalty: 0.8
  disagreement_ratio: 2.0
augment:
  level: L0                       # L0 | L1 | L2 | L3
eval:
  conf: 0.001
  nms_iou: 0.7
  primary_metric: mask_mAP50-95   # FIXED before experiments
```

---

## 16. Implementation Milestones (with exit gates)

Each milestone ends with a **gate** — do not start the next until it passes.

### Milestone 1 — Audit inputs and models
- Confirm both checkpoints load; confirm class maps (or define conversion).
- Fill the teacher-provenance table (§5.1); confirm an unseen calibration/validation split exists or create one.
- Measure teacher inference memory/time on a small sample; estimate cache size (§6.6).
- Confirm student architecture/output format; agree compute budget.
- **Gate:** provenance table complete; splits frozen and hashed; Appendix B questions answered or explicitly assumed.

### Milestone 2 — Small cache prototype
- Run both teachers on ~200 images; write Tier 1 (+ Tier 2 fields).
- Reload by ID; visually verify masks/classes; run all cache validity checks (§6.5).
- Implement the matcher (§7.3) and verify it against hand-checked cases.
- **Gate:** validation checks pass; overlays reviewed; matcher unit tests pass; storage estimate within 2× of measured.

### Milestone 3 — GT-only baseline student
- Train E2 with ≥ 3 seeds; save checkpoints and metrics; build the shared evaluation module.
- Measure seed variance → set `δ_min`; compute `r_ref[c]` per class (validation).
- **Gate:** E2 reproducible within expected variance; evaluation code locked (config hash recorded).

### Milestone 4 — Full cache + reliability artifacts
- Generate full caches for both teachers; compute matches, validation reliability, shrinkage, calibration maps, gates, shares, `ω`.
- Produce `calibration_report.md`; review quarantined `FP_unmatched` sample.
- **Gate:** report reviewed; no unexplained class with extreme weight; cache validated end to end.

### Milestone 5 — Add one teacher
- Implement Tier 1/2 distillation for A (E3) at `L0`.
- Verify loss values, gradient flow, and that **only student parameters change**; check gradient-norm ratio.
- **Gate:** E3 trains stably; effective distillation mass non-trivial; no alignment errors.

### Milestone 6 — Second teacher + fixed weights + fusion
- Add B (E4), then fused fixed-weight E5 with the fusion policy.
- Log cluster statistics (§9.4).
- **Gate:** each loss verified independently; fusion outputs visually correct.

### Milestone 7 — Adaptive weighting
- Start with class-level `g·s`; then add `ω`; add output-type split only if justified.
- Run controls C1–C3 and ablations A2–A6, A8, A9 on validation.
- **Gate:** E6 > E5 and E6 > C1/C2/C3 on validation; otherwise simplify or stop (decision rule in §2.4).

### Milestone 8 — Augmentation ladder (A7)
- Enable L1 → L2 → L3 one at a time, each behind the alignment tests (§11.3).
- **Gate:** alignment tests pass; validation does not regress.

### Milestone 9 — Final comparison
- Freeze the best configuration; run all required experiments with ≥ 3 seeds; evaluate **test once**.
- Produce the report table, per-class deltas, bootstrap CIs, error analysis, and resource costs.
- **Gate:** go/no-go gates in §2.4 evaluated and recorded.

### Milestone 10 — Decide whether to proceed
Proceed to richer methods (Tier 3, online refinement, Phase C at scale) only if adaptive distillation provides a measurable benefit that justifies its compute, storage, and complexity. Otherwise adopt the simplest variant that passed its gate (e.g., fixed weights, or GT-only).

---

## 17. Optional Extensions

### 17.1 Phase C — Unlabeled images (E8)
- Run teachers on unlabeled data; keep only instances above class-specific calibrated confidence (`q_hat ≥ q_min`) and, for added safety, agreement between A and B.
- Train with GT on labeled data plus weighted pseudo-labels on unlabeled data; mix labeled/unlabeled in each batch at a fixed ratio.
- Evaluate on the same untouched test set; compare against E6 without unlabeled data to show whether extra images help.

### 17.2 Iterative refinement (only after v2 passes)
Use the best student to re-audit disagreements and quarantined detections, fix GT errors via normal annotation, regenerate reliability, and retrain. This is a label-quality loop, not a license to train on test data.

### 17.3 Class-specific teacher routing
If one teacher is dominant on a subset of classes, consider distilling those classes from that teacher only (a hard `s ∈ {0,1}` per class) — a special case already available via `T → 0`.

### 17.4 Student capacity study
Repeat E2/E6 with two student sizes. Distillation benefits typically depend on the capacity gap between teacher and student; this informs whether to deploy a smaller student.

---

## 18. Recommended First-Version Defaults

Starting recommendations, not universal truths.

| Component | First-version choice | Reason |
|---|---|---|
| Teacher updates | Frozen | Stable targets; no backprop cost |
| Teacher inference | Offline, sequential | Lower peak memory |
| Cache | **Tier 1 + Tier 2 fields**, sharded, indexed by image ID | Small, architecture-agnostic, augmentation-safe |
| Cache confidence floor | Low (0.05), stricter at use time | Enables calibration |
| Reliability | Validation-derived per-class, recall-aware, shrunk (`m=20`) | Robust to small classes |
| Weight form | `w = g·s·ω` with smooth gate, softmax shares (`T` tuned) | Separates absolute vs. relative trust |
| Per-instance weighting | GT **veto** only (asymmetry rule) | Avoids memorization inflation |
| Fusion | Cluster by mask IoU; GT arbitrates conflicts | Consistent targets |
| Ignore policy | Vetoed, quarantined, and teacher-absence regions excluded | "Absence is not evidence" |
| Student GT loss | Always on | Trusted anchor |
| `λ` | Warm-up then constant; tune on val | Avoid early noise domination |
| Augmentation | L0 first; ladder with alignment tests | Correctness first |
| Evaluation | E0–E6 + controls C1–C5; ≥ 3 seeds; paired bootstrap | Credible conclusions |
| Test set | Final evaluation only | Prevents leakage |

---

## 19. Final Recommendation

Implement in this order:

1. **Audit teachers, splits, and provenance.**
2. **Cache Tier 1/2 teacher outputs with a validated manifest.**
3. **Train and lock the GT-only baseline (≥ 3 seeds); measure noise and per-class `r_ref`.**
4. **Add single-teacher distillation, then two-teacher fixed weights with fusion.**
5. **Add adaptive weights (`g·s·ω`) and prove them with controls C1–C3.**
6. **Stream only each batch's cached records during training.**
7. **Introduce augmentation behind alignment tests.**
8. **Evaluate against all baselines with identical metrics, seeds, and bootstrap CIs; test once.**
9. **Adopt the simplest variant that clears its gate.**

The defining principle:

> **The teachers provide guidance; GT remains the trusted target and can veto a teacher but never inflate its trust; measured, held-out reliability determines how much each teacher contributes; absence of a teacher prediction is never treated as evidence; and only the student's parameters are updated during student training.**

This design reduces repeated teacher inference and improves reproducibility. It does not guarantee higher accuracy or remove every memory bottleneck; those claims must be verified with controlled experiments and resource measurements.

---

## Appendix A — Core Training Logic (Conceptual)

```python
# ---------- Phase A: offline preparation ----------
for teacher in [teacher_a, teacher_b]:
    teacher.eval(); freeze_parameters(teacher)
    for images, gt, image_ids in dataset_loader:          # train + val (+calib)
        with no_grad():
            preds = teacher(images)                       # low conf floor
        matches = match_to_gt(preds, gt, tau_match)       # §7.3 (records FN too)
        cache_writer.write(image_ids, preds, matches)     # raw per-teacher records
    teacher.to("cpu"); free_accelerator_memory()

# reliability (validation / calibration data only)
r      = class_reliability(matches_val)                   # recall-aware, §7.2
r_hat  = shrink(r, n, r_global, m)                        # §7.4
q_hat  = fit_calibration(matches_val)                     # §7.5
g, s   = gate_and_share(r_hat, r_ref, T, s_min)           # §8.2
omega  = instance_weights(matches_train, tau_low)         # veto/quality, §8.2(c)
write_reliability_artifacts(r_hat, q_hat, g, s, omega)

# ---------- Phase B: student training ----------
student = initialize_from_scratch(seed)
for step, (images, gt, ids) in enumerate(train_loader):
    ta, tb = cache_a.read(ids), cache_b.read(ids)
    t = fuse(ta, tb, g, s, omega)                         # §9 (or read precomputed)
    images, gt, t = aligned_augment(images, gt, t)        # §11
    out = student(images)
    loss = supervised_loss(out, gt) \
         + lam(step) * distill_loss(out, t, ignore=t.ignore)   # §10
    optimizer.zero_grad(set_to_none=True)
    loss.backward(); optimizer.step()
```

All functions must be implemented for the chosen architecture and label format; this is conceptual pseudocode.

---

## Appendix B — Questions to Resolve Before Coding (with design impact)

| # | Question | Decision it changes |
|---|---|---|
| 1 | Same or different teacher architectures? | Whether Tier 3 (dense) is even possible; Tier 1/2 remain valid either way |
| 2 | Segmentation, detection, classification, or mixed? | Which of mask/box/class distillation and which reliability scores apply (§7.2) |
| 3 | Same classes and label mapping? | Need for an explicit conversion table; cluster rules (§9) |
| 4 | Which outputs to distill (probabilities, boxes, masks, logits)? | Cache tier (§6.2) and loss form (§10.3) |
| 5 | Sizes of train/val/test sets? | Shrinkage `m`, `n_min`, number of seeds feasible, bootstrap power |
| 6 | Image resolution and number of classes? | Cache size, mask encoding, per-class sample counts |
| 7 | RAM, disk, GPU budget? | Tier choice, shard size, batch size, worker count |
| 8 | Was a validation set unseen by both teachers? | Whether held-out reliability is trustworthy (§5.1) or a new calibration split is needed |
| 9 | Authoritative metrics? | Primary metric and gate thresholds (§2.4–2.5) |
| 10 | Random geometric augmentation planned? | Whether Tier 3 is excluded; ladder in §11 |
| 11 | **(New)** Is annotation quality known to be imperfect (missing objects)? | How aggressively to use quarantine/review (§8.6) |
| 12 | **(New)** Is extra unlabeled data available? | Whether Phase C (§17.1) is worth planning now |
| 13 | **(New)** Is there a deployment constraint (student size/latency)? | Student architecture and the capacity study (§17.4) |

Resolve these before committing to a cache schema or loss implementation.

---

## Appendix C — Suggested Code Layout and Interfaces

```text
src/
├── data/            manifest.py, dataset.py, transforms.py        # aligned_augment lives here
├── teachers/        infer.py (Phase A), cache_writer.py, cache_reader.py
├── reliability/     matcher.py, metrics.py, calibrate.py, weights.py (g, s, ω)
├── fusion/          cluster.py, fuse.py
├── losses/          gt_loss.py, distill_loss.py
├── train/           train_student.py, schedule.py
├── eval/            evaluate.py (single shared implementation), bootstrap.py
└── tools/           validate_cache.py, estimate_cache_size.py, make_report.py
```

Suggested CLI (each step idempotent and keyed by config hash):

```text
python -m tools.estimate_cache_size --config configs/teacher_a.yaml --sample 200
python -m teachers.infer           --config configs/teacher_a.yaml
python -m tools.validate_cache     --cache teacher_cache/
python -m reliability.weights      --config configs/reliability.yaml
python -m train.train_student      --config configs/student.yaml
python -m eval.evaluate            --run <run_id> --split val
python -m eval.evaluate            --run <final_run_id> --split test    # logged, run once
python -m tools.make_report        --runs e2,e3,e4,e5,e6,c1,c2,c3
```

Key interface contracts:

```python
class TeacherRecord:        # read from cache
    image_id: str
    instances: list         # class, conf, box, mask(RLE/prob), class_probs(optional)
    match_status: list      # TP / FP_class / FP_loc / FP_unmatched (labeled images)
    omega: list             # per-instance weight (0 for vetoed)
    missed_gt: list         # FN objects (never dropped)

def distill_loss(student_out, fused_targets, ignore) -> (loss, stats): ...
# stats must include: effective_mass (Σw), per_class_mass, n_vetoed, n_ignored_pixels
```

---

## Appendix D — Test Checklist

**Unit tests**
- Matcher: perfect match, class swap, duplicate predictions, empty prediction, empty GT, crowded overlap.
- Shrinkage: `n=0` returns global; `n→∞` returns raw.
- Gate/share: both teachers weak → `g=0`; one teacher dominant with `T→0`; shares sum to 1 over usable teachers.
- Veto: `FP_class` instance has `ω=0`; `TP` weight never exceeds class-level reliability.
- Loss reduction: zero trusted instances → finite loss; scaling `w` scales loss linearly (for fixed-constant normalization).
- Ignore masks: teacher-absence pixels contribute zero gradient to distillation.
- Cache: write→read round-trip; NaN rejection; manifest-mismatch hard error; empty vs. missing record.

**Integration tests**
- 20-image overfit run: student loss decreases; teacher params unchanged (hash before/after).
- Augmentation golden test (§11.3).
- Evaluation module reproduces a known teacher's published metric within tolerance.
- Determinism: same seed → same first 10 loss values.

**Sanity experiments (cheap, high value)**
- Set all `w = 0`: result must equal E2 within seed noise.
- Replace teacher targets with GT itself ("perfect teacher"): the student should match or exceed E2 quickly; if not, the distillation plumbing is broken.
- Replace teacher targets with random instances: should hurt; confirms the loss is wired to the targets.

---

## Appendix E — Formula Quick Reference

```text
r_hat[k,c]   = (n·r + m·r_global) / (n + m)                         shrinkage
g[c]         = 0 if r_best<r_ref else clip((r_best−r_ref)/(r_max−r_ref), g_min, 1)
s[k,c]       = softmax_k(r_hat[k,c]/T)  (zero for teachers below r_ref; floor s_min)
ω            = 0 (FP_class, FP_loc, FP_unmatched);  clip((IoU−τ_low)/(1−τ_low), ω_min, 1) for TP
w[k,i,j]     = g[c] · s[k,c] · ω[k,i,j]
L_total      = L_GT + λ(t) · Σ_k Σ_j w[k,i,j] · ℓ_distill(student, target_k,j) / N_norm
P_fused(x)   = Σ_k w_k·P_k(x) / Σ_k w_k
```

---

## Appendix F — Glossary of Added Concepts

- **Veto:** GT contradicting a teacher instance, setting its weight to 0.
- **Asymmetry rule:** GT may remove trust but may not add trust beyond held-out reliability.
- **Quarantine:** holding back unmatched teacher detections (possible missing labels) for review.
- **Effective distillation mass:** `Σ w`; how much teacher guidance actually reaches the loss.
- **Shrinkage:** pulling small-sample estimates toward a global estimate.
- **Provenance table:** record of which data each teacher was trained on.
