# Car Parts Segmentation Project — Full Journey Log

## 1. The Original Goal

Build a computer vision model that can detect **all car body parts** in an image
with very high accuracy (~98% target), for use cases like automated vehicle
inspection / insurance claims. The taxonomy needed is fine-grained — not just
"door" or "wheel," but specific variants like `front_left_door` vs
`back_left_door`, `left_mirror` vs `right_mirror`, `wheel_cap`, `fuel_lid`,
`pillar`, etc. — with position/side clearly identified.

Two sub-problems emerged early and shaped everything after:

1. **We don't have a large labeled dataset of our own car images.**
2. **Models often confuse visually similar but positionally different parts**
   — e.g., swapping a front door for a rear door, or left mirror for right
   mirror — because these parts can look nearly identical in isolation.

---

## 2. Stage 1 — Exploring How to Get to 98% Accuracy

**Question asked:** Is 98% accuracy achievable, and how?

**Conclusion reached:** No single off-the-shelf model hits 98% out of the box.
The realistic path is a **two-stage pipeline**:
- Use a zero-shot foundation model (Grounding DINO + SAM 2) to auto-label data
  with no manual annotation required.
- Fine-tune a lightweight production model (YOLO-seg, or a transformer-based
  segmenter) on that auto-labeled data.

We also discussed several *other* levers besides model architecture:
- Multi-view fusion (photograph the car from multiple angles to resolve
  left/right and front/rear ambiguity) — identified as probably the single
  highest-leverage, lowest-cost fix.
- Active learning (only manually fix the model's low-confidence predictions,
  not everything).
- Synthetic data from 3D car models rendered in Blender, with perfect
  automatic labels since part identity is known from the 3D mesh.
- Relationship-aware models (see Stage 3 below) instead of pure data volume.

**Key realization:** More data alone does **not** fix front/rear or
left/right confusion. If a cropped door looks identical whichever side it's
on, more repeated examples of the same ambiguous crop don't add new
information — the model needs *contextual/positional* information (what's
near the part), not just more images.

---

## 3. Stage 2 — Building the Auto-Annotation Pipeline (Grounding DINO + SAM 2)

**Plan:** Grounding DINO (text-prompted object detector) finds each car part
via natural-language prompts ("front bumper," "left mirror," etc.) and
outputs bounding boxes. SAM 2 takes those boxes and produces exact pixel-level
polygon masks. Together this auto-labels raw car images without manual
annotation.

**What we built:** `auto_annotate_carparts.py` — reads images from a folder,
runs DINO → SAM2, and writes out a **CVAT-format `annotations.xml`** (matched
exactly to a sample CVAT export the user uploaded, including the `Position`
attribute schema for left/right/front/rear variants). The script also asks
the user via terminal prompt whether to save DINO's raw visualized boxes
(for QA), while always saving the final SAM2 polygon annotations.

**Supporting scripts added:**
- `setup.sh` / later merged into `setup_all.sh` — creates a Python venv,
  installs CUDA PyTorch, clones and installs Grounding DINO + SAM 2, and
  auto-downloads all model weights.
- `weights_downloader.py` — downloads Grounding DINO and SAM2 checkpoints,
  skipping any already present.

**Problem discovered when testing on real images:** Screenshots of DINO's
raw box output and SAM2's polygon output showed real accuracy issues —
overlapping/misfiring boxes (e.g. "Wheel Cap" and "Fog Lamp" stacked on the
same region), and SAM2 merging multiple body panels into a single blob
instead of separate per-part polygons.

**Diagnosis:** The issue wasn't SAM2's boundary-finding (it reliably follows
real visual edges/seams in the car body) — it was **DINO's text-to-region
matching**, which had no way to reason about part *position* relative to
other parts (wheels, mirrors, bumpers). This is a labeling/classification
problem, not a segmentation problem.

**Proposed fixes** (not yet implemented in code): more specific/mutually
exclusive text prompts, per-class confidence thresholds, class-agnostic NMS
across all detected boxes, crop-and-repeat for small parts, and geometric
post-filtering rules (e.g. "Wheel Cap" must be inside a "Wheel" box).

---

## 4. Stage 3 — Why Some Models Understand Part Relationships and Others Don't

**Question asked:** Why did YOLO correctly separate front/rear doors in a
CVAT-style example when Grounding DINO struggled with the same distinction?

**Answer worked out:**
- **YOLO** was trained on human-labeled ground truth, so it directly learned
  the visual distinction. It also **accidentally** learns positional bias
  (e.g. "front door" labels tend to appear on the left side of training
  images) purely from repeated exposure — not because it explicitly reasons
  about position.
- **Grounding DINO** is zero-shot — it was never trained on car parts
  specifically, so it's guessing from general language-image similarity,
  with no learned understanding of car-specific left/right conventions.

**Broader distinction identified — relationship-aware vs. non-relationship-aware models:**

| Model | Understands part relationships? | Why |
|---|---|---|
| YOLOv8/v11-seg | No (mostly local receptive field) | Learns positional bias only by accident, not by design |
| Mask2Former | Yes | Transformer self-attention lets every detected part "see" every other part before finalizing its label |
| MaskDINO | Yes | Same attention mechanism, DETR-family, from the Grounding DINO team |
| OneFormer | Yes | Newer unified transformer segmentation model, often stronger than Mask2Former |

This led directly to the decision to **empirically test** this claim rather
than assume it.

---

## 5. Stage 4 — Deciding to Run a Controlled Model Comparison

**Decision made:** Instead of debating architecture choice further, train
multiple models on the same labeled, public datasets and directly measure
how often each one confuses `front_left_door` vs `back_left_door`,
`left_mirror` vs `right_mirror`, etc. This removes "our own annotation
quality" as a variable and isolates architecture as the cause.

**Datasets chosen:**
- **Ultralytics Carparts-seg** — 3,833 images, 23 classes, pre-split
  train/val/test, YOLO-seg format (ships with official YAML).
- **DSMLR Car-Parts-Segmentation** — 500 images, no official split (we wrote
  an auto-splitter, 80/10/10, seeded for reproducibility).

**Models selected for comparison — 4 total:**
1. **YOLOv8/v11-seg** (Ultralytics) — the "no relationship understanding" baseline.
2. **Mask2Former** (HuggingFace `facebook/mask2former-swin-tiny-coco-instance`).
3. **OneFormer** (HuggingFace `shi-labs/oneformer_coco_swin_large`) — added later as a 4th model, since it's currently one of the strongest transformer segmentation models and tells us if Mask2Former/MaskDINO are near the practical ceiling.
4. **MaskDINO** (IDEA-Research, Detectron2-based) — used in place of plain Deformable DETR, because plain Deformable DETR only outputs bounding boxes, not polygon masks, so it can't be fairly compared on segmentation quality.

**Scripts built for this stage:**
- `download_and_prepare_datasets.sh` — downloads both datasets, unzips, calls the split + conversion scripts.
- `prepare_dsmlr_split.py` — auto-splits DSMLR 80/10/10.
- `yolo_to_coco.py` — converts YOLO-seg polygon format to COCO JSON (needed by Mask2Former/OneFormer/MaskDINO).
- `train_yolo_seg.py` — trains YOLO-seg, GPU auto-detect.
- `train_mask2former.py` — fine-tunes Mask2Former via HuggingFace `transformers`.
- `train_oneformer.py` — fine-tunes OneFormer via HuggingFace `transformers`.
- `register_carparts_dataset.py` + `train_maskdino.sh` — registers datasets with Detectron2, clones/builds MaskDINO (including its CUDA deformable-attention ops), and launches training.
- `evaluate_confusion_matrix.py` — runs each trained model on the test set, matches predictions to ground truth via IoU, builds a full confusion matrix, and specifically prints out mixup rates for the exact pairs we care about (`front_left_door` vs `back_left_door`, `left_mirror` vs `right_mirror`, etc.).
- `compare_all_models.py` — loads all four models' saved confusion matrices and prints one final side-by-side comparison table (overall accuracy + mixup rate per pair, per model).
- `setup_all.sh` — master install script covering **both** Stage 2 (DINO+SAM2) and Stage 4 (YOLO/Mask2Former/OneFormer/MaskDINO) dependencies in one place; also does a GPU auto-detect and a final sanity-check import test for every component.
- `requirements.txt` — plain pip-installable subset, for anyone who wants to manage the venv manually instead of running the full shell script.

---

## 6. Stage 5 — Real-World Setup Problems Encountered on the GPU Server

This is the part that ate the most time — not modeling decisions, but basic
environment issues. Logged here in the order they actually happened, since
each one caused a different failure downstream.

### Issue 1: Wrong Python / pip being used (root cause of most early failures)
**Symptom:** `ImportError: .../libtorch_cuda.so: undefined symbol: ncclCommResume`
on every script that imported `torch`.
**Diagnosis:** `which python` → `/usr/bin/python`, `which pip` →
`/home/orchvate/.local/bin/pip`, and `$VIRTUAL_ENV` was empty. The venv had
never actually been created/activated — everything had been installing into
the user-level (`--user`) site-packages instead, which had a broken/mismatched
CUDA build of PyTorch.
**Fix:** Explicitly create and activate the venv, uninstall any stray user-level
torch, and reinstall PyTorch with the correct CUDA index URL *inside* the venv:
```bash
python3 -m venv venv && source venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Confirmed working: `Torch: 2.5.1+cu121 | CUDA: True`.

### Issue 2: Shell scripts not executable
**Symptom:** `bash: ./setup_all.sh: Permission denied`.
**Fix:** `chmod +x setup_all.sh download_and_prepare_datasets.sh train_maskdino.sh`.

### Issue 3: GroundingDINO / SAM2 / Detectron2 build failures — `No module named pip`
**Symptom:** `pip install -e .` for GroundingDINO failed deep inside a
pip build-isolation subprocess, ultimately erroring with
`ModuleNotFoundError: No module named 'torch'` and
`/venv/bin/python3: No module named pip`.
**Diagnosis:** GroundingDINO's `setup.py` tries to run `pip install torch` as
a subprocess *inside pip's isolated build sandbox*, which doesn't have pip
available in it — a known packaging quirk of that repo, not a real
dependency problem (torch was already installed correctly in the venv).
**Fix:** Install with build isolation turned off, since torch is already
present in the venv:
```bash
pip install -e . --no-build-isolation
```
Applied this fix to every git-based editable install in the pipeline:
GroundingDINO, SAM2, Detectron2, and MaskDINO's `requirements.txt` install.
All three scripts (`setup_all.sh`, `train_maskdino.sh`) were patched
permanently with this fix.

### Issue 4: `weights_downloader.py` missing on the server
**Symptom:** `python: can't open file '.../weights_downloader.py'`.
**Diagnosis:** The file had been shared in chat but never actually saved
onto the remote GPU server — only the commands referencing it had been run.
**Resolution:** Deferred — this file is only needed for Stage 2
(DINO+SAM2 auto-annotation of the user's *own* raw images), not for the
Stage 4 model comparison, so it was safely skipped for now and flagged to
revisit later.

### Issue 5: Cascading failures from mixing `&&` and `;` in one chained command
**Symptom:** A long chained command used `&&` between the early setup steps
but `;` between later steps (so later steps kept running even after earlier
ones failed) — resulting in five different scripts failing one after another
with unrelated-looking errors (`ModuleNotFoundError: No module named
'ultralytics'`, HuggingFace `HFValidationError` on a local path being
mistaken for a hub repo ID, `No module named 'detectron2'`), all really
downstream symptoms of `setup_all.sh` dying early.
**Fix:** Switched every chained command to use `&&` throughout, so the whole
pipeline **stops cleanly at the first real failure** instead of limping
through later steps with a broken environment and producing confusing
cascade errors.

### Issue 6: Confusing a stale/appended log file for a live error
**Symptom:** A fresh log file appeared to show a GroundingDINO build error
that had already been fixed several steps earlier.
**Diagnosis:** The error text in the log was old content from a previous
run, not a new failure — confirmed by directly testing
`python -c "import groundingdino"`, which succeeded immediately.
**Lesson applied going forward:** Always start log files fresh with `>` (or
`rm -f` first), never append with `>>`, when debugging live.

### Issue 7: Nothing had actually been trained yet
**Symptom:** `ls runs_comparison/...` and `tree` showed no `datasets/` or
`runs_comparison/` folders at all — despite several "training" commands
having apparently been run.
**Diagnosis:** `download_and_prepare_datasets.sh` had never successfully
completed in any prior attempt (it was always short-circuited by an earlier
`&&`-chained failure), so every downstream training/evaluation command had
been silently operating on nonexistent data and failing immediately.
**Fix:** Ran dataset prep on its own first, confirmed it fully completed
(3,156 train / 401 val / 276 test images for Carparts-seg; 320/40/40 for
DSMLR), *then* proceeded to training.

### Issue 8: venv silently deactivated inside a new `tmux` session
**Symptom:** After correctly fixing the torch/venv issue earlier, a
`pip install ultralytics` inside a new `tmux new -s train` session
re-installed into the broken user-level location again, and the prompt no
longer showed `(venv)`.
**Diagnosis:** Starting a new `tmux` session starts a fresh shell that does
**not** inherit the parent shell's activated venv — `source venv/bin/activate`
must be re-run inside every new terminal/tmux session.
**Fix:** Always confirm `(venv)` appears in the prompt (or check
`which python` points inside `./venv/bin/python`) before running any pip
install or script, especially right after opening a new terminal/tmux pane.
**Lasting habit adopted:** Check the prompt for `(venv)` before running
anything, every single time a new session is opened.

---

## 7. Where We Are Right Now

- Environment is fully set up correctly inside the venv: PyTorch 2.5.1+cu121
  with CUDA confirmed working on an NVIDIA A10-12Q GPU.
- GroundingDINO, SAM2, `ultralytics`, and `detectron2` are all installed and
  import-tested successfully.
- Both datasets (Ultralytics Carparts-seg, DSMLR) are downloaded, split, and
  converted to COCO format.
- The full training + evaluation + comparison pipeline is currently running:
  YOLOv11m-seg (20 epochs) → Mask2Former (10 epochs) → OneFormer (10 epochs)
  → MaskDINO → confusion-matrix evaluation for all four → final side-by-side
  comparison table via `compare_all_models.py`.
- Running inside `tmux` (`session: train`) so it survives disconnects, with
  full output logged live to `full_pipeline_log.txt` via `tee`.

## 8. What Happens After This Run Finishes

1. Review `compare_all_models.py`'s output table — overall accuracy and,
   specifically, the front/rear and left/right mixup rate per model.
2. Pick the strongest-performing architecture for the actual confusion
   problem (expected candidates: Mask2Former, OneFormer, or MaskDINO, based
   on their relationship-aware attention mechanism — to be confirmed by
   real numbers, not assumption).
3. Return to Stage 2 (Grounding DINO + SAM2 auto-annotation) to label the
   user's own raw car images, improving prompt specificity, per-class
   thresholds, NMS, and geometric post-filtering as previously identified.
4. Fine-tune the winning architecture from Step 2 on the now auto-labeled
   (and lightly human-corrected) dataset of the user's own images as the
   final production model.

---

## 9. Results — YOLO vs Mask2Former (final 2-model comparison)

**OneFormer and MaskDINO were dropped before final evaluation** — both hit
genuine, unrelated environment/architecture blockers rather than being
ruled out on merit:
- **OneFormer:** required real text-query supervision as part of its
  training loss (a contrastive image-text loss baked into the architecture),
  which our COCO-style data pipeline never supplied — properly supporting
  it would need replicating OneFormer's official text-tokenization
  pipeline, a nontrivial addition, not a quick fix. It also separately hit
  a hard CUDA out-of-memory wall on `swin_large` even at batch size 1 on
  the available 12GB GPU.
- **MaskDINO:** its deformable-attention CUDA kernels require compiling
  with `nvcc` from a matching CUDA Toolkit version. The GPU server only had
  the NVIDIA *driver* installed, not the toolkit/compiler, and the
  system-package version available via `apt` (CUDA 11.5) was mismatched
  against the installed PyTorch build (CUDA 12.1) — installing a matching
  toolkit locally was judged not worth the additional setup time and risk
  on a shared server for a comparison that already had a clear answer from
  two models.

**Final test-set results (Ultralytics Carparts-seg, 276 test images, 23 classes):**

| Metric | YOLOv11m-seg | Mask2Former |
|---|---|---|
| Overall accuracy | 81.6% | 83.2% |

At the aggregate level the two models look almost equivalent — only a 1.6
point gap. **The real difference only shows up inside the confusion
matrix, specifically on left/right mirror-image part pairs:**

| Confusion pair | YOLO mixup rate | Mask2Former mixup rate | Improvement |
|---|---|---|---|
| front_left_door ↔ front_right_door | 44.1% | 4.4% | ~10x better |
| back_left_door ↔ back_right_door | 42.2% | 1.6% | ~26x better |
| front_left_light ↔ front_right_light | 55.1% | 3.4% | ~16x better |
| left_mirror ↔ right_mirror | 32.2% | 4.2% | ~8x better |
| back_left_light ↔ back_right_light | 50.0% | 30.8% | ~1.6x better (still weak for both) |
| front_left_door ↔ back_left_door | 1.3% | 0.0% | both already strong |
| front_right_door ↔ back_right_door | 0.0% | 0.0% | both already strong |

**Conclusion — the original hypothesis was confirmed with real numbers:**
- **Front/rear confusion was never actually YOLO's weak point** — both
  models handle it well (near 0%), because front and rear parts have
  strong distinguishing visual cues (different light shapes, different
  proportions, different context).
- **Left/right mirror-image confusion is exactly where the architectures
  diverge**, and diverge hard. YOLO gets left vs right wrong on doors,
  mirrors, and lights roughly as often as a coin flip (32-55% error rates),
  because it has no mechanism to reason about a part's position relative
  to the rest of the car — exactly the theoretical gap identified back in
  Stage 3. Mask2Former's transformer self-attention, which lets every
  detected part "see" every other detected part before finalizing its
  label, cuts these specific error rates by roughly 8-26x.
- **Aggregate accuracy alone would have hidden this finding** — the 1.6
  point overall gap looks minor until you break it down by confusion pair,
  where the real, actionable difference lives. This is a concrete
  practical lesson for evaluating any future model on this dataset: always
  check the confusion matrix on the specific pairs that matter, not just
  the headline metric.
- **One open problem remains even for the better model:**
  `back_left_light` vs `back_right_light` stays hard for both YOLO and
  Mask2Former (50.0% vs 30.8%) — worth flagging as unresolved rather than
  declaring total victory.

**Practical takeaway for Stage 4 (final production model choice):**
Mask2Former (or another relationship-aware/attention-based architecture)
is the clear choice over plain YOLO specifically because of how this
project's real confusion problem — telling left from right on
mirror-image car parts — behaves. This directly validates the reasoning
from Stage 3 with real experimental evidence rather than architecture
theory alone.

## 10. Next Step — Auto-Annotating the User's Own Test Images

With the architecture question settled, the project returns to Stage 2:\nrunning the Grounding DINO + SAM2 auto-annotation pipeline\n(`auto_annotate_carparts.py`) on a small batch of the user's own raw car\nimages (3 images in a local test folder) to validate the annotation\npipeline end-to-end before scaling up to a larger unlabeled dataset.

---

## 11. Pipeline Migration — 2026-08 (YOLO / Mask R-CNN / Fast R-CNN + Azure ML)

**Context:** This project was restructured away from the 5-model comparison
(YOLO / Mask2Former / OneFormer / MaskDINO / Mask R-CNN) toward a cleaner
3-model pipeline focused on production-readiness and Azure ML integration.

**Models kept:** YOLO11m-seg, Mask R-CNN, Fast R-CNN.

**Models removed:** Mask2Former, OneFormer, MaskDINO.
- Mask2Former / OneFormer: HuggingFace Transformers dependency removed; the
  experimental phase that motivated those models is complete (see Stage 9).
- MaskDINO: Detectron2 + nvcc compiler requirement made it impractical on
  the target Azure VM without a full CUDA Toolkit install.

**Key architectural changes:**
- Training moved from running locally on the GPU VM to submitting jobs via
  Azure ML SDK (`azure_train.py`), using the attached `gpu-vm-clean` compute.
- Dataset uploads managed via `upload_dataset.py` (Azure ML Data Asset).
- Rust orchestrator (`orchestrator/src/main.rs`) updated to reflect the
  3-model menu (YOLO / Mask R-CNN, with Fast R-CNN added via `train_fastrcnn.py`).
- Known bug fixed: orchestrator was passing `--output_dir` to `train_yolo_seg.py`,
  which only accepts `--project`. Fixed in `orchestrator/src/main.rs` line ~327.

---

## 12. Dataset Analytics Multi-Sheet Excel Export — 2026-09

**Feature Addition:** Added comprehensive dataset analytics export to multi-sheet Excel workbooks (`.xlsx`).

**Key capabilities added:**
- **Dedicated Exporter Utility:** `scripts/data/export_dataset_analytics.py` parses dataset annotations and split layouts (`coco_*.json` or YOLO text labels) and extracts original source origins (`raw`, `matched_carparts-seg`, `matched_dsmlr`, etc.) from image filename prefixes.
- **Multi-Sheet Workbook Layout:**
  1. `Overview & Summary`: High-level metrics, split image/annotation counts, source overview, and overall class distribution.
  2. `Class Matrix`: Full breakdown of annotation counts per category across `train`, `val`, and `test` splits.
  3. `Src - <source_name>`: Dedicated worksheet for every dataset source with source-specific split summaries and class distributions.
- **Orchestrator Integration:** `analyze_dataset()` in `orchestrator.py` automatically generates and saves `<dataset_name>_analytics.xlsx` whenever Dataset Analytics (Option 5) is run, reporting the file path in the terminal.

---

## 13. Combined Dataset CVAT XML Export & ZIP Archive — 2026-09

**Feature Addition:** Added dataset export to CVAT XML 1.1 format with automated ZIP packaging.

**Key capabilities added:**
- **Exporter Utility:** `scripts/data/export_dataset_xml.py` converts all combined dataset splits (`train`, `val`, `test`) into a single `combined_annotations.xml` file matching the CVAT XML 1.1 format of `RAW_DATASET/01_orchvate_custom/XML/combined_annotations.xml`.
- **Image Consolidation & ZIP Packaging:** Copies all combined images (~5.6k to 7k) into `<output_dir>/images/` and compresses the entire package into a `.zip` archive.
- **Orchestrator Integration:** Added dedicated Main Menu option `6) Export CVAT XML & ZIP` in `orchestrator.py` (menu choices 1–11).

---

## 14. Taxonomy Refinement: Front Fender Inclusion & 100% Raw Coverage — 2026-09

**Taxonomy Update:** 
- Promoted `front_fender` (1,055 annotations in `01_orchvate_custom`) into a dedicated target class (`front_fender`: Class ID 10 in 24-class taxonomy).
- Added mapping aliases for `wheel_cap` (2,213), `pillar` (1,298), and `antenna` (296) to `object` in `prepare_raw_dataset.py` and `remap_github_labels.py`.

**Key changes made:**
- **Zero Skipped Annotations:** **100% of all 29,870 annotations** in `RAW_DATASET/01_orchvate_custom/XML/combined_annotations.xml` are now preserved and converted (0 skipped).
- **Analytics Regeneration:** Re-processed raw dataset and refreshed `combined_carparts_analytics.xlsx` multi-sheet Excel report.
