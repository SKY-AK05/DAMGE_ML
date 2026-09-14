# Mask2Former Car-Parts Evaluation Pipeline

Inference, annotation, and evaluation pipeline for the fine-tuned
Mask2Former car-parts segmentation model.

## What it does

1. Runs your trained Mask2Former checkpoint on every image in a directory
2. Draws predicted segmentation masks and class labels on each image
3. Loads your CVAT 1.1 ground-truth XML and normalises the label names to
   match the model's 32 snake_case classes
4. Matches predictions to GT instances with greedy IoU matching
5. Computes per-class and overall precision / recall / F1 / IoU / Dice
6. Builds a confusion matrix showing which car parts are confused with one another
7. Flags each error as one of four distinct types so you can decide whether
   to fix the model or fix the annotation
8. Saves a self-contained `summary_report.html` with all results

---

## Project layout

```
mask2former_eval/
    mask2former_eval.py   <- main script (run this)
    utils.py              <- label maps, XML loader, metrics, drawing helpers
    report_generator.py   <- HTML report builder
    README.md             <- this file
```

All outputs go to `annotation_cleaning_pipeline/outputs/` by default.

---

## 1. Dependencies

### Install once

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.38.0
pip install Pillow opencv-python-headless numpy
```

> If you do not have a GPU, replace the `--index-url` line with the CPU build:
> ```powershell
> pip install torch torchvision
> ```

### Full pinned requirements

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.38.0
Pillow>=10.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
```

---

## 2. Model checkpoint

The script defaults to the best-epoch checkpoint that is already in your
project:

```
analytics_export/analytics_export/best_model/best/
    config.json
    model.safetensors
    preprocessor_config.json
```

Nothing needs to be downloaded or changed. The checkpoint is
`facebook/mask2former-swin-tiny-coco-instance` fine-tuned on your 32
car-part classes for 27 epochs (mAP50 = 0.6455).

---

## 3. Ground-truth XML

The script expects the CVAT 1.1 XML exported from your annotation tool.
By default it looks for:

```
GROUND/annotations.xml
```

The GT XML uses different label names than the model (e.g. `Bonnet` vs
`hood`, `ORVM` vs `left_mirror`). The pipeline maps them automatically
using a built-in table in `utils.py`. Any GT label that has no mapping is
skipped and reported as `unmapped` — it is never silently dropped or
relabelled.

### Full GT-to-model label mapping

| GT label (XML)            | Position attribute | Model label             |
|---------------------------|--------------------|-------------------------|
| Front Bumper              | Front / any        | front_bumper            |
| Rear Bumper               | Rear / any         | back_bumper             |
| Bonnet / Hood             | —                  | hood                    |
| Trunk Lid / Trunk         | —                  | trunk                   |
| Front Fender              | Left               | front_left_fender       |
| Front Fender              | Right              | front_right_fender      |
| Front Door                | Front-Left         | front_left_door         |
| Front Door                | Front-Right        | front_right_door        |
| Rear Door                 | Rear-Left          | back_left_door          |
| Rear Door                 | Rear-Right         | back_right_door         |
| Rear Quarter Panel        | Left / Right       | rear_quarter_panel      |
| Headlamp                  | Left               | front_left_light        |
| Headlamp                  | Right              | front_right_light       |
| Taillight                 | Left               | back_left_light         |
| Taillight                 | Right              | back_right_light        |
| Indicator                 | Left / Right       | indicator               |
| Fog Lamp                  | Left / Right       | fog_lamp                |
| ORVM                      | Left               | left_mirror             |
| ORVM                      | Right              | right_mirror            |
| Windscreen                | Front              | front_glass             |
| Rear Windscreen           | Rear               | back_glass              |
| Door Glass                | any                | door_glass              |
| Roof                      | —                  | roof                    |
| Wheel                     | any                | wheel                   |
| Wheel Cap                 | any                | wheel_cap               |
| Grille                    | —                  | grille                  |
| Fuel Lid                  | Left / Right       | fuel_lid                |
| Side Skirt / Rocker Panel | Left / Right       | side_skirt_rocker_panel |
| Number Plate              | Front / Rear       | number_plate            |
| Antenna                   | —                  | antenna                 |
| Pillar                    | Left Side / Right  | pillar                  |
| Tailgate                  | —                  | tailgate                |

To add a new mapping, edit the `_GT_LABEL_MAP` dict in `utils.py`.

---

## 4. Running the pipeline

### Simplest usage (all defaults, PowerShell)

```powershell
cd "C:\Users\Aakash\Documents\PROJECT WEB\orchvate\CAR_DAMAGE\700-cleaning\annotation_cleaning_pipeline\mask2former_eval"
python mask2former_eval.py
```

This automatically uses:
- Input images: `../GROUND/`
- Model: `../analytics_export/analytics_export/best_model/best/`
- GT XML: `../GROUND/annotations.xml`
- Output: `../outputs/`
- Confidence threshold: `0.5`
- IoU match threshold: `0.5`
- Device: `cpu`

### Override any path or setting

```powershell
python mask2former_eval.py `
    --input   "C:\...\GROUND" `
    --model   "C:\...\best_model\best" `
    --output  "C:\...\outputs" `
    --gt_xml  "C:\...\GROUND\annotations.xml" `
    --conf    0.45 `
    --iou_thresh 0.5 `
    --device  cuda
```

### Inference only (no GT evaluation)

```powershell
python mask2former_eval.py --no_gt
```

### Use GPU

```powershell
python mask2former_eval.py --device cuda
```

### Lower the confidence threshold to catch more instances

```powershell
python mask2former_eval.py --conf 0.35
```

### All command-line arguments

| Argument        | Default                        | Description                                              |
|-----------------|--------------------------------|----------------------------------------------------------|
| `--input`       | `../GROUND`                    | Image directory (searched recursively)                   |
| `--model`       | `../best_model/best`           | Checkpoint directory                                     |
| `--output`      | `../outputs`                   | Output root directory                                    |
| `--gt_xml`      | `<input>/annotations.xml`      | CVAT 1.1 ground-truth XML                                |
| `--no_gt`       | off                            | Inference-only, skip evaluation                          |
| `--conf`        | `0.5`                          | Confidence threshold (0–1)                               |
| `--iou_thresh`  | `0.5`                          | IoU threshold for GT matching (0–1)                      |
| `--device`      | `cpu`                          | `cpu` / `cuda` / `mps`                                   |
| `--verbose`     | off                            | Enable DEBUG logging                                     |

---

## 5. Output files explained

```
outputs/
    predicted_annotations/          <- model predictions drawn on original image
    ground_truth_annotations/       <- GT polygons drawn on original image
    comparison_images/              <- side-by-side GT (left) | Prediction (right)
    error_cases/
        <image>_errors.jpg          <- error annotations colour-coded by type
        <image>_review.txt          <- text inspection report per image
    metrics/
        overall_metrics.json        <- overall P/R/F1/IoU/Dice + TP/FP/FN counts
        per_class_metrics.csv       <- same metrics for each of the 32 classes
        per_image_summary.csv       <- per-image TP/FP/FN/WC counts
        confusion_matrix.csv        <- (32+1) x (32+1) matrix
    predictions.xml                 <- all predictions in CVAT 1.1 format
    summary_report.html             <- self-contained interactive HTML report
```

---

## 6. Interpreting the results

### Predicted annotation images

Each predicted mask is drawn with a semi-transparent coloured fill and a
solid border. The label shows the model's class name and its confidence
score (e.g. `hood 0.87`). Every class has a fixed colour defined in
`CLASS_COLOURS` in `utils.py`.

### Comparison images

Left panel shows the ground-truth polygons labelled `GT: <class>`.
Right panel shows the model predictions. Predictions with a correct mask
but wrong class are highlighted with a red border and the note
`should be <gt_label>`.

### Error case images

Three colours indicate three types of problem:

| Colour | Error type | Meaning |
|--------|-----------|---------|
| Red    | TP_WRONG_CLASS | Mask IoU passes threshold but predicted class differs from GT |
| Orange | FP_LOW_OVERLAP | Prediction has no GT instance with IoU >= threshold |
| Purple | GT_MISSED | GT instance was not matched by any prediction |

### Review text files (`error_cases/<image>_review.txt`)

Each file lists every match result for that image and classifies it as one
of four situations:

1. **CLASSIFICATION_ERROR** — the mask overlaps the GT polygon well
   (IoU >= threshold) but the labels differ.
   - If the IoU is high (>= 0.7): inspect the GT label. The model may be
     correct and the annotation may have used the wrong label.
   - If IoU is moderate: either the class or the boundary could be wrong.
   - **Never assume the model is wrong just because it disagrees with GT.**

2. **BOUNDARY_ERROR** — correct class, but IoU is between 0 and the
   threshold. The polygon shape needs improvement.

3. **FALSE_POSITIVE** — the model predicted an instance with no
   corresponding GT polygon. Either the model hallucinates, or the GT is
   missing an annotation.

4. **MISSED_DETECTION** — a GT polygon has no matching prediction. Either
   the model failed to detect the part, or confidence threshold is too high.

### overall_metrics.json

```json
{
  "tp": 312,
  "fp": 88,
  "fn": 64,
  "wrong_class": 21,
  "precision": 0.780,
  "recall": 0.830,
  "f1": 0.804,
  "mean_iou": 0.712,
  "mean_dice": 0.823,
  "conf_threshold": 0.5,
  "iou_threshold": 0.5,
  "total_images": 110,
  "images_with_gt": 110
}
```

- `wrong_class` counts predictions where the mask overlaps GT but the
  class label is different. These are classification errors, not boundary
  errors.
- `fp` counts predictions that found no matching GT instance at all.
- `fn` counts GT instances that no prediction was matched to.

### per_class_metrics.csv

One row per class, sorted by class ID. Key columns:

| Column | Meaning |
|--------|---------|
| `precision` | Of all predictions for this class, what fraction were correct |
| `recall` | Of all GT instances of this class, what fraction were detected |
| `f1` | Harmonic mean of precision and recall |
| `mean_iou` | Average IoU over correctly-matched instances only |
| `wrong_class` | Times the model mask matched GT but used the wrong class label |

### confusion_matrix.csv

Rows = GT class, Columns = predicted class.

- **Diagonal cells** (green in HTML): correct predictions.
- **Off-diagonal cells** (red in HTML): the model predicted column-class
  when the GT was row-class.
- **Last row** (`NO_GT`): predictions with no matching GT instance.
- **Last column** (`UNMATCHED`): GT instances that received no prediction.

Large off-diagonal values tell you exactly which pairs of car parts the
model confuses. For example, a high value at row `grille`, column
`front_bumper` means the model frequently predicts `front_bumper` on a GT
polygon labelled `grille`.

---

## 7. Answering the main question

> How accurately does my model segment car parts, and how much are results
> affected by incorrect GT annotations?

Work through these steps with the output files:

**Step 1 — Look at overall_metrics.json.**
If F1 >= 0.75 and mean_iou >= 0.65, the model is performing well overall.
Check `wrong_class` vs `fp`: if `wrong_class` is large relative to `fp`,
classification errors dominate over hallucinations.

**Step 2 — Open per_class_metrics.csv.**
Classes with low recall but acceptable IoU on matched instances suggest the
model finds parts correctly when it detects them but misses many. Classes
with low precision suggest false positives or frequent wrong-class errors.

**Step 3 — Read the confusion_matrix.**
If grille / indicator / fog_lamp show large off-diagonal values, the model
is confusing visually similar parts. This suggests the model needs more
diverse training data for those classes, not just label corrections.

**Step 4 — Open comparison images for the weak classes.**
Look at `comparison_images/` for images where those classes appear. Ask:
does the model's polygon look like the right part in the right position?
If yes, and GT says something different, suspect a GT error.

**Step 5 — Read the `_review.txt` files for CLASSIFICATION_ERROR cases.**
For each entry with high IoU (>= 0.7) and a wrong class, the evidence
strongly suggests either:
  - The model learned the correct visual pattern but GT used a legacy or
    inconsistent label name, **or**
  - The model consistently confuses two visually similar parts at that
    location.

You cannot decide which is true from metrics alone — visual inspection is
required. The pipeline deliberately does not correct GT automatically.

**Step 6 — Decision point.**
- If most errors come from `wrong_class` with high IoU: fix GT labels first,
  then re-evaluate before retraining.
- If errors come from `fp` / `fn` with low IoU: the model boundary quality
  or detection coverage needs improvement — consider training with more
  augmentation or a larger backbone.
- If a small set of classes (e.g. grille, indicator, antenna) dominate errors:
  targeted data collection for those classes will give more return than a
  full retrain.

---

## 8. Adding a new GT label mapping

If your GT XML contains a label not in the mapping table, the pipeline logs
a warning and skips those polygons from evaluation. To add it:

1. Open `utils.py`.
2. Find the `_GT_LABEL_MAP` dict.
3. Add a new entry:
   ```python
   ("your gt label name lowercase", "position_lowercase"): "model_snake_case_label",
   ```
4. Re-run the pipeline — no other changes needed.

---

## 9. Importing predictions back into CVAT

The script saves `outputs/predictions.xml` in CVAT 1.1 format. You can
import this into a CVAT task to view predictions alongside your annotations:

1. Open your CVAT task.
2. Actions > Upload annotations > CVAT 1.1 format.
3. Select `predictions.xml`.

This lets you visually compare model predictions and GT side-by-side in
the annotation tool.
