# -*- coding: utf-8 -*-
"""
report_generator.py -- generate the summary_report.html from pipeline results.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_to_b64(path: Path) -> str:
    """Return a base-64 data URI for embedding an image inline in HTML."""
    ext = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    return f"data:{mime};base64,{data}"


def _tier_badge(f1: float) -> str:
    if f1 >= 0.80:
        return '<span class="badge excellent">Excellent &ge;80%</span>'
    elif f1 >= 0.60:
        return '<span class="badge good">Good &ge;60%</span>'
    elif f1 >= 0.40:
        return '<span class="badge fair">Fair &ge;40%</span>'
    else:
        return '<span class="badge weak">Weak &lt;40%</span>'


# ---------------------------------------------------------------------------
# Confusion matrix HTML table
# ---------------------------------------------------------------------------

def _cm_html(cm: np.ndarray, labels: List[str]) -> str:
    n = len(labels)
    header_labels = labels + ["UNMATCHED"]

    # Normalise row-wise for colour intensity
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm / row_sums

    rows = ['<table class="cm-table"><thead><tr><th>GT \\ Pred</th>']
    for lbl in header_labels:
        short = lbl.replace("_", "<br>")
        rows.append(f'<th title="{lbl}">{short}</th>')
    rows.append("</tr></thead><tbody>")

    for r, gt_lbl in enumerate(header_labels):
        rows.append(f"<tr><th>{gt_lbl}</th>")
        for c in range(n + 1):
            val  = int(cm[r, c])
            norm = float(cm_norm[r, c])
            # Green diagonal, red off-diagonal
            if r == c and r < n:
                bg = f"rgba(60,200,80,{norm:.2f})"
            elif val > 0:
                bg = f"rgba(220,60,60,{min(norm * 2, 1.0):.2f})"
            else:
                bg = "transparent"
            cell_cls = "diag" if r == c else ""
            rows.append(f'<td class="{cell_cls}" style="background:{bg}">{val if val > 0 else ""}</td>')
        rows.append("</tr>")

    rows.append("</tbody></table>")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Example thumbnails
# ---------------------------------------------------------------------------

def _example_thumbnails(image_paths: List[Path], caption: str, max_images: int = 6) -> str:
    if not image_paths:
        return ""
    html = [f'<h3>{caption}</h3><div class="thumb-row">']
    for p in image_paths[:max_images]:
        if p.exists():
            src = _img_to_b64(p)
            html.append(
                f'<figure><img src="{src}" class="thumb" '
                f'title="{p.name}" /><figcaption>{p.stem[:20]}</figcaption></figure>'
            )
    html.append("</div>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# CSS + JS
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0;
       line-height: 1.6; padding: 20px; }
h1 { color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 8px; margin-bottom: 20px; font-size: 1.8em; }
h2 { color: #a8d8ff; margin: 30px 0 10px; font-size: 1.3em; border-left: 4px solid #00d4ff; padding-left: 10px; }
h3 { color: #c8e6ff; margin: 20px 0 8px; font-size: 1.1em; }
.card { background: #16213e; border: 1px solid #0f3460; border-radius: 8px;
        padding: 20px; margin-bottom: 20px; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.stat { background: #0f3460; border-radius: 6px; padding: 14px; text-align: center; }
.stat .val { font-size: 2em; font-weight: bold; color: #00d4ff; }
.stat .lbl { font-size: 0.8em; color: #aaa; margin-top: 4px; }
.badge { border-radius: 4px; padding: 2px 8px; font-size: 0.75em; font-weight: bold; }
.excellent { background: #1a6e2e; color: #7eff99; }
.good      { background: #3a5c1a; color: #b8ff6e; }
.fair      { background: #6e4a1a; color: #ffd06e; }
.weak      { background: #6e1a1a; color: #ff8080; }
table.metrics { width: 100%; border-collapse: collapse; font-size: 0.85em; margin-top: 10px; }
table.metrics th, table.metrics td { border: 1px solid #0f3460; padding: 6px 10px; text-align: right; }
table.metrics th { background: #0f3460; color: #a8d8ff; text-align: center; }
table.metrics td:first-child { text-align: left; font-weight: bold; }
table.metrics tr:hover { background: #0f2040; }
.cm-table { border-collapse: collapse; font-size: 0.65em; margin-top: 10px; overflow-x: auto; display: block;
            max-width: 100%; white-space: nowrap; }
.cm-table th, .cm-table td { border: 1px solid #222; padding: 3px 5px; text-align: center; min-width: 32px; }
.cm-table th { background: #0f3460; color: #a8d8ff; }
.cm-table .diag { font-weight: bold; }
.thumb-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
.thumb-row figure { text-align: center; }
.thumb { width: 200px; height: 130px; object-fit: cover; border: 2px solid #0f3460;
         border-radius: 4px; cursor: pointer; transition: transform .2s; }
.thumb:hover { transform: scale(1.05); border-color: #00d4ff; }
figcaption { font-size: 0.7em; color: #aaa; margin-top: 4px; max-width: 200px; word-break: break-all; }
.legend { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; font-size: 0.8em; }
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot { width: 14px; height: 14px; border-radius: 3px; }
.error-table { width: 100%; border-collapse: collapse; font-size: 0.82em; margin-top: 10px; }
.error-table th, .error-table td { border: 1px solid #0f3460; padding: 5px 8px; }
.error-table th { background: #0f3460; color: #a8d8ff; }
.error-table tr.tp   { background: rgba(60,200,80,0.08); }
.error-table tr.wc   { background: rgba(255,100,100,0.12); }
.error-table tr.fp   { background: rgba(255,160,0,0.10); }
.error-table tr.fn   { background: rgba(160,60,200,0.10); }
footer { text-align: center; color: #555; margin-top: 40px; font-size: 0.8em; }
#lightbox { display:none; position:fixed; top:0;left:0;width:100%;height:100%;
            background:rgba(0,0,0,0.88); z-index:1000; align-items:center; justify-content:center; }
#lightbox.show { display:flex; }
#lightbox img { max-width:95vw; max-height:95vh; border-radius:6px; }
#lightbox span { position:absolute; top:16px;right:24px; font-size:2em; cursor:pointer; color:#fff; }
"""

_JS = """
document.querySelectorAll('.thumb').forEach(img => {
  img.addEventListener('click', () => {
    document.getElementById('lb-img').src = img.src;
    document.getElementById('lightbox').classList.add('show');
  });
});
document.getElementById('lightbox').addEventListener('click', e => {
  if (e.target !== document.getElementById('lb-img'))
    document.getElementById('lightbox').classList.remove('show');
});
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_html_report(
    output_dir: Path,
    overall_metrics: dict,
    per_class_metrics: dict,
    confusion_matrix: np.ndarray,
    all_classes: List[str],
    match_summary: List[dict],           # list of dicts per image
    predicted_images: List[Path],
    gt_images: List[Path],
    comparison_images: List[Path],
    error_images: List[Path],
    model_checkpoint: str,
    input_dir: str,
    conf_threshold: float,
    iou_threshold: float,
    total_images: int,
    images_with_gt: int,
) -> Path:
    """
    Render a self-contained HTML report and save it.

    Returns the path to the saved file.
    """

    # -- Per-class table rows -----------------------------------------------
    class_rows = []
    for cls in all_classes:
        m = per_class_metrics.get(cls, {})
        f1 = m.get("f1", 0.0)
        badge = _tier_badge(f1)
        class_rows.append(
            f'<tr>'
            f'<td>{cls}</td>'
            f'<td>{m.get("tp",0)}</td>'
            f'<td>{m.get("fp",0)}</td>'
            f'<td>{m.get("fn",0)}</td>'
            f'<td>{m.get("wrong_class",0)}</td>'
            f'<td>{m.get("precision",0):.3f}</td>'
            f'<td>{m.get("recall",0):.3f}</td>'
            f'<td>{m.get("f1",0):.3f}</td>'
            f'<td>{m.get("mean_iou",0):.3f}</td>'
            f'<td>{m.get("mean_dice",0):.3f}</td>'
            f'<td>{badge}</td>'
            f'</tr>'
        )
    class_table = "\n".join(class_rows)

    # -- Per-image error summary table ---------------------------------------
    img_summary_rows = []
    for rec in match_summary[:100]:   # cap at 100 rows for readability
        name  = rec.get("image", "")
        tp    = rec.get("tp", 0)
        fp    = rec.get("fp", 0)
        fn    = rec.get("fn", 0)
        wc    = rec.get("wrong_class", 0)
        n_gt  = rec.get("n_gt", 0)
        n_pr  = rec.get("n_pred", 0)
        mean_iou = rec.get("mean_iou", 0.0)
        status_cls = "tp" if (fp == 0 and fn == 0 and wc == 0) else "wc" if wc > 0 else "fp" if fp > 0 else "fn"
        img_summary_rows.append(
            f'<tr class="{status_cls}">'
            f'<td>{name}</td>'
            f'<td>{n_gt}</td><td>{n_pr}</td>'
            f'<td>{tp}</td><td>{fp}</td><td>{fn}</td><td>{wc}</td>'
            f'<td>{mean_iou:.3f}</td>'
            f'</tr>'
        )
    img_summary_table = "\n".join(img_summary_rows)

    # -- Confusion matrix ----------------------------------------------------
    cm_html = _cm_html(confusion_matrix, all_classes)

    # -- Thumbnail galleries -------------------------------------------------
    pred_thumbs = _example_thumbnails(predicted_images[:12], "Sample Predicted Annotations")
    gt_thumbs   = _example_thumbnails(gt_images[:12],        "Sample Ground-Truth Annotations")
    comp_thumbs = _example_thumbnails(comparison_images[:12],"Side-by-Side Comparisons (GT | Prediction)")
    err_thumbs  = _example_thumbnails(error_images[:12],     "Error Cases")

    # -- Overall stat cards --------------------------------------------------
    om = overall_metrics
    stat_cards = f"""
<div class="stat-grid">
  <div class="stat"><div class="val">{om.get('tp',0)}</div><div class="lbl">True Positives</div></div>
  <div class="stat"><div class="val">{om.get('fp',0)}</div><div class="lbl">False Positives</div></div>
  <div class="stat"><div class="val">{om.get('fn',0)}</div><div class="lbl">Missed Detections</div></div>
  <div class="stat"><div class="val">{om.get('wrong_class',0)}</div><div class="lbl">Wrong Class (mask ok)</div></div>
  <div class="stat"><div class="val">{om.get('precision',0):.3f}</div><div class="lbl">Precision</div></div>
  <div class="stat"><div class="val">{om.get('recall',0):.3f}</div><div class="lbl">Recall</div></div>
  <div class="stat"><div class="val">{om.get('f1',0):.3f}</div><div class="lbl">F1 Score</div></div>
  <div class="stat"><div class="val">{om.get('mean_iou',0):.3f}</div><div class="lbl">Mean IoU</div></div>
  <div class="stat"><div class="val">{om.get('mean_dice',0):.3f}</div><div class="lbl">Mean Dice</div></div>
  <div class="stat"><div class="val">{total_images}</div><div class="lbl">Images Processed</div></div>
  <div class="stat"><div class="val">{images_with_gt}</div><div class="lbl">Images with GT</div></div>
</div>"""

    # -- Build HTML ----------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mask2Former Car-Parts Evaluation Report</title>
<meta charset="UTF-8">
<style>{_CSS}</style>
</head>
<body>

<div id="lightbox"><span>X</span><img id="lb-img" src="" /></div>

<h1>Mask2Former Car-Parts Segmentation &mdash; Evaluation Report</h1>

<div class="card">
  <h2>&#9881; Configuration</h2>
  <table class="metrics" style="font-size:0.9em">
    <tr><td>Model checkpoint</td><td>{model_checkpoint}</td></tr>
    <tr><td>Input directory</td><td>{input_dir}</td></tr>
    <tr><td>Confidence threshold</td><td>{conf_threshold}</td></tr>
    <tr><td>IoU match threshold</td><td>{iou_threshold}</td></tr>
  </table>
</div>

<div class="card">
  <h2>&#128202; Overall Performance</h2>
  {stat_cards}
</div>

<div class="card">
  <h2>&#128203; Per-Class Metrics</h2>
  <p style="font-size:0.82em;color:#aaa;margin-bottom:8px">
    <em>Wrong Class</em> = prediction mask IoU &ge; threshold but class label differs &mdash; this is a classification error, not a boundary error.
  </p>
  <table class="metrics">
    <thead><tr>
      <th>Class</th><th>TP</th><th>FP</th><th>FN</th><th>Wrong Class</th>
      <th>Precision</th><th>Recall</th><th>F1</th><th>Mean IoU</th><th>Mean Dice</th><th>Tier</th>
    </tr></thead>
    <tbody>{class_table}</tbody>
  </table>
</div>

<div class="card">
  <h2>&#128260; Confusion Matrix</h2>
  <p style="font-size:0.82em;color:#aaa;margin-bottom:8px">
    Rows = Ground-Truth class &middot; Columns = Predicted class &middot;
    Green diagonal = correct &middot; Red off-diagonal = class confusion &middot;
    Last row = GT instance with no matching prediction &middot;
    Last column = prediction with no matching GT
  </p>
  {cm_html}
</div>

<div class="card">
  <h2>&#128240; Per-Image Summary</h2>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:rgba(60,200,80,0.5)"></div>All correct</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(255,100,100,0.5)"></div>Wrong class</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(255,160,0,0.5)"></div>False positive</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(160,60,200,0.5)"></div>Missed detection</div>
  </div>
  <table class="error-table">
    <thead><tr>
      <th>Image</th><th>GT Count</th><th>Pred Count</th>
      <th>TP</th><th>FP</th><th>FN</th><th>Wrong Class</th><th>Mean IoU</th>
    </tr></thead>
    <tbody>{img_summary_table}</tbody>
  </table>
</div>

<div class="card">
  <h2>&#128269; Annotation Review</h2>
  <p style="font-size:0.82em;color:#aaa;margin-bottom:12px">
    <strong style="color:#ffa0a0">&#9888; Classification Error</strong> = model mask overlaps GT but labels differ.
    This <em>may</em> indicate either a model error <em>or</em> an incorrect GT label &mdash; inspect the image to decide.
    The pipeline never automatically corrects GT labels.
  </p>
  <h3>Understanding error types</h3>
  <table class="metrics" style="font-size:0.85em">
    <tr><td style="color:#ff6060">(1) Genuine model error</td><td>Predicted wrong class, GT label is clearly correct on visual inspection</td></tr>
    <tr><td style="color:#ffb060">(2) Possible GT error</td><td>Model predicts a visually plausible class, but GT uses a different (possibly outdated) label</td></tr>
    <tr><td style="color:#a0a0ff">(3) Label-mapping mismatch</td><td>GT uses a name not in the normalisation table &mdash; the annotation is not evaluated</td></tr>
    <tr><td style="color:#a0ffa0">(4) Boundary error</td><td>Correct class but IoU &lt; threshold &mdash; mask shape needs improvement</td></tr>
  </table>
</div>

<div class="card">
  <h2>&#128444; Visual Results</h2>
  {pred_thumbs}
  {gt_thumbs}
  {comp_thumbs}
  {err_thumbs}
</div>

<footer>
  Generated by mask2former_eval pipeline &middot;
  Model: mask2former-swin-tiny &middot;
  32-class car-parts segmentation
</footer>

<script>{_JS}</script>
</body>
</html>
"""

    report_path = output_dir / "summary_report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return report_path
