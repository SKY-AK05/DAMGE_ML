#!/usr/bin/env python3
"""
best_model_analytics.py  --  Deep Per-Class Analytics for the Best Trained Model
=================================================================================

Produces a self-contained export bundle inside <out_dir>:

  per_class_analytics/
      ranked_parts.csv              -- all classes ranked by mAP50
      map50_bar.png                 -- horizontal bar chart (green/amber/red tiers)
      radar_chart.png               -- spider chart precision vs recall per class
      precision_recall_heatmap.png  -- grid heatmap: class × metric
      score_distribution.png        -- mAP50 distribution bar (sorted)
  test_results/                     -- inference output images (if test images available)
  analytics_summary.xlsx            -- 4-sheet Excel: Overview | Per-Class | Rankings | Weak Classes
  best_model/                       -- copy of best model weights
  comparison_plots/                 -- copied comparison plots from parent run
  model_comparison_report.xlsx      -- copied from parent run
  README_export.txt                 -- auto-generated index

Usage (standalone):
    python scripts/evaluation/best_model_analytics.py \\
        --run_dir runs_comparison/run_20260912_204030/ \\
        --test_images test_data/ \\
        --out_dir runs_comparison/run_20260912_204030/analytics_export/

Or let it auto-find the best run (no --run_dir needed):
    python scripts/evaluation/best_model_analytics.py --auto
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ── optional heavy deps (graceful degradation) ────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib / numpy not found — chart generation will be skipped.")

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("[WARN] openpyxl not found — Excel export will be skipped.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── colour tiers ──────────────────────────────────────────────────────────────
TIER_EXCELLENT = 0.80
TIER_GOOD      = 0.60
TIER_FAIR      = 0.40
# below FAIR → weak/red

COLOUR_EXCELLENT = "#2ecc71"  # green
COLOUR_GOOD      = "#f39c12"  # amber
COLOUR_FAIR      = "#e67e22"  # orange
COLOUR_WEAK      = "#e74c3c"  # red


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-DETECT BEST RUN
# ═══════════════════════════════════════════════════════════════════════════════

def find_all_runs(runs_comparison_dir: Path):
    """Return list of (run_dir, model_subdir, best_metrics_dict)."""
    results = []
    if not runs_comparison_dir.exists():
        return results
    for run_dir in sorted(runs_comparison_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        for model_dir in run_dir.iterdir():
            if not model_dir.is_dir():
                continue
            bm = model_dir / "best_metrics.json"
            if bm.exists():
                try:
                    with open(bm, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    results.append((run_dir, model_dir, data))
                except Exception:
                    pass
    return results


def select_best_run(runs_comparison_dir: Path):
    """
    Scan all runs and return the (run_dir, model_dir, best_metrics) tuple
    with the highest mask mAP50 (or box mAP50 for YOLO).
    Returns None if nothing found.
    """
    all_runs = find_all_runs(runs_comparison_dir)
    if not all_runs:
        return None

    def _score(entry):
        _, _, data = entry
        # mask mAP50 preferred; fall back to box
        vm = data.get("val_metrics", {})
        s = vm.get("map50_mask", vm.get("map50_box", 0.0))
        if s == 0.0:
            s = data.get("best_mask_map50", data.get("best_box_map50", 0.0))
        return s

    best = max(all_runs, key=_score)
    return best


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-CLASS DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def load_per_class(model_dir: Path):
    """
    Load per-class metrics. Prefers best_metrics.json (val_metrics.per_class),
    falls back to per_class_metrics.json.
    Returns dict: {class_name: {map50, map5095, precision, recall}}
    """
    # Try best_metrics.json first (best epoch per-class)
    bm_path = model_dir / "best_metrics.json"
    if bm_path.exists():
        try:
            with open(bm_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pc = data.get("val_metrics", {}).get("per_class", {})
            if pc:
                return pc
        except Exception:
            pass

    # Fall back to per_class_metrics.json (final epoch)
    pc_path = model_dir / "per_class_metrics.json"
    if pc_path.exists():
        try:
            with open(pc_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {}


def load_overall_metrics(model_dir: Path):
    """Load overall best-epoch metrics dict."""
    bm_path = model_dir / "best_metrics.json"
    if bm_path.exists():
        try:
            with open(bm_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def tier_colour(map50_val):
    if map50_val >= TIER_EXCELLENT:
        return COLOUR_EXCELLENT
    elif map50_val >= TIER_GOOD:
        return COLOUR_GOOD
    elif map50_val >= TIER_FAIR:
        return COLOUR_FAIR
    else:
        return COLOUR_WEAK


def tier_label(map50_val):
    if map50_val >= TIER_EXCELLENT:
        return "Excellent"
    elif map50_val >= TIER_GOOD:
        return "Good"
    elif map50_val >= TIER_FAIR:
        return "Fair"
    else:
        return "Weak"


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def _ranked_items(per_class: dict):
    """Return list of (name, map50, map5095, precision, recall) sorted desc by map50."""
    rows = []
    for name, vals in per_class.items():
        rows.append((
            name,
            float(vals.get("map50", 0.0)),
            float(vals.get("map5095", 0.0)),
            float(vals.get("precision", 0.0)),
            float(vals.get("recall", 0.0)),
        ))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def save_ranked_csv(per_class: dict, out_dir: Path):
    rows = _ranked_items(per_class)
    csv_path = out_dir / "ranked_parts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Part", "mAP50", "mAP50-95", "Precision", "Recall", "Tier"])
        for i, (name, m50, m5095, prec, rec) in enumerate(rows, 1):
            writer.writerow([i, name, f"{m50:.4f}", f"{m5095:.4f}",
                             f"{prec:.4f}", f"{rec:.4f}", tier_label(m50)])
    print(f"[OK] Ranked CSV -> {csv_path}")
    return csv_path


def plot_map50_bar(per_class: dict, out_dir: Path, model_name: str = ""):
    if not HAS_MPL:
        return None
    rows = _ranked_items(per_class)
    names  = [r[0].replace("_", " ") for r in rows]
    map50s = [r[1] for r in rows]
    colours = [tier_colour(v) for v in map50s]

    fig_h = max(8, len(names) * 0.42)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(names, map50s, color=colours, height=0.7, edgecolor="#0f3460", linewidth=0.5)

    # value labels
    for bar, val in zip(bars, map50s):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8.5,
                color="white", fontweight="bold")

    # tier reference lines
    for thresh, lbl, clr in [
        (TIER_EXCELLENT, "Excellent", COLOUR_EXCELLENT),
        (TIER_GOOD,      "Good",      COLOUR_GOOD),
        (TIER_FAIR,      "Fair",      COLOUR_FAIR),
    ]:
        ax.axvline(thresh, color=clr, linestyle="--", linewidth=1.2, alpha=0.6)
        ax.text(thresh + 0.003, len(names) - 0.5, lbl,
                color=clr, fontsize=7.5, va="top", alpha=0.85)

    ax.set_xlim(0, 1.08)
    ax.set_xlabel("mAP @ IoU 0.50", color="white", fontsize=11)
    title_str = f"Per-Part mAP50 — {model_name}" if model_name else "Per-Part mAP50"
    ax.set_title(title_str, color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#0f3460")
    ax.invert_yaxis()

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOUR_EXCELLENT, label=f"Excellent  (≥{TIER_EXCELLENT:.0%})"),
        Patch(facecolor=COLOUR_GOOD,      label=f"Good       (≥{TIER_GOOD:.0%})"),
        Patch(facecolor=COLOUR_FAIR,      label=f"Fair       (≥{TIER_FAIR:.0%})"),
        Patch(facecolor=COLOUR_WEAK,      label=f"Weak       (<{TIER_FAIR:.0%})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.3,
              facecolor="#1a1a2e", edgecolor="#0f3460", labelcolor="white", fontsize=8)

    plt.tight_layout()
    out_path = out_dir / "map50_bar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] mAP50 bar chart -> {out_path}")
    return out_path


def plot_radar_chart(per_class: dict, out_dir: Path, model_name: str = ""):
    if not HAS_MPL:
        return None
    rows = _ranked_items(per_class)
    if len(rows) < 3:
        return None

    names = [r[0].replace("_", "\n") for r in rows]
    precs  = [r[3] for r in rows]
    recs   = [r[4] for r in rows]
    map50s = [r[1] for r in rows]

    N = len(names)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    def _plot_series(values, label, colour):
        vals_closed = values + [values[0]]
        ax.plot(angles_closed, vals_closed, color=colour, linewidth=2, label=label)
        ax.fill(angles_closed, vals_closed, color=colour, alpha=0.12)

    _plot_series(precs,  "Precision", "#3498db")
    _plot_series(recs,   "Recall",    "#2ecc71")
    _plot_series(map50s, "mAP50",     "#e74c3c")

    ax.set_xticks(angles)
    ax.set_xticklabels(names, size=7.5, color="white")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8, color="#aaaaaa")
    ax.set_ylim(0, 1.0)
    ax.spines["polar"].set_color("#0f3460")
    ax.grid(color="#0f3460", linestyle="--", linewidth=0.6)

    title_str = f"Per-Part Precision / Recall / mAP50\n{model_name}" if model_name else "Per-Part Precision / Recall / mAP50"
    ax.set_title(title_str, color="white", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#1a1a2e", edgecolor="#0f3460", labelcolor="white", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "radar_chart.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Radar chart -> {out_path}")
    return out_path


def plot_heatmap(per_class: dict, out_dir: Path, model_name: str = ""):
    if not HAS_MPL:
        return None
    rows = _ranked_items(per_class)  # sorted by map50 desc
    if not rows:
        return None

    part_names = [r[0].replace("_", " ") for r in rows]
    metrics    = ["mAP50", "mAP50-95", "Precision", "Recall"]
    data = np.array([
        [r[1], r[2], r[3], r[4]] for r in rows
    ])  # shape: (n_classes, 4)

    fig_h = max(10, len(part_names) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, color="white", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(part_names)))
    ax.set_yticklabels(part_names, color="white", fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # annotate cells
    for i in range(len(part_names)):
        for j in range(len(metrics)):
            val = data[i, j]
            text_colour = "black" if val > 0.55 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=text_colour, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    title_str = f"Per-Class Metrics Heatmap — {model_name}" if model_name else "Per-Class Metrics Heatmap"
    ax.set_title(title_str, color="white", fontsize=12, fontweight="bold", pad=18)

    plt.tight_layout()
    out_path = out_dir / "precision_recall_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Heatmap -> {out_path}")
    return out_path


def plot_score_distribution(per_class: dict, out_dir: Path, model_name: str = ""):
    """Grouped bar: mAP50 vs mAP50-95 per class (sorted by mAP50)."""
    if not HAS_MPL:
        return None
    rows = _ranked_items(per_class)
    if not rows:
        return None

    names  = [r[0].replace("_", " ") for r in rows]
    map50s = np.array([r[1] for r in rows])
    map95s = np.array([r[2] for r in rows])

    x = np.arange(len(names))
    width = 0.4

    fig_w = max(14, len(names) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    b1 = ax.bar(x - width / 2, map50s, width, label="mAP50",    color="#3498db", alpha=0.85, edgecolor="#0f3460")
    b2 = ax.bar(x + width / 2, map95s, width, label="mAP50-95", color="#e74c3c", alpha=0.85, edgecolor="#0f3460")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8, color="white")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color="white", fontsize=11)
    title_str = f"mAP50 vs mAP50-95 Per Part — {model_name}" if model_name else "mAP50 vs mAP50-95 Per Part"
    ax.set_title(title_str, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.legend(facecolor="#1a1a2e", edgecolor="#0f3460", labelcolor="white", fontsize=9)
    ax.axhline(0.6, color=COLOUR_GOOD, linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0.8, color=COLOUR_EXCELLENT, linestyle="--", linewidth=1, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#0f3460")

    plt.tight_layout()
    out_path = out_dir / "score_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Score distribution -> {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  EXCEL EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

_DARK_BG     = "FF1A1A2E"
_HEADER_BG   = "FF0F3460"
_ACCENT_BLUE = "FF3498DB"
_ACCENT_GRN  = "FF2ECC71"
_ACCENT_AMB  = "FFF39C12"
_ACCENT_RED  = "FFE74C3C"
_WHITE       = "FFFFFFFF"
_LIGHT_BG    = "FF16213E"


def _hdr_font():
    return Font(name="Calibri", bold=True, color=_WHITE, size=11)

def _cell_font():
    return Font(name="Calibri", color=_WHITE, size=10)

def _hdr_fill():
    return PatternFill("solid", fgColor=_HEADER_BG)

def _dark_fill():
    return PatternFill("solid", fgColor=_DARK_BG)

def _alt_fill():
    return PatternFill("solid", fgColor=_LIGHT_BG)

def _tier_fill(map50_val):
    clr_map = {
        "Excellent": "FF27AE60",
        "Good":      "FFD68910",
        "Fair":      "FFE67E22",
        "Weak":      "FFC0392B",
    }
    t = tier_label(map50_val)
    return PatternFill("solid", fgColor=clr_map[t])

def _centre():
    return Alignment(horizontal="center", vertical="center", wrap_text=True)

def _thin_border():
    s = Side(border_style="thin", color="FF0F3460")
    return Border(left=s, right=s, top=s, bottom=s)

def _set_col_widths(ws, widths: dict):
    for col, w in widths.items():
        ws.column_dimensions[col].width = w

def _write_header_row(ws, row, headers):
    for col, text in enumerate(headers, 1):
        c = ws.cell(row=row, column=col, value=text)
        c.font = _hdr_font()
        c.fill = _hdr_fill()
        c.alignment = _centre()
        c.border = _thin_border()


def build_excel(per_class: dict, overall: dict, out_path: Path, model_name: str = ""):
    if not HAS_OPENPYXL:
        print("[WARN] Skipping Excel — openpyxl not available.")
        return None

    wb = openpyxl.Workbook()

    # ── Sheet 1: Overview ─────────────────────────────────────────────────────
    ws_ov = wb.active
    ws_ov.title = "Overview"
    ws_ov.sheet_view.showGridLines = False

    vm = overall.get("val_metrics", {})
    ts = overall.get("train_stats", {})
    best_epoch = overall.get("best_epoch", "—")

    overview_data = [
        ("Model", overall.get("model_name", model_name)),
        ("Best Epoch", best_epoch),
        ("Mask mAP50",    f"{vm.get('map50_mask', vm.get('map50_box', 0)):>.4f}"),
        ("Mask mAP50-95", f"{vm.get('map5095_mask', vm.get('map5095_box', 0)):>.4f}"),
        ("Precision",     f"{vm.get('precision_mask', vm.get('precision_box', 0)):>.4f}"),
        ("Recall",        f"{vm.get('recall_mask', vm.get('recall_box', 0)):>.4f}"),
        ("Mask IoU",      f"{vm.get('mask_iou', 0):>.4f}"),
        ("Dice / F1",     f"{vm.get('dice_f1', 0):>.4f}"),
        ("Boundary IoU",  f"{vm.get('boundary_iou', 0):>.4f}"),
        ("Train Loss",    f"{ts.get('train_loss', 0):>.4f}"),
        ("Val Loss",      f"{ts.get('val_loss', 0):>.4f}"),
        ("GPU Mem (GB)",  f"{ts.get('gpu_mem_gb', 0):>.2f}"),
        ("Epoch Time (s)",f"{ts.get('epoch_time_sec', 0):>.1f}"),
        ("Images/sec",    f"{ts.get('images_sec', 0):>.2f}"),
    ]

    ws_ov.cell(1, 1, "Car-Parts Segmentation — Best Model Overview").font = Font(
        name="Calibri", bold=True, size=14, color=_WHITE)
    ws_ov.cell(1, 1).fill = PatternFill("solid", fgColor=_DARK_BG)
    ws_ov.merge_cells("A1:B1")

    _write_header_row(ws_ov, 2, ["Metric", "Value"])
    for i, (metric, value) in enumerate(overview_data, 3):
        fill = _dark_fill() if i % 2 == 0 else _alt_fill()
        for col, val in enumerate([metric, value], 1):
            c = ws_ov.cell(row=i, column=col, value=val)
            c.font = _cell_font()
            c.fill = fill
            c.alignment = _centre()
            c.border = _thin_border()

    _set_col_widths(ws_ov, {"A": 22, "B": 18})

    # ── Sheet 2: Per-Class Detail ─────────────────────────────────────────────
    ws_pc = wb.create_sheet("Per-Class Detail")
    ws_pc.sheet_view.showGridLines = False
    headers = ["Part Name", "mAP50", "mAP50-95", "Precision", "Recall", "Tier"]
    _write_header_row(ws_pc, 1, headers)

    rows = _ranked_items(per_class)
    for i, (name, m50, m5095, prec, rec) in enumerate(rows, 2):
        fill = _dark_fill() if i % 2 == 0 else _alt_fill()
        vals = [name.replace("_", " "), f"{m50:.4f}", f"{m5095:.4f}", f"{prec:.4f}", f"{rec:.4f}", tier_label(m50)]
        for col, val in enumerate(vals, 1):
            c = ws_pc.cell(row=i, column=col, value=val)
            c.font = _cell_font()
            c.fill = fill
            c.alignment = _centre()
            c.border = _thin_border()
        # colour the tier cell
        tier_c = ws_pc.cell(row=i, column=6)
        tier_c.fill = _tier_fill(m50)
        tier_c.font = Font(name="Calibri", bold=True, color=_WHITE, size=10)

    _set_col_widths(ws_pc, {"A": 28, "B": 12, "C": 14, "D": 14, "E": 12, "F": 12})

    # ── Sheet 3: Rankings ─────────────────────────────────────────────────────
    ws_rk = wb.create_sheet("Rankings")
    ws_rk.sheet_view.showGridLines = False
    ws_rk.cell(1, 1, "Parts Ranked by mAP50 (Best → Worst)").font = Font(
        name="Calibri", bold=True, size=12, color=_WHITE)
    ws_rk.cell(1, 1).fill = PatternFill("solid", fgColor=_DARK_BG)
    ws_rk.merge_cells("A1:G1")

    rank_headers = ["Rank", "Part Name", "mAP50", "Gap to Excellent\n(0.80)", "Precision", "Recall", "Tier"]
    _write_header_row(ws_rk, 2, rank_headers)

    for i, (name, m50, m5095, prec, rec) in enumerate(rows, 3):
        fill = _dark_fill() if i % 2 == 0 else _alt_fill()
        gap = max(0, TIER_EXCELLENT - m50)
        rank = i - 2
        vals = [rank, name.replace("_", " "), f"{m50:.4f}", f"{gap:.4f}", f"{prec:.4f}", f"{rec:.4f}", tier_label(m50)]
        for col, val in enumerate(vals, 1):
            c = ws_rk.cell(row=i, column=col, value=val)
            c.font = _cell_font()
            c.fill = fill
            c.alignment = _centre()
            c.border = _thin_border()
        ws_rk.cell(row=i, column=7).fill = _tier_fill(m50)
        ws_rk.cell(row=i, column=7).font = Font(name="Calibri", bold=True, color=_WHITE, size=10)

    _set_col_widths(ws_rk, {"A": 7, "B": 28, "C": 10, "D": 18, "E": 12, "F": 10, "G": 12})

    # ── Sheet 4: Weak Classes ─────────────────────────────────────────────────
    ws_wk = wb.create_sheet("Weak Classes")
    ws_wk.sheet_view.showGridLines = False
    weak_rows = [(n, m, m2, p, r) for n, m, m2, p, r in rows if m < TIER_GOOD]

    ws_wk.cell(1, 1, f"Weak Classes (mAP50 < {TIER_GOOD:.0%}) — Priority Improvement Targets").font = Font(
        name="Calibri", bold=True, size=12, color=_WHITE)
    ws_wk.cell(1, 1).fill = PatternFill("solid", fgColor="FF7B241C")
    ws_wk.merge_cells("A1:F1")

    _write_header_row(ws_wk, 2, ["Part Name", "mAP50", "mAP50-95", "Precision", "Recall", "Improvement Needed"])
    for i, (name, m50, m5095, prec, rec) in enumerate(weak_rows, 3):
        fill = _dark_fill() if i % 2 == 0 else _alt_fill()
        needed = f"+{TIER_EXCELLENT - m50:.3f} to Excellent"
        vals = [name.replace("_", " "), f"{m50:.4f}", f"{m5095:.4f}", f"{prec:.4f}", f"{rec:.4f}", needed]
        for col, val in enumerate(vals, 1):
            c = ws_wk.cell(row=i, column=col, value=val)
            c.font = _cell_font()
            c.fill = fill
            c.alignment = _centre()
            c.border = _thin_border()

    _set_col_widths(ws_wk, {"A": 28, "B": 10, "C": 14, "D": 12, "E": 10, "F": 24})

    # ── Apply dark bg to all remaining empty cells ────────────────────────────
    for ws in [ws_ov, ws_pc, ws_rk, ws_wk]:
        ws.sheet_properties.tabColor = "0F3460"

    wb.save(out_path)
    print(f"[OK] Analytics Excel -> {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE (TEST IMAGES)
# ═══════════════════════════════════════════════════════════════════════════════

def run_test_inference(model_dir: Path, model_name: str, test_images: Path,
                       out_dir: Path, conf: float = 0.35):
    """Run inference on test_images using the best model weights."""
    if not test_images.exists():
        print(f"[SKIP] Test images dir not found: {test_images}")
        return False

    n_imgs = len(list(test_images.rglob("*.jpg"))) + len(list(test_images.rglob("*.png")))
    if n_imgs == 0:
        print(f"[SKIP] No images in {test_images}")
        return False

    print(f"[*] Running inference on {n_imgs} test images -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_script = PROJECT_ROOT / "scripts" / "inference" / "infer_both_models.py"
    if not infer_script.exists():
        print(f"[WARN] Inference script not found: {infer_script}")
        return False

    # Resolve weights path
    weights_dir = model_dir / "weights"
    weights_best_dir = weights_dir / "best"
    weights_best_pt  = weights_dir / "best.pt"

    cmd = [sys.executable, str(infer_script),
           "--input", str(test_images),
           "--output", str(out_dir),
           "--conf", str(conf)]

    mtype = model_name.lower()
    if "mask2former" in mtype:
        if weights_best_dir.exists():
            cmd += ["--mask2former_weights", str(weights_best_dir),
                    "--skip_yolo", "--skip_maskrcnn"]
        else:
            print("[SKIP] Mask2Former weights/best/ not found.")
            return False
    elif "yolo" in mtype:
        if weights_best_pt.exists():
            cmd += ["--yolo_weights", str(weights_best_pt),
                    "--skip_mask2former", "--skip_maskrcnn"]
        else:
            print("[SKIP] YOLO best.pt not found.")
            return False
    elif "maskrcnn" in mtype:
        if weights_best_pt.exists():
            cmd += ["--maskrcnn_weights", str(weights_best_pt),
                    "--skip_yolo", "--skip_mask2former"]
        else:
            print("[SKIP] Mask R-CNN best.pt not found.")
            return False
    else:
        # Generic: pass all available weights
        if weights_best_pt.exists():
            cmd += ["--yolo_weights", str(weights_best_pt)]
        if weights_best_dir.exists():
            cmd += ["--mask2former_weights", str(weights_best_dir)]

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=1800,
                                capture_output=False)
        if result.returncode == 0:
            print(f"[OK] Inference complete -> {out_dir}")
            return True
        else:
            print(f"[WARN] Inference exited with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("[WARN] Inference timed out after 30 min.")
        return False
    except Exception as e:
        print(f"[WARN] Inference failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  README INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def write_readme(out_dir: Path, run_dir: Path, model_dir: Path,
                 overall: dict, per_class: dict, generated_files: list):
    rows = _ranked_items(per_class)
    n_classes = len(rows)
    n_excellent = sum(1 for r in rows if r[1] >= TIER_EXCELLENT)
    n_good      = sum(1 for r in rows if TIER_GOOD <= r[1] < TIER_EXCELLENT)
    n_fair      = sum(1 for r in rows if TIER_FAIR <= r[1] < TIER_GOOD)
    n_weak      = sum(1 for r in rows if r[1] < TIER_FAIR)
    vm = overall.get("val_metrics", {})
    mask_map50 = vm.get("map50_mask", vm.get("map50_box", 0.0))

    top3  = [r[0] for r in rows[:3]]
    bot3  = [r[0] for r in rows[-3:]]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    text = textwrap.dedent(f"""\
    ╔══════════════════════════════════════════════════════════════════╗
    ║         CAR-PARTS SEGMENTATION — BEST MODEL EXPORT BUNDLE        ║
    ╚══════════════════════════════════════════════════════════════════╝

    Generated : {now}
    Run       : {run_dir.name}
    Model     : {overall.get("model_name", model_dir.name)}
    Best Epoch: {overall.get("best_epoch", "—")}

    ── OVERALL PERFORMANCE ────────────────────────────────────────────
      Mask mAP50     : {mask_map50:.4f}
      Mask mAP50-95  : {vm.get("map5095_mask", vm.get("map5095_box", 0)):>.4f}
      Precision      : {vm.get("precision_mask", vm.get("precision_box", 0)):>.4f}
      Recall         : {vm.get("recall_mask", vm.get("recall_box", 0)):>.4f}
      Mask IoU       : {vm.get("mask_iou", 0):>.4f}
      Dice / F1      : {vm.get("dice_f1", 0):>.4f}

    ── PER-PART TIER BREAKDOWN ({n_classes} classes) ──────────────────────────
      ✅  Excellent (≥80%):  {n_excellent:>3} classes
      🟡  Good      (≥60%):  {n_good:>3} classes
      🟠  Fair      (≥40%):  {n_fair:>3} classes
      🔴  Weak      (<40%):  {n_weak:>3} classes

      Top performers : {", ".join(top3)}
      Needs work     : {", ".join(bot3)}

    ── FOLDER STRUCTURE ───────────────────────────────────────────────
      analytics_export/
      ├── README_export.txt               ← this file
      ├── analytics_summary.xlsx          ← 4-sheet Excel workbook
      ├── model_comparison_report.xlsx    ← cross-model comparison (if available)
      ├── best_model/                     ← best model weights (copy)
      ├── comparison_plots/               ← learning curves & metrics bar chart
      ├── per_class_analytics/
      │   ├── map50_bar.png               ← horizontal bar chart (colour-coded tiers)
      │   ├── radar_chart.png             ← spider chart: precision vs recall
      │   ├── precision_recall_heatmap.png← grid heatmap: class × metric
      │   ├── score_distribution.png      ← mAP50 vs mAP50-95 grouped bar
      │   └── ranked_parts.csv            ← all classes ranked by mAP50
      └── test_results/                   ← inference output images

    ── FILES IN THIS BUNDLE ───────────────────────────────────────────
    """)

    for fpath in generated_files:
        rel = Path(fpath).relative_to(out_dir) if Path(fpath).is_relative_to(out_dir) else Path(fpath)
        text += f"  {rel}\n"

    text += "\n──────────────────────────────────────────────────────────────────\n"

    readme_path = out_dir / "README_export.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] README -> {readme_path}")
    return readme_path


# ═══════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY TO CONSOLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_console_summary(per_class: dict, overall: dict, model_name: str):
    rows = _ranked_items(per_class)
    vm = overall.get("val_metrics", {})
    mask_map50 = vm.get("map50_mask", vm.get("map50_box", 0.0))

    print()
    print("=" * 70)
    print(f"  BEST MODEL DEEP ANALYTICS — {model_name.upper()}")
    print("=" * 70)
    print(f"  Best Epoch      : {overall.get('best_epoch', '—')}")
    print(f"  Mask mAP50      : {mask_map50:.4f}")
    print(f"  Mask mAP50-95   : {vm.get('map5095_mask', vm.get('map5095_box', 0)):.4f}")
    print(f"  Precision       : {vm.get('precision_mask', vm.get('precision_box', 0)):.4f}")
    print(f"  Recall          : {vm.get('recall_mask', vm.get('recall_box', 0)):.4f}")
    print(f"  Mask IoU        : {vm.get('mask_iou', 0):.4f}")
    print(f"  Dice / F1       : {vm.get('dice_f1', 0):.4f}")
    print()
    print(f"  {'Part':<30} {'mAP50':>7}  {'mAP50-95':>9}  {'Precision':>10}  {'Recall':>8}  {'Tier':<10}")
    print(f"  {'-'*30} {'-'*7}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*10}")
    for name, m50, m5095, prec, rec in rows:
        tier = tier_label(m50)
        print(f"  {name:<30} {m50:>7.4f}  {m5095:>9.4f}  {prec:>10.4f}  {rec:>8.4f}  {tier:<10}")
    print("=" * 70)

    n_excellent = sum(1 for r in rows if r[1] >= TIER_EXCELLENT)
    n_good      = sum(1 for r in rows if TIER_GOOD <= r[1] < TIER_EXCELLENT)
    n_fair      = sum(1 for r in rows if TIER_FAIR <= r[1] < TIER_GOOD)
    n_weak      = sum(1 for r in rows if r[1] < TIER_FAIR)
    print(f"  Tier Summary:  Excellent={n_excellent}  Good={n_good}  Fair={n_fair}  Weak={n_weak}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_analytics(run_dir: Path, model_dir: Path, overall: dict,
                  test_images: Path, out_dir: Path, conf: float = 0.35):
    """Full analytics pipeline. Returns list of generated file paths."""
    model_name = overall.get("model_name", model_dir.name)
    per_class = load_per_class(model_dir)

    if not per_class:
        print("[ERROR] No per-class metrics found. Cannot generate analytics.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    analytics_dir = out_dir / "per_class_analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    # ── Console summary ───────────────────────────────────────────────────────
    print_console_summary(per_class, overall, model_name)

    generated = []

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = save_ranked_csv(per_class, analytics_dir)
    if csv_path:
        generated.append(str(csv_path))

    # ── Charts ────────────────────────────────────────────────────────────────
    for fn in [plot_map50_bar, plot_radar_chart, plot_heatmap, plot_score_distribution]:
        p = fn(per_class, analytics_dir, model_name)
        if p:
            generated.append(str(p))

    # ── Excel ─────────────────────────────────────────────────────────────────
    excel_path = out_dir / "analytics_summary.xlsx"
    p = build_excel(per_class, overall, excel_path, model_name)
    if p:
        generated.append(str(p))

    # ── Copy best model weights ───────────────────────────────────────────────
    src_weights = model_dir / "weights"
    dst_weights = out_dir / "best_model"
    if src_weights.exists():
        if dst_weights.exists():
            shutil.rmtree(dst_weights)
        try:
            shutil.copytree(src_weights, dst_weights)
            print(f"[OK] Best model weights -> {dst_weights}")
            generated.append(str(dst_weights))
        except Exception as e:
            print(f"[WARN] Could not copy weights: {e}")

    # ── Copy comparison plots & Excel from parent run dir ─────────────────────
    src_plots = run_dir / "comparison_plots"
    dst_plots = out_dir / "comparison_plots"
    if src_plots.exists():
        if dst_plots.exists():
            shutil.rmtree(dst_plots)
        try:
            shutil.copytree(src_plots, dst_plots)
            print(f"[OK] Comparison plots -> {dst_plots}")
            generated.append(str(dst_plots))
        except Exception as e:
            print(f"[WARN] Could not copy comparison plots: {e}")

    src_excel = run_dir / "model_comparison_report.xlsx"
    dst_excel = out_dir / "model_comparison_report.xlsx"
    if src_excel.exists():
        try:
            shutil.copy2(src_excel, dst_excel)
            print(f"[OK] model_comparison_report.xlsx -> {dst_excel}")
            generated.append(str(dst_excel))
        except Exception as e:
            print(f"[WARN] Could not copy comparison Excel: {e}")

    # ── Test inference ────────────────────────────────────────────────────────
    test_out = out_dir / "test_results"
    ok = run_test_inference(model_dir, model_name, test_images, test_out, conf=conf)
    if ok:
        generated.append(str(test_out))

    # ── README ────────────────────────────────────────────────────────────────
    readme = write_readme(out_dir, run_dir, model_dir, overall, per_class, generated)
    generated.append(str(readme))

    return generated


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Best Model Deep Analytics")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-find best run across all runs_comparison/ subfolders.")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to the specific run folder (e.g. runs_comparison/run_XXXX).")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model subfolder inside run_dir. Auto-detected if omitted.")
    parser.add_argument("--test_images", type=str, default=None,
                        help="Path to test images directory (default: ./test_data).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for the export bundle. Default: <run_dir>/analytics_export/")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Inference confidence threshold (default: 0.35).")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip test-image inference step.")
    args = parser.parse_args()

    runs_dir = PROJECT_ROOT / "runs_comparison"

    if args.auto or args.run_dir is None:
        print("[*] Auto-detecting best model across all runs...")
        result = select_best_run(runs_dir)
        if result is None:
            print("[ERROR] No runs with best_metrics.json found in runs_comparison/.")
            sys.exit(1)
        run_dir, model_dir, overall = result
        print(f"[OK] Best run : {run_dir.name}")
        print(f"     Model    : {model_dir.name}")
        print(f"     Score    : {overall.get('best_mask_map50', overall.get('best_box_map50', '?')):.4f}")
    else:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            print(f"[ERROR] run_dir not found: {run_dir}")
            sys.exit(1)

        if args.model_dir:
            model_dir = Path(args.model_dir).resolve()
        else:
            # Auto-pick the model subdir with the best score inside the given run
            candidates = [(d, json.load(open(d / "best_metrics.json")))
                          for d in run_dir.iterdir()
                          if d.is_dir() and (d / "best_metrics.json").exists()]
            if not candidates:
                print(f"[ERROR] No best_metrics.json found in {run_dir}")
                sys.exit(1)
            def _s(entry):
                _, d = entry
                vm = d.get("val_metrics", {})
                return vm.get("map50_mask", vm.get("map50_box", d.get("best_mask_map50", 0.0)))
            model_dir, overall = max(candidates, key=_s)

        overall = load_overall_metrics(model_dir)

    test_images = Path(args.test_images).resolve() if args.test_images else (PROJECT_ROOT / "EVA/GROUND/images")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "analytics_export")

    if args.skip_inference:
        test_images = Path("/nonexistent_skip")  # force skip

    print(f"[*] Output bundle : {out_dir}")
    generated = run_analytics(run_dir, model_dir, overall, test_images, out_dir,
                              conf=args.conf)

    print()
    print("[*] Running custom 110-image CVAT evaluation...")
    import subprocess
    custom_eval_out = out_dir / "custom_cvat_eval"
    custom_eval_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "EVA/mask2former_eval/mask2former_eval.py",
        "--input", "EVA/GROUND/images",
        "--model", str(out_dir / "best_model" / "best"),
        "--output", str(custom_eval_out),
        "--gt_xml", "EVA/GROUND/annotations.xml"
    ]
    try:
        subprocess.run(cmd, check=True)
        generated.append(str(custom_eval_out))
        print(f"[OK] Custom evaluation complete -> {custom_eval_out}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Custom evaluation failed: {e}")

    print()
    print("=" * 70)
    print(f"  DONE — {len(generated)} artefacts saved to:")
    print(f"  {out_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
