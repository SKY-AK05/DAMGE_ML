#!/usr/bin/env python3
"""
generate_pipeline_report.py
----------------------------
Generates a Markdown pipeline report and saves it to two locations:
  - logs/pipeline_reports/pipeline_report_<timestamp>.md  (permanent archive)
  - logs/latest_pipeline_report.md                        (always the latest, overwritten each run)

Called automatically by ensure_and_prepare_datasets() after every pipeline run.
Can also be called standalone:
  python scripts/data/generate_pipeline_report.py --run_config path/to/run_config.json

Contents:
  1. Dataset source breakdown (per source, per split)
  2. Class distribution — train and val splits separately
  3. Training config for this run (from run_config.json if provided)
"""

import argparse
import json
import datetime
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS     = PROJECT_ROOT / "datasets"
LOGS_DIR     = PROJECT_ROOT / "logs"

# 32-class taxonomy — matches combined_data/unified_categories.yaml (list index = COCO ID - 1)
NAMES = [
    "wheel", "wheel_cap", "front_left_light", "front_right_light", "door_glass",
    "left_mirror", "right_mirror", "front_bumper", "grille", "front_left_door",
    "front_right_door", "hood", "pillar", "back_left_door", "back_right_door",
    "front_glass", "front_left_fender", "front_right_fender", "back_left_light",
    "back_right_light", "rear_quarter_panel", "roof", "side_skirt_rocker_panel",
    "fog_lamp", "back_bumper", "indicator", "number_plate", "back_glass",
    "tailgate", "fuel_lid", "antenna", "trunk",
]



# ── Dataset helpers ─────────────────────────────────────────────────────────

def count_source_images(combined_dir: Path):
    """Count images per source prefix across all splits."""
    src_counts = Counter()
    for split in ["train", "val", "test"]:
        img_dir = combined_dir / "images" / split
        if not img_dir.exists():
            continue
        for f in img_dir.glob("*.*"):
            name = f.name
            if name.startswith("raw_"):
                src_counts["raw (custom_carparts)"] += 1
            elif name.startswith("matched_carparts-seg_"):
                src_counts["matched/carparts-seg"] += 1
            elif name.startswith("matched_dsmlr_"):
                src_counts["matched/dsmlr"] += 1
            else:
                src_counts[name.split("_")[0] + " (unknown)"] += 1
    return src_counts


def get_class_distribution(coco_json: Path):
    """Return {class_name: count} from a COCO JSON file."""
    if not coco_json.exists():
        return {}
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    return Counter(cats.get(a["category_id"], f"id_{a['category_id']}") for a in data.get("annotations", []))


def get_split_image_counts(combined_dir: Path):
    counts = {}
    for split in ["train", "val", "test"]:
        img_dir = combined_dir / "images" / split
        counts[split] = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
    return counts


# ── Report builder ───────────────────────────────────────────────────────────

def build_report(run_config: dict = None, timestamp: str = None) -> str:
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    combined = DATASETS / "combined_carparts"
    lines = []

    lines.append(f"# Pipeline Report")
    lines.append(f"")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"")

    # ── Section 1: Dataset source breakdown ──────────────────────────────────
    lines.append(f"## 1. Dataset Source Breakdown")
    lines.append(f"")

    src_counts = count_source_images(combined)
    split_counts = get_split_image_counts(combined)
    total = sum(src_counts.values())

    lines.append(f"| Source | Images | % of Total |")
    lines.append(f"|--------|-------:|----------:|")
    for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100 if total else 0
        lines.append(f"| {src} | {cnt:,} | {pct:.1f}% |")
    lines.append(f"| **TOTAL** | **{total:,}** | **100%** |")
    lines.append(f"")

    lines.append(f"| Split | Images |")
    lines.append(f"|-------|-------:|")
    for split, n in split_counts.items():
        if n:
            lines.append(f"| {split} | {n:,} |")
    lines.append(f"")

    # ── Section 2: Class distribution ────────────────────────────────────────
    lines.append(f"## 2. Class Distribution")
    lines.append(f"")

    for split_label, json_name in [("Train", "coco_train.json"), ("Val", "coco_val.json")]:
        coco_path = combined / json_name
        dist = get_class_distribution(coco_path)
        split_total = sum(dist.values())

        if not dist:
            lines.append(f"### {split_label} — *{json_name} not found*")
            lines.append(f"")
            continue

        lines.append(f"### {split_label} ({split_total:,} annotations)")
        lines.append(f"")
        lines.append(f"| # | Class | Count | % |")
        lines.append(f"|---|-------|------:|--:|")
        for i, name in enumerate(NAMES):
            cnt = dist.get(name, 0)
            pct = cnt / split_total * 100 if split_total else 0
            bar = "█" * int(pct / 2)  # rough visual bar, max ~50 chars at 100%
            lines.append(f"| {i} | {name} | {cnt:,} | {pct:.1f}% |")
        lines.append(f"")

    # ── Section 3: Training config ────────────────────────────────────────────
    lines.append(f"## 3. Training Configuration")
    lines.append(f"")

    if run_config:
        run_id    = run_config.get("run_id", "N/A")
        task      = run_config.get("task", "N/A")
        started   = run_config.get("started_at", "N/A")
        dataset   = run_config.get("dataset", "N/A")
        models    = run_config.get("models_selected", [])
        hparams   = run_config.get("hyperparameters", {})

        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Run ID | `{run_id}` |")
        lines.append(f"| Task | {task} |")
        lines.append(f"| Started | {started} |")
        lines.append(f"| Dataset | `{dataset}` |")
        lines.append(f"| Models | {', '.join(models) if models else 'N/A'} |")
        lines.append(f"")

        for model_key, hp in hparams.items():
            lines.append(f"### {model_key}")
            lines.append(f"")
            lines.append(f"| Hyperparameter | Value |")
            lines.append(f"|----------------|-------|")
            for k, v in hp.items():
                lines.append(f"| {k} | {v} |")
            lines.append(f"")

        # AMP status (inferred from model type — Mask R-CNN has --amp, Mask2Former has it hardcoded)
        amp_notes = []
        if "yolo" in " ".join(models).lower():
            amp_notes.append("YOLO: AMP enabled (`amp=True` hardcoded in train_yolo_seg.py)")
        if "maskrcnn" in models:
            amp_notes.append("Mask R-CNN: AMP enabled by default (`--amp` default=True)")
        if "mask2former" in models:
            amp_notes.append("Mask2Former: AMP enabled (`torch.cuda.amp.autocast` in training loop)")
        if amp_notes:
            lines.append(f"**AMP (Mixed Precision):**")
            for note in amp_notes:
                lines.append(f"- {note}")
            lines.append(f"")
    else:
        lines.append(f"*No run_config.json provided — training config not available for this report.*")
        lines.append(f"")
        lines.append(f"To include training config, ensure orchestrator saves run_config.json before calling the report generator.")
        lines.append(f"")

    # ── Section 4: Pipeline integrity ────────────────────────────────────────
    lines.append(f"## 4. Pipeline Integrity")
    lines.append(f"")
    lines.append(f"| Check | Status |")
    lines.append(f"|-------|--------|")

    raw_ok    = (DATASETS / "raw" / "images").exists() and any((DATASETS / "raw" / "images").rglob("*.*"))
    ext_cs_ok = (DATASETS / "external" / "carparts-seg").exists()
    ext_ds_ok = (DATASETS / "external" / "dsmlr").exists()
    mat_cs_ok = (DATASETS / "matched" / "carparts-seg" / "images").exists() and \
                any((DATASETS / "matched" / "carparts-seg" / "images").rglob("*.*"))
    mat_ds_ok = (DATASETS / "matched" / "dsmlr" / "images").exists() and \
                any((DATASETS / "matched" / "dsmlr" / "images").rglob("*.*"))
    coco_ok   = (combined / "coco_train.json").exists() and (combined / "coco_val.json").exists()

    def status(ok): return "✅ OK" if ok else "❌ Missing"

    lines.append(f"| `datasets/raw/` populated | {status(raw_ok)} |")
    lines.append(f"| `datasets/external/carparts-seg/` present | {status(ext_cs_ok)} |")
    lines.append(f"| `datasets/external/dsmlr/` present | {status(ext_ds_ok)} |")
    lines.append(f"| `datasets/matched/carparts-seg/` populated | {status(mat_cs_ok)} |")
    lines.append(f"| `datasets/matched/dsmlr/` populated | {status(mat_ds_ok)} |")
    lines.append(f"| COCO JSON files generated | {status(coco_ok)} |")
    lines.append(f"")

    return "\n".join(lines)


def save_report(content: str, timestamp: str = None) -> tuple:
    """Save to both archive and latest locations. Returns (archive_path, latest_path)."""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    archive_dir = LOGS_DIR / "pipeline_reports"
    archive_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    archive_path = archive_dir / f"pipeline_report_{timestamp}.md"
    latest_path  = LOGS_DIR / "latest_pipeline_report.md"

    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(content)
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(content)

    return archive_path, latest_path


def main():
    parser = argparse.ArgumentParser(description="Generate a Markdown pipeline report.")
    parser.add_argument("--run_config", default=None,
                        help="Path to run_config.json for training config section.")
    args = parser.parse_args()

    run_config = None
    if args.run_config:
        rc_path = Path(args.run_config)
        if rc_path.exists():
            with open(rc_path, "r", encoding="utf-8") as f:
                run_config = json.load(f)
        else:
            print(f"[WARN] run_config not found: {args.run_config}")

    ts_display = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_file    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    content = build_report(run_config=run_config, timestamp=ts_display)
    archive_path, latest_path = save_report(content, ts_file)

    print(f"[OK] Pipeline report saved:")
    print(f"     Archive : {archive_path}")
    print(f"     Latest  : {latest_path}")


if __name__ == "__main__":
    main()
