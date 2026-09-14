#!/usr/bin/env python3
"""
export_dataset_analytics.py
----------------------------
Generates a comprehensive, multi-sheet Excel report (.xlsx) for a dataset
under datasets/<dataset_name> or a given dataset directory.

Sheets generated:
  1. Overview & Summary -- Overall dataset metrics, split breakdown, source summary, class distribution
  2. Class Matrix       -- Category counts broken down by train, val, and test splits
  3. Src - <source>     -- Dedicated sheet for each dataset source (raw, matched_carparts-seg, matched_dsmlr, etc.)

Usage:
    python scripts/data/export_dataset_analytics.py --dataset combined_carparts
    python scripts/data/export_dataset_analytics.py --dataset datasets/combined_carparts --output_file custom_report.xlsx
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"

SOURCE_METADATA = {
    "raw": {
        "folder": "01_orchvate_custom",
        "display_name": "01_orchvate_custom (Orchvate Custom)",
        "download_url": "RAW_DATASET/01_orchvate_custom (Local Custom Annotations)",
    },
    "matched_carparts-seg": {
        "folder": "02_opensource_carparts_seg",
        "display_name": "02_opensource_carparts_seg (Ultralytics Open-Source)",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip",
    },
    "matched_dsmlr": {
        "folder": "03_opensource_dsmlr",
        "display_name": "03_opensource_dsmlr (DSMLR Open-Source)",
        "download_url": "https://github.com/dsmlr/Car-Parts-Segmentation.git",
    }
}


def extract_source_label(filename: str) -> str:
    """
    Parses image filename to determine its original source dataset.
    Example conventions:
      raw_Coupe_000003.jpg                    -> raw
      matched_carparts-seg_car10.jpg          -> matched_carparts-seg
      matched_dsmlr_train1.jpg                -> matched_dsmlr
    """
    if filename.startswith("raw_"):
        return "raw"
    if filename.startswith("matched_"):
        parts = filename.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    if "_" in filename:
        parts = filename.split("_")
        if parts[0]:
            return parts[0]
    return "default"


def sanitize_sheet_name(name: str, existing_names: set) -> str:
    """Sanitizes source name to fit Excel 31-character sheet limit."""
    clean = name.replace(":", "_").replace("/", "_").replace("\\", "_").replace("?", "_").replace("*", "_").replace("[", "_").replace("]", "_")
    sheet_title = f"Src - {clean}"
    if len(sheet_title) > 31:
        sheet_title = sheet_title[:31]
    
    base_title = sheet_title
    counter = 1
    while sheet_title in existing_names:
        suffix = f"_{counter}"
        sheet_title = base_title[:31 - len(suffix)] + suffix
        counter += 1
    existing_names.add(sheet_title)
    return sheet_title


def parse_dataset_analytics(dataset_path: Path):
    """
    Parses COCO JSON / YOLO files for train, val, and test splits.
    Returns aggregated metrics for dataset summary, splits, sources, and class matrices.
    """
    splits = ["train", "val", "test"]
    
    # Global counters
    total_images = 0
    total_annotations = 0
    categories_dict = {}  # cat_id -> cat_name
    
    split_metrics = {s: {"images": 0, "annotations": 0, "images_with_ann": 0} for s in splits}
    class_matrix = {}     # cat_name -> {cat_id, train, val, test, total, images_count}
    sources_data = {}     # source_name -> {train: {img, ann, class_counts}, val: ..., test: ..., total_img, total_ann}

    for split in splits:
        coco_path = dataset_path / f"coco_{split}.json"
        if coco_path.exists():
            try:
                with open(coco_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                imgs = data.get("images", [])
                anns = data.get("annotations", [])
                cats = data.get("categories", [])
                
                for c in cats:
                    categories_dict[c["id"]] = c["name"]
                
                cat_map = {c["id"]: c["name"] for c in cats}
                img_id_to_file = {img["id"]: img["file_name"] for img in imgs}
                img_id_to_source = {img["id"]: extract_source_label(img["file_name"]) for img in imgs}
                
                # Image source counts
                imgs_with_ann_set = set()
                for img in imgs:
                    src = extract_source_label(img["file_name"])
                    if src not in sources_data:
                        sources_data[src] = {
                            "splits": {s: {"images": 0, "annotations": 0, "class_counts": {}} for s in splits},
                            "total_images": 0,
                            "total_annotations": 0
                        }
                    sources_data[src]["splits"][split]["images"] += 1
                    sources_data[src]["total_images"] += 1
                
                split_metrics[split]["images"] += len(imgs)
                total_images += len(imgs)

                # Annotations parsing
                for ann in anns:
                    cat_id = ann.get("category_id")
                    cat_name = cat_map.get(cat_id, f"class_{cat_id}")
                    img_id = ann.get("image_id")
                    imgs_with_ann_set.add(img_id)
                    src = img_id_to_source.get(img_id, "default")
                    
                    split_metrics[split]["annotations"] += 1
                    total_annotations += 1
                    
                    # Class matrix update
                    if cat_name not in class_matrix:
                        class_matrix[cat_name] = {
                            "cat_id": cat_id, "train": 0, "val": 0, "test": 0, "total": 0, "img_set": set()
                        }
                    class_matrix[cat_name][split] += 1
                    class_matrix[cat_name]["total"] += 1
                    if img_id in img_id_to_file:
                        class_matrix[cat_name]["img_set"].add(img_id)

                    # Source data update
                    if src not in sources_data:
                        sources_data[src] = {
                            "splits": {s: {"images": 0, "annotations": 0, "class_counts": {}} for s in splits},
                            "total_images": 0,
                            "total_annotations": 0
                        }
                    sources_data[src]["splits"][split]["annotations"] += 1
                    sources_data[src]["total_annotations"] += 1
                    
                    src_cls = sources_data[src]["splits"][split]["class_counts"]
                    src_cls[cat_name] = src_cls.get(cat_name, 0) + 1

                split_metrics[split]["images_with_ann"] = len(imgs_with_ann_set)

            except Exception as e:
                print(f"[WARN] Error reading {coco_path.name}: {e}")
        else:
            # Fallback to images directory (YOLO layout)
            img_dir = dataset_path / "images" / split
            lbl_dir = dataset_path / "labels" / split
            if img_dir.exists():
                img_files = list(img_dir.glob("*.*"))
                n_img = len(img_files)
                split_metrics[split]["images"] += n_img
                total_images += n_img
                
                for img_file in img_files:
                    src = extract_source_label(img_file.name)
                    if src not in sources_data:
                        sources_data[src] = {
                            "splits": {s: {"images": 0, "annotations": 0, "class_counts": {}} for s in splits},
                            "total_images": 0,
                            "total_annotations": 0
                        }
                    sources_data[src]["splits"][split]["images"] += 1
                    sources_data[src]["total_images"] += 1

                # If labels directory exists, parse .txt annotations
                if lbl_dir.exists():
                    for lbl_file in lbl_dir.glob("*.txt"):
                        src = extract_source_label(lbl_file.name)
                        try:
                            with open(lbl_file, "r") as lf:
                                for line in lf:
                                    parts = line.strip().split()
                                    if not parts:
                                        continue
                                    cid = int(parts[0])
                                    cat_name = categories_dict.get(cid, f"class_{cid}")
                                    
                                    split_metrics[split]["annotations"] += 1
                                    total_annotations += 1
                                    
                                    if cat_name not in class_matrix:
                                        class_matrix[cat_name] = {
                                            "cat_id": cid, "train": 0, "val": 0, "test": 0, "total": 0, "img_set": set()
                                        }
                                    class_matrix[cat_name][split] += 1
                                    class_matrix[cat_name]["total"] += 1

                                    if src not in sources_data:
                                        sources_data[src] = {
                                            "splits": {s: {"images": 0, "annotations": 0, "class_counts": {}} for s in splits},
                                            "total_images": 0,
                                            "total_annotations": 0
                                        }
                                    sources_data[src]["splits"][split]["annotations"] += 1
                                    sources_data[src]["total_annotations"] += 1
                                    src_cls = sources_data[src]["splits"][split]["class_counts"]
                                    src_cls[cat_name] = src_cls.get(cat_name, 0) + 1
                        except Exception:
                            pass

    return {
        "dataset_name": dataset_path.name,
        "dataset_path": str(dataset_path),
        "total_images": total_images,
        "total_annotations": total_annotations,
        "total_categories": len(class_matrix),
        "split_metrics": split_metrics,
        "class_matrix": class_matrix,
        "sources_data": sources_data
    }


def generate_excel_analytics_report(data: dict, output_file: Path) -> Path:
    """
    Builds a beautifully styled multi-sheet Excel file.
    """
    dataset_name = data["dataset_name"]
    total_images = data["total_images"]
    total_annotations = data["total_annotations"]
    split_metrics = data["split_metrics"]
    class_matrix = data["class_matrix"]
    sources_data = data["sources_data"]
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = pd.ExcelWriter(output_file, engine="openpyxl")
    workbook = writer.book

    # Color Palette & Styles
    NAVY_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    HEADER_FONT = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    TITLE_FONT = Font(name="Segoe UI", size=14, bold=True, color="1F4E78")
    SUBTITLE_FONT = Font(name="Segoe UI", size=11, bold=True, color="333333")
    REGULAR_FONT = Font(name="Segoe UI", size=10)
    BOLD_FONT = Font(name="Segoe UI", size=10, bold=True)
    
    THIN_BORDER_SIDE = Side(border_style="thin", color="D9D9D9")
    CELL_BORDER = Border(left=THIN_BORDER_SIDE, right=THIN_BORDER_SIDE, top=THIN_BORDER_SIDE, bottom=THIN_BORDER_SIDE)
    TOTAL_BORDER = Border(top=Side(border_style="thin", color="000000"), bottom=Side(border_style="double", color="000000"))

    existing_sheet_names = set()

    # -------------------------------------------------------------------------
    # SHEET 1: Overview & Summary
    # -------------------------------------------------------------------------
    sheet1_name = "Overview & Summary"
    existing_sheet_names.add(sheet1_name)
    ws1 = workbook.create_sheet(title=sheet1_name)

    # Title Block
    ws1.cell(row=1, column=1, value=f"Dataset Analytics Overview -- '{dataset_name}'").font = TITLE_FONT
    ws1.cell(row=2, column=1, value=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Path: {data['dataset_path']}").font = REGULAR_FONT

    # Table 1: Metadata Summary
    ws1.cell(row=4, column=1, value="Dataset Overview Metrics").font = SUBTITLE_FONT
    meta_headers = ["Metric", "Value"]
    for col_idx, h in enumerate(meta_headers, start=1):
        cell = ws1.cell(row=5, column=col_idx, value=h)
        cell.fill = NAVY_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")

    meta_rows = [
        ("Dataset Name", dataset_name),
        ("Total Images", total_images),
        ("Total Annotations", total_annotations),
        ("Total Label Classes", data["total_categories"]),
        ("Avg Annotations / Image", round(total_annotations / total_images, 2) if total_images > 0 else 0),
        ("Dataset Sources Count", len(sources_data)),
    ]
    for row_idx, (k, v) in enumerate(meta_rows, start=6):
        c1 = ws1.cell(row=row_idx, column=1, value=k)
        c2 = ws1.cell(row=row_idx, column=2, value=v)
        c1.font = BOLD_FONT
        c2.font = REGULAR_FONT
        c1.border = CELL_BORDER
        c2.border = CELL_BORDER
        if isinstance(v, (int, float)):
            c2.number_format = "#,##0.00" if isinstance(v, float) else "#,##0"
            c2.alignment = Alignment(horizontal="right")

    # Table 2: Split Summary
    start_r2 = 14
    ws1.cell(row=start_r2, column=1, value="Split Breakdown").font = SUBTITLE_FONT
    split_headers = ["Split", "Image Count", "% of Images", "Annotation Count", "% of Annotations", "Avg Ann/Image"]
    for col_idx, h in enumerate(split_headers, start=1):
        cell = ws1.cell(row=start_r2 + 1, column=col_idx, value=h)
        cell.fill = NAVY_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")

    curr_r = start_r2 + 2
    for split in ["train", "val", "test"]:
        n_img = split_metrics[split]["images"]
        n_ann = split_metrics[split]["annotations"]
        pct_img = (n_img / total_images) if total_images > 0 else 0
        pct_ann = (n_ann / total_annotations) if total_annotations > 0 else 0
        avg_ann = (n_ann / n_img) if n_img > 0 else 0
        
        row_vals = [split.upper(), n_img, pct_img, n_ann, pct_ann, avg_ann]
        for col_idx, val in enumerate(row_vals, start=1):
            cell = ws1.cell(row=curr_r, column=col_idx, value=val)
            cell.font = REGULAR_FONT
            cell.border = CELL_BORDER
            if col_idx == 1:
                cell.alignment = Alignment(horizontal="center")
            elif col_idx in (2, 4):
                cell.number_format = "#,##0"
                cell.alignment = Alignment(horizontal="right")
            elif col_idx in (3, 5):
                cell.number_format = "0.00%"
                cell.alignment = Alignment(horizontal="right")
            elif col_idx == 6:
                cell.number_format = "0.00"
                cell.alignment = Alignment(horizontal="right")
        curr_r += 1

    # Total row for Split Summary
    tot_row_vals = [
        "TOTAL", total_images, 1.0, total_annotations, 1.0,
        (total_annotations / total_images) if total_images > 0 else 0
    ]
    for col_idx, val in enumerate(tot_row_vals, start=1):
        cell = ws1.cell(row=curr_r, column=col_idx, value=val)
        cell.font = BOLD_FONT
        cell.border = TOTAL_BORDER
        if col_idx == 1:
            cell.alignment = Alignment(horizontal="center")
        elif col_idx in (2, 4):
            cell.number_format = "#,##0"
            cell.alignment = Alignment(horizontal="right")
        elif col_idx in (3, 5):
            cell.number_format = "0.00%"
            cell.alignment = Alignment(horizontal="right")
        elif col_idx == 6:
            cell.number_format = "0.00"
            cell.alignment = Alignment(horizontal="right")

    # Table 3: Dataset Sources Overview
    start_r3 = curr_r + 3
    ws1.cell(row=start_r3, column=1, value="Dataset Sources Overview").font = SUBTITLE_FONT
    src_headers = ["RAW_DATASET Folder", "Source Key", "Download Link / Source URL", "Total Images", "% Images", "Total Annotations", "% Annotations", "Train Img", "Val Img", "Test Img"]
    for col_idx, h in enumerate(src_headers, start=1):
        cell = ws1.cell(row=start_r3 + 1, column=col_idx, value=h)
        cell.fill = NAVY_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")

    curr_r = start_r3 + 2
    for src_name, src_info in sorted(sources_data.items(), key=lambda x: x[1]["total_images"], reverse=True):
        meta = SOURCE_METADATA.get(src_name, {
            "folder": src_name,
            "display_name": src_name,
            "download_url": "N/A"
        })
        raw_folder = meta["folder"]
        dl_url = meta["download_url"]

        s_img = src_info["total_images"]
        s_ann = src_info["total_annotations"]
        tr_img = src_info["splits"]["train"]["images"]
        va_img = src_info["splits"]["val"]["images"]
        te_img = src_info["splits"]["test"]["images"]
        pct_img = (s_img / total_images) if total_images > 0 else 0
        pct_ann = (s_ann / total_annotations) if total_annotations > 0 else 0

        row_vals = [raw_folder, src_name, dl_url, s_img, pct_img, s_ann, pct_ann, tr_img, va_img, te_img]
        for col_idx, val in enumerate(row_vals, start=1):
            cell = ws1.cell(row=curr_r, column=col_idx, value=val)
            cell.font = REGULAR_FONT
            cell.border = CELL_BORDER
            if col_idx in (1, 2):
                cell.font = BOLD_FONT
            elif col_idx == 3:
                cell.font = REGULAR_FONT
                if dl_url.startswith("http"):
                    cell.hyperlink = dl_url
                    cell.font = Font(name="Segoe UI", size=10, color="0563C1", underline="single")
            elif col_idx in (4, 6, 8, 9, 10):
                cell.number_format = "#,##0"
                cell.alignment = Alignment(horizontal="right")
            elif col_idx in (5, 7):
                cell.number_format = "0.00%"
                cell.alignment = Alignment(horizontal="right")
        curr_r += 1


    # -------------------------------------------------------------------------
    # SHEET 2: Class Matrix
    # -------------------------------------------------------------------------
    sheet2_name = "Class Matrix"
    existing_sheet_names.add(sheet2_name)
    ws2 = workbook.create_sheet(title=sheet2_name)

    ws2.cell(row=1, column=1, value=f"Category Distribution Matrix -- '{dataset_name}'").font = TITLE_FONT
    ws2.cell(row=2, column=1, value="Annotations count per label class across train, val, and test splits").font = REGULAR_FONT

    matrix_headers = ["Rank", "Category ID", "Category Name", "Train Annotations", "Val Annotations", "Test Annotations", "Total Annotations", "% of Total", "Images Count"]
    for col_idx, h in enumerate(matrix_headers, start=1):
        cell = ws2.cell(row=4, column=col_idx, value=h)
        cell.fill = NAVY_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")

    sorted_classes = sorted(class_matrix.items(), key=lambda x: x[1]["total"], reverse=True)
    curr_r = 5
    for rank, (cat_name, info) in enumerate(sorted_classes, start=1):
        c_id = info["cat_id"]
        tr_ann = info["train"]
        va_ann = info["val"]
        te_ann = info["test"]
        tot_ann = info["total"]
        pct_ann = (tot_ann / total_annotations) if total_annotations > 0 else 0
        img_count = len(info["img_set"])

        row_vals = [rank, c_id, cat_name, tr_ann, va_ann, te_ann, tot_ann, pct_ann, img_count]
        for col_idx, val in enumerate(row_vals, start=1):
            cell = ws2.cell(row=curr_r, column=col_idx, value=val)
            cell.font = REGULAR_FONT
            cell.border = CELL_BORDER
            if col_idx in (1, 2):
                cell.alignment = Alignment(horizontal="center")
            elif col_idx == 3:
                cell.font = BOLD_FONT
            elif col_idx in (4, 5, 6, 7, 9):
                cell.number_format = "#,##0"
                cell.alignment = Alignment(horizontal="right")
            elif col_idx == 8:
                cell.number_format = "0.00%"
                cell.alignment = Alignment(horizontal="right")
        curr_r += 1

    # Matrix Total Row
    tot_matrix = [
        "TOTAL", "-", "ALL CATEGORIES",
        sum(x[1]["train"] for x in sorted_classes),
        sum(x[1]["val"] for x in sorted_classes),
        sum(x[1]["test"] for x in sorted_classes),
        total_annotations,
        1.0,
        total_images
    ]
    for col_idx, val in enumerate(tot_matrix, start=1):
        cell = ws2.cell(row=curr_r, column=col_idx, value=val)
        cell.font = BOLD_FONT
        cell.border = TOTAL_BORDER
        if col_idx in (1, 2):
            cell.alignment = Alignment(horizontal="center")
        elif col_idx in (4, 5, 6, 7, 9):
            cell.number_format = "#,##0"
            cell.alignment = Alignment(horizontal="right")
        elif col_idx == 8:
            cell.number_format = "0.00%"
            cell.alignment = Alignment(horizontal="right")

    # -------------------------------------------------------------------------
    # SEPARATE SHEETS FOR EACH SOURCE
    # -------------------------------------------------------------------------
    for src_name, src_info in sorted(sources_data.items(), key=lambda x: x[1]["total_images"], reverse=True):
        meta = SOURCE_METADATA.get(src_name, {
            "folder": src_name,
            "display_name": src_name,
            "download_url": "N/A"
        })
        raw_folder = meta["folder"]
        dl_url = meta["download_url"]

        sheet_title = sanitize_sheet_name(raw_folder, existing_sheet_names)
        ws_src = workbook.create_sheet(title=sheet_title)

        ws_src.cell(row=1, column=1, value=f"Source Analytics -- '{raw_folder}' ({src_name})").font = TITLE_FONT
        ws_src.cell(row=2, column=1, value=f"RAW Folder: /home/orchvate/CAR_TRAIN1/RAW_DATASET/{raw_folder} | Source Link: {dl_url}").font = REGULAR_FONT
        ws_src.cell(row=3, column=1, value=f"Dataset: {dataset_name} | Total Images: {src_info['total_images']:,} | Total Annotations: {src_info['total_annotations']:,}").font = REGULAR_FONT

        # Source Splits Table
        ws_src.cell(row=5, column=1, value="Split Summary for Source").font = SUBTITLE_FONT
        src_split_headers = ["Split", "Image Count", "% Source Images", "Annotation Count", "% Source Annotations", "Avg Ann/Image"]
        for col_idx, h in enumerate(src_split_headers, start=1):
            cell = ws_src.cell(row=6, column=col_idx, value=h)
            cell.fill = NAVY_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center")

        curr_r = 7
        s_total_img = src_info["total_images"]
        s_total_ann = src_info["total_annotations"]
        for split in ["train", "val", "test"]:
            sp_img = src_info["splits"][split]["images"]
            sp_ann = src_info["splits"][split]["annotations"]
            pct_s_img = (sp_img / s_total_img) if s_total_img > 0 else 0
            pct_s_ann = (sp_ann / s_total_ann) if s_total_ann > 0 else 0
            avg_s_ann = (sp_ann / sp_img) if sp_img > 0 else 0

            row_vals = [split.upper(), sp_img, pct_s_img, sp_ann, pct_s_ann, avg_s_ann]
            for col_idx, val in enumerate(row_vals, start=1):
                cell = ws_src.cell(row=curr_r, column=col_idx, value=val)
                cell.font = REGULAR_FONT
                cell.border = CELL_BORDER
                if col_idx == 1:
                    cell.alignment = Alignment(horizontal="center")
                elif col_idx in (2, 4):
                    cell.number_format = "#,##0"
                    cell.alignment = Alignment(horizontal="right")
                elif col_idx in (3, 5):
                    cell.number_format = "0.00%"
                    cell.alignment = Alignment(horizontal="right")
                elif col_idx == 6:
                    cell.number_format = "0.00"
                    cell.alignment = Alignment(horizontal="right")
            curr_r += 1

        # Source Total Row
        s_tot_row = [
            "TOTAL", s_total_img, 1.0, s_total_ann, 1.0,
            (s_total_ann / s_total_img) if s_total_img > 0 else 0
        ]
        for col_idx, val in enumerate(s_tot_row, start=1):
            cell = ws_src.cell(row=curr_r, column=col_idx, value=val)
            cell.font = BOLD_FONT
            cell.border = TOTAL_BORDER
            if col_idx == 1:
                cell.alignment = Alignment(horizontal="center")
            elif col_idx in (2, 4):
                cell.number_format = "#,##0"
                cell.alignment = Alignment(horizontal="right")
            elif col_idx in (3, 5):
                cell.number_format = "0.00%"
                cell.alignment = Alignment(horizontal="right")
            elif col_idx == 6:
                cell.number_format = "0.00"
                cell.alignment = Alignment(horizontal="right")

        # Source Class Breakdown Table
        start_r_cls = curr_r + 3
        
        # Check if we can load original raw XML labels for 01_orchvate_custom (raw)
        raw_xml_path = PROJECT_ROOT / "RAW_DATASET" / "01_orchvate_custom" / "XML" / "combined_annotations.xml"
        raw_xml_counts = None
        if src_name == "raw" and raw_xml_path.exists():
            try:
                import re
                with open(raw_xml_path, "r", encoding="utf-8") as f:
                    xml_text = f.read()
                raw_labels = re.findall(r'<polygon[^>]+label=\"([^\"]+)\"', xml_text)
                raw_xml_counts = {}
                for l in raw_labels:
                    raw_xml_counts[l] = raw_xml_counts.get(l, 0) + 1
            except Exception as e:
                print(f"[WARN] Failed to parse raw XML for 01_orchvate_custom: {e}")

        # Table A: YOLO Training Set Breakdown (23-Class Target Schema used by YOLO/MaskRCNN)
        ws_src.cell(row=start_r_cls, column=1, value="YOLO Model Training Schema Breakdown (23-Class Target Taxonomy)").font = SUBTITLE_FONT
        src_cls_headers = ["Rank", "Category Name", "Train Annotations", "Val Annotations", "Test Annotations", "Total Annotations", "% of Source Annotations"]
        for col_idx, h in enumerate(src_cls_headers, start=1):
            cell = ws_src.cell(row=start_r_cls + 1, column=col_idx, value=h)
            cell.fill = NAVY_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center")

        all_src_classes = set()
        for s in ["train", "val", "test"]:
            all_src_classes.update(src_info["splits"][s]["class_counts"].keys())
        
        src_cls_data = []
        for cname in all_src_classes:
            tr = src_info["splits"]["train"]["class_counts"].get(cname, 0)
            va = src_info["splits"]["val"]["class_counts"].get(cname, 0)
            te = src_info["splits"]["test"]["class_counts"].get(cname, 0)
            tot = tr + va + te
            src_cls_data.append((cname, tr, va, te, tot))

        src_cls_data.sort(key=lambda x: x[4], reverse=True)
        curr_r = start_r_cls + 2
        for rank, (cname, tr, va, te, tot) in enumerate(src_cls_data, start=1):
            pct = (tot / s_total_ann) if s_total_ann > 0 else 0
            row_vals = [rank, cname, tr, va, te, tot, pct]
            for col_idx, val in enumerate(row_vals, start=1):
                cell = ws_src.cell(row=curr_r, column=col_idx, value=val)
                cell.font = REGULAR_FONT
                cell.border = CELL_BORDER
                if col_idx == 1:
                    cell.alignment = Alignment(horizontal="center")
                elif col_idx == 2:
                    cell.font = BOLD_FONT
                elif col_idx in (3, 4, 5, 6):
                    cell.number_format = "#,##0"
                    cell.alignment = Alignment(horizontal="right")
                elif col_idx == 7:
                    cell.number_format = "0.00%"
                    cell.alignment = Alignment(horizontal="right")
            curr_r += 1

        s_cls_tot_row = [
            "TOTAL", "ALL CATEGORIES",
            sum(x[1] for x in src_cls_data),
            sum(x[2] for x in src_cls_data),
            sum(x[3] for x in src_cls_data),
            s_total_ann,
            1.0
        ]
        for col_idx, val in enumerate(s_cls_tot_row, start=1):
            cell = ws_src.cell(row=curr_r, column=col_idx, value=val)
            cell.font = BOLD_FONT
            cell.border = TOTAL_BORDER
            if col_idx in (1, 2):
                cell.alignment = Alignment(horizontal="left" if col_idx == 2 else "center")
            elif col_idx in (3, 4, 5, 6):
                cell.number_format = "#,##0"
                cell.alignment = Alignment(horizontal="right")
            elif col_idx == 7:
                cell.number_format = "0.00%"
                cell.alignment = Alignment(horizontal="right")

        # Table B (Optional): Original RAW XML Breakdown for 01_orchvate_custom
        if src_name == "raw" and raw_xml_counts:
            start_r_xml = curr_r + 3
            ws_src.cell(row=start_r_xml, column=1, value="Original RAW XML Class Breakdown (25 Fine-Grained Human Annotated Classes)").font = SUBTITLE_FONT
            raw_cls_headers = ["Rank", "Category Name", "Raw XML Annotations", "% of RAW Annotations"]
            for col_idx, h in enumerate(raw_cls_headers, start=1):
                cell = ws_src.cell(row=start_r_xml + 1, column=col_idx, value=h)
                cell.fill = NAVY_FILL
                cell.font = HEADER_FONT
                cell.alignment = Alignment(horizontal="center", vertical="center")

            tot_xml_ann = sum(raw_xml_counts.values())
            raw_cls_data = sorted(raw_xml_counts.items(), key=lambda x: x[1], reverse=True)
            curr_r = start_r_xml + 2

            for rank, (cname, tot) in enumerate(raw_cls_data, start=1):
                pct = (tot / tot_xml_ann) if tot_xml_ann > 0 else 0
                row_vals = [rank, cname, tot, pct]
                for col_idx, val in enumerate(row_vals, start=1):
                    cell = ws_src.cell(row=curr_r, column=col_idx, value=val)
                    cell.font = REGULAR_FONT
                    cell.border = CELL_BORDER
                    if col_idx == 1:
                        cell.alignment = Alignment(horizontal="center")
                    elif col_idx == 2:
                        cell.font = BOLD_FONT
                    elif col_idx == 3:
                        cell.number_format = "#,##0"
                        cell.alignment = Alignment(horizontal="right")
                    elif col_idx == 4:
                        cell.number_format = "0.00%"
                        cell.alignment = Alignment(horizontal="right")
                curr_r += 1

            raw_tot_row = ["TOTAL", "25 FINE-GRAINED CLASSES", tot_xml_ann, 1.0]
            for col_idx, val in enumerate(raw_tot_row, start=1):
                cell = ws_src.cell(row=curr_r, column=col_idx, value=val)
                cell.font = BOLD_FONT
                cell.border = TOTAL_BORDER
                if col_idx in (1, 2):
                    cell.alignment = Alignment(horizontal="left" if col_idx == 2 else "center")
                elif col_idx == 3:
                    cell.number_format = "#,##0"
                    cell.alignment = Alignment(horizontal="right")
                elif col_idx == 4:
                    cell.number_format = "0.00%"
                    cell.alignment = Alignment(horizontal="right")



    # Remove default sheet created by openpyxl if unused
    if "Sheet" in workbook.sheetnames:
        workbook.remove(workbook["Sheet"])

    # Auto-adjust column widths for all sheets
    for ws in workbook.worksheets:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val_str = str(cell.value or '')
                if cell.number_format and "0.00%" in cell.number_format:
                    val_str += "    "
                max_len = max(max_len, len(val_str))
            ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    workbook.save(output_file)
    return output_file


def export_analytics_excel(dataset_name_or_path: str, output_path: str = None) -> Path:
    """Main function to parse dataset and save Excel report."""
    if not _HAS_OPENPYXL:
        print("[ERROR] openpyxl is required for Excel export. Please install openpyxl.")
        return None

    path_obj = Path(dataset_name_or_path)
    if path_obj.is_dir():
        ds_path = path_obj.resolve()
    else:
        ds_path = (DATASETS_DIR / dataset_name_or_path).resolve()

    if not ds_path.exists():
        print(f"[ERROR] Dataset path '{ds_path}' does not exist.")
        return None

    if output_path is None:
        out_file = ds_path / f"{ds_path.name}_analytics.xlsx"
    else:
        out_path_obj = Path(output_path).resolve()
        if out_path_obj.suffix.lower() == ".xlsx":
            out_file = out_path_obj
        else:
            out_path_obj.mkdir(parents=True, exist_ok=True)
            out_file = out_path_obj / f"{ds_path.name}_analytics.xlsx"


    print(f"[*] Parsing dataset analytics for '{ds_path.name}'...")
    data = parse_dataset_analytics(ds_path)
    
    print(f"[*] Generating Excel workbook report...")
    saved_path = generate_excel_analytics_report(data, out_file)
    print(f"[OK] Excel report successfully generated: {saved_path}")
    return saved_path


def main():
    parser = argparse.ArgumentParser(description="Export dataset analytics to multi-sheet Excel file.")
    parser.add_argument("--dataset", default="combined_carparts", help="Dataset name under ./datasets or direct directory path")
    parser.add_argument("--output_file", default=None, help="Optional output path for .xlsx file")
    args = parser.parse_args()

    export_analytics_excel(args.dataset, args.output_file)


if __name__ == "__main__":
    main()
