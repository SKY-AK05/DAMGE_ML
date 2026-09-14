#!/usr/bin/env python3
"""
export_dataset_xml.py
---------------------
Converts a dataset (images + COCO JSON / YOLO polygon text labels) under datasets/<dataset_name>
into CVAT XML 1.1 format (combined_annotations.xml), copies all images into a dedicated output directory,
and creates a compressed .zip file of the complete package.

Output structure:
  <output_dir>/
    combined_annotations.xml
    images/
      raw_Coupe_000003.jpg
      matched_carparts-seg_car10.jpg
      matched_dsmlr_train1.jpg
      ...

Zip archive:
  <output_dir>.zip

Usage:
  python scripts/data/export_dataset_xml.py --dataset combined_carparts
  python scripts/data/export_dataset_xml.py --dataset combined_carparts --out_dir runs_comparison/run_20260912_150000/combined_xml
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"


def export_dataset_xml(dataset_name_or_path: str, out_dir: str = None, make_zip: bool = True):
    """
    Parses COCO JSON / YOLO format for train, val, and test splits,
    generates a CVAT XML 1.1 annotation file, copies images, and zips the output directory.
    """
    path_obj = Path(dataset_name_or_path)
    if path_obj.is_dir():
        ds_path = path_obj.resolve()
    else:
        ds_path = (DATASETS_DIR / dataset_name_or_path).resolve()

    if not ds_path.exists():
        print(f"[ERROR] Dataset directory '{ds_path}' does not exist.")
        return None, None

    if out_dir is None:
        target_dir = ds_path.parent / f"{ds_path.name}_xml"
    else:
        target_path = Path(out_dir).resolve()
        if target_path.suffix.lower() == ".zip":
            target_dir = target_path.with_suffix("")
        else:
            target_dir = target_path / f"{ds_path.name}_xml"

    target_images_dir = target_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]

    # Root XML structure (CVAT 1.1)
    root = ET.Element("annotations")
    version_elem = ET.SubElement(root, "version")
    version_elem.text = "1.1"

    meta = ET.SubElement(root, "meta")
    project = ET.SubElement(meta, "project")
    proj_name = ET.SubElement(project, "name")
    proj_name.text = f"Combined Dataset - {ds_path.name}"

    labels_elem = ET.SubElement(project, "labels")

    categories_map = {}  # cat_id -> name
    categories_set = set()
    all_images = []
    copy_jobs = []

    global_img_id = 0

    print(f"[*] Collecting dataset images and annotations from '{ds_path.name}'...")

    for split in splits:
        coco_file = ds_path / f"coco_{split}.json"
        img_split_dir = ds_path / "images" / split
        lbl_split_dir = ds_path / "labels" / split

        if coco_file.exists():
            try:
                with open(coco_file, "r", encoding="utf-8") as f:
                    coco = json.load(f)

                cats = coco.get("categories", [])
                for c in cats:
                    cid = c["id"]
                    cname = c["name"]
                    categories_map[cid] = cname
                    categories_set.add(cname)

                imgs = coco.get("images", [])
                anns = coco.get("annotations", [])

                local_anns = {}
                for ann in anns:
                    local_anns.setdefault(ann["image_id"], []).append(ann)

                for img in imgs:
                    fname = img["file_name"]
                    src_img_path = img_split_dir / fname
                    if not src_img_path.exists():
                        matches = list(img_split_dir.glob(fname))
                        if matches:
                            src_img_path = matches[0]

                    dst_img_path = target_images_dir / fname
                    if src_img_path.exists():
                        copy_jobs.append((src_img_path, dst_img_path))

                    img_entry = {
                        "global_id": global_img_id,
                        "file_name": fname,
                        "width": img.get("width", 0),
                        "height": img.get("height", 0),
                        "annotations": local_anns.get(img["id"], []),
                        "split": split
                    }
                    all_images.append(img_entry)
                    global_img_id += 1

            except Exception as e:
                print(f"[WARN] Failed to parse {coco_file.name}: {e}")
        elif img_split_dir.exists():
            # Fallback to YOLO text format
            img_files = sorted(list(img_split_dir.glob("*.*")))
            for img_file in img_files:
                fname = img_file.name
                dst_img_path = target_images_dir / fname
                copy_jobs.append((img_file, dst_img_path))

                # Parse label .txt if exists
                lbl_file = lbl_split_dir / (img_file.stem + ".txt")
                yolo_anns = []
                w_img, h_img = 1000, 1000  # Default fallback header if unread
                if lbl_file.exists():
                    try:
                        with open(lbl_file, "r") as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if len(parts) >= 7:
                                    cid = int(parts[0])
                                    cname = f"class_{cid}"
                                    categories_set.add(cname)
                                    coords = list(map(float, parts[1:]))
                                    yolo_anns.append({"category_name": cname, "coords_norm": coords})
                    except Exception:
                        pass

                img_entry = {
                    "global_id": global_img_id,
                    "file_name": fname,
                    "width": w_img,
                    "height": h_img,
                    "yolo_annotations": yolo_anns,
                    "split": split
                }
                all_images.append(img_entry)
                global_img_id += 1

    # Populate <meta><project><labels>
    for cat_name in sorted(categories_set):
        lbl = ET.SubElement(labels_elem, "label")
        name_e = ET.SubElement(lbl, "name")
        name_e.text = cat_name
        type_e = ET.SubElement(lbl, "type")
        type_e.text = "polygon"

    size_e = ET.SubElement(meta, "size")
    size_e.text = str(len(all_images))

    # Add <image> tags with polygon elements
    for img_item in all_images:
        img_elem = ET.SubElement(root, "image", {
            "id": str(img_item["global_id"]),
            "name": img_item["file_name"],
            "width": str(img_item["width"]),
            "height": str(img_item["height"]),
            "subset": img_item["split"]
        })

        if "annotations" in img_item:
            for ann in img_item["annotations"]:
                cat_id = ann.get("category_id")
                cat_name = categories_map.get(cat_id, f"class_{cat_id}")
                segmentation = ann.get("segmentation", [])

                for seg in segmentation:
                    if len(seg) >= 6 and len(seg) % 2 == 0:
                        pts = [f"{seg[i]:.2f},{seg[i+1]:.2f}" for i in range(0, len(seg), 2)]
                        points_str = ";".join(pts)

                        ET.SubElement(img_elem, "polygon", {
                            "label": cat_name,
                            "source": "manual",
                            "occluded": "0",
                            "points": points_str,
                            "z_order": "0"
                        })
        elif "yolo_annotations" in img_item:
            w_img = img_item["width"]
            h_img = img_item["height"]
            for y_ann in img_item["yolo_annotations"]:
                cat_name = y_ann["category_name"]
                coords = y_ann["coords_norm"]
                pts = [f"{coords[i]*w_img:.2f},{coords[i+1]*h_img:.2f}" for i in range(0, len(coords), 2)]
                points_str = ";".join(pts)
                ET.SubElement(img_elem, "polygon", {
                    "label": cat_name,
                    "source": "manual",
                    "occluded": "0",
                    "points": points_str,
                    "z_order": "0"
                })

    # Write combined_annotations.xml
    xml_path = target_dir / "combined_annotations.xml"
    print(f"[*] Formatting and writing CVAT XML to '{xml_path}' ({len(all_images)} images)...")

    rough_bytes = ET.tostring(root, encoding="utf-8")
    reparsed = minidom.parseString(rough_bytes)
    pretty_xml = reparsed.toprettyxml(indent="  ", encoding="utf-8")

    with open(xml_path, "wb") as f:
        f.write(pretty_xml)

    # Copy images in parallel
    print(f"[*] Copying {len(copy_jobs)} images to '{target_images_dir}'...")

    def _copy_worker(job):
        src, dst = job
        try:
            if not dst.exists():
                shutil.copy2(src, dst)
        except Exception as e:
            print(f"[WARN] Copy failed for {src}: {e}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_copy_worker, copy_jobs))

    print(f"[OK] XML dataset folder created: {target_dir}")

    # Zip output directory
    zip_path = None
    if make_zip:
        zip_file_path = target_dir.with_suffix(".zip")
        print(f"[*] Creating ZIP archive: {zip_file_path} ...")
        shutil.make_archive(str(target_dir), "zip", root_dir=target_dir.parent, base_dir=target_dir.name)
        zip_path = Path(f"{target_dir}.zip")
        print(f"[OK] ZIP archive created: {zip_path}")

    return target_dir, zip_path


def main():
    parser = argparse.ArgumentParser(description="Export combined dataset to CVAT XML format and ZIP package.")
    parser.add_argument("--dataset", default="combined_carparts", help="Dataset name under ./datasets or direct path")
    parser.add_argument("--out_dir", default=None, help="Output directory path")
    parser.add_argument("--no_zip", action="store_true", help="Skip creating ZIP archive")
    args = parser.parse_args()

    export_dataset_xml(args.dataset, out_dir=args.out_dir, make_zip=not args.no_zip)


if __name__ == "__main__":
    main()
