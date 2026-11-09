#!/usr/bin/env python3
"""
unify_raw_dataset2.py
---------------------
Unifies RAW_DATASET2 annotations from 46 fine-grained classes
into the canonical 32-class taxonomy used across CAR_TRAIN1.

Outputs:
  - RAW_DATASET2/annotations/instances_unified.json
  - RAW_DATASET2/unified_categories.yaml
"""

import json
import yaml
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW2_DIR = PROJECT_ROOT / "RAW_DATASET2"
INPUT_JSON = RAW2_DIR / "annotations" / "instances.json"
OUTPUT_JSON = RAW2_DIR / "annotations" / "instances_unified.json"
OUTPUT_YAML = RAW2_DIR / "unified_categories.yaml"

# 32 Canonical Classes (1-indexed matching CAR_TRAIN1 standard)
CANONICAL_CATEGORIES = {
    1: "wheel",
    2: "wheel_cap",
    3: "front_left_light",
    4: "front_right_light",
    5: "door_glass",
    6: "left_mirror",
    7: "right_mirror",
    8: "front_bumper",
    9: "grille",
    10: "front_left_door",
    11: "front_right_door",
    12: "hood",
    13: "pillar",
    14: "back_left_door",
    15: "back_right_door",
    16: "front_glass",
    17: "front_left_fender",
    18: "front_right_fender",
    19: "back_left_light",
    20: "back_right_light",
    21: "rear_quarter_panel",
    22: "roof",
    23: "side_skirt_rocker_panel",
    24: "fog_lamp",
    25: "back_bumper",
    26: "indicator",
    27: "number_plate",
    28: "back_glass",
    29: "tailgate",
    30: "fuel_lid",
    31: "antenna",
    32: "trunk",
}

CANONICAL_NAME_TO_ID = {name: cid for cid, name in CANONICAL_CATEGORIES.items()}

# Mapping from RAW_DATASET2 46 classes to 32 canonical classes
FINE_TO_CANONICAL_NAME = {
    "wheel": "wheel",
    "front_left_wheel": "wheel",
    "front_right_wheel": "wheel",
    "rear_left_wheel": "wheel",
    "rear_right_wheel": "wheel",
    "front_left_wheel_cap": "wheel_cap",
    "front_right_wheel_cap": "wheel_cap",
    "rear_left_wheel_cap": "wheel_cap",
    "rear_right_wheel_cap": "wheel_cap",
    "wheel_cap": "wheel_cap",
    "front_left_light": "front_left_light",
    "front_right_light": "front_right_light",
    "rear_left_light": "back_left_light",
    "rear_right_light": "back_right_light",
    "back_left_light": "back_left_light",
    "back_right_light": "back_right_light",
    "door_glass": "door_glass",
    "front_left_door_glass": "door_glass",
    "front_right_door_glass": "door_glass",
    "rear_left_door_glass": "door_glass",
    "rear_right_door_glass": "door_glass",
    "left_mirror": "left_mirror",
    "right_mirror": "right_mirror",
    "front_bumper": "front_bumper",
    "rear_bumper": "back_bumper",
    "back_bumper": "back_bumper",
    "grille": "grille",
    "front_left_door": "front_left_door",
    "front_right_door": "front_right_door",
    "rear_left_door": "back_left_door",
    "rear_right_door": "back_right_door",
    "back_left_door": "back_left_door",
    "back_right_door": "back_right_door",
    "hood": "hood",
    "left_pillar": "pillar",
    "right_pillar": "pillar",
    "pillar": "pillar",
    "front_glass": "front_glass",
    "rear_glass": "back_glass",
    "back_glass": "back_glass",
    "front_left_fender": "front_left_fender",
    "front_right_fender": "front_right_fender",
    "rear_left_quarter_panel": "rear_quarter_panel",
    "rear_right_quarter_panel": "rear_quarter_panel",
    "rear_quarter_panel": "rear_quarter_panel",
    "roof": "roof",
    "left_side_skirt": "side_skirt_rocker_panel",
    "right_side_skirt": "side_skirt_rocker_panel",
    "side_skirt_rocker_panel": "side_skirt_rocker_panel",
    "front_left_fog_lamp": "fog_lamp",
    "front_right_fog_lamp": "fog_lamp",
    "fog_lamp": "fog_lamp",
    "left_indicator": "indicator",
    "right_indicator": "indicator",
    "indicator": "indicator",
    "front_license_plate": "number_plate",
    "rear_license_plate": "number_plate",
    "number_plate": "number_plate",
    "tailgate": "tailgate",
    "fuel_lid": "fuel_lid",
    "antenna": "antenna",
    "trunk": "trunk",
}


def unify():
    print(f"[*] Reading {INPUT_JSON}...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build old_id -> new_canonical_id
    old_id_to_new_id = {}
    for cat in data.get("categories", []):
        old_id = cat["id"]
        old_name = cat["name"]
        canon_name = FINE_TO_CANONICAL_NAME.get(old_name)
        if canon_name and canon_name in CANONICAL_NAME_TO_ID:
            old_id_to_new_id[old_id] = CANONICAL_NAME_TO_ID[canon_name]
        else:
            print(f"[WARN] Category {old_name} (ID: {old_id}) could not be mapped!")

    print(f"[*] Remapped {len(old_id_to_new_id)} / {len(data['categories'])} categories.")

    # Remap annotations
    remapped_count = 0
    new_annotations = []
    for ann in data.get("annotations", []):
        old_cat_id = ann.get("category_id")
        if old_cat_id in old_id_to_new_id:
            ann["category_id"] = old_id_to_new_id[old_cat_id]
            new_annotations.append(ann)
            remapped_count += 1

    data["annotations"] = new_annotations

    # Replace categories with the 32 canonical classes
    data["categories"] = [
        {"id": cid, "name": name, "supercategory": "car_part"}
        for cid, name in sorted(CANONICAL_CATEGORIES.items(), key=lambda x: x[0])
    ]

    # Update info
    if "info" in data:
        data["info"]["description"] = "Combined Vehicle Parts Dataset (32 Canonical Classes, unified)"

    print(f"[*] Writing {OUTPUT_JSON} ({remapped_count} annotations, {len(data['images'])} images)...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Write unified_categories.yaml
    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.dump({"final_categories": CANONICAL_CATEGORIES}, f, sort_keys=False)

    print(f"[OK] Created {OUTPUT_JSON}")
    print(f"[OK] Created {OUTPUT_YAML}")


if __name__ == "__main__":
    unify()
