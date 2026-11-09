import json
import yaml
from pathlib import Path

def check_and_compare():
    root = Path(__file__).resolve().parents[2]
    raw2_json_path = root / "RAW_DATASET2" / "annotations" / "instances.json"
    unified_yaml_path = root / "RAW_DATASET" / "combined_data" / "unified_categories.yaml"

    with open(unified_yaml_path) as f:
        unified_yaml = yaml.safe_load(f)["final_categories"]
    unified_names = {v: k for k, v in unified_yaml.items()}

    print(f"Reading categories from {raw2_json_path.name}...")
    with open(raw2_json_path, "r") as f:
        raw2_data = json.load(f)

    raw2_cats = raw2_data.get("categories", [])
    raw2_images = raw2_data.get("images", [])
    raw2_annotations = raw2_data.get("annotations", [])

    print(f"Total RAW_DATASET2 images: {len(raw2_images)}")
    print(f"Total RAW_DATASET2 annotations: {len(raw2_annotations)}")
    print(f"Total RAW_DATASET2 categories: {len(raw2_cats)}")
    print(f"Total Canonical unified categories: {len(unified_names)}")
    print("=" * 70)

    # Classification & Mapping
    direct_match = []
    mapped = []
    unmapped = []

    # Map fine-grained 46 categories to 32 canonical categories
    # Notice:
    # front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel -> wheel (1)
    # front_left_wheel_cap, etc. -> wheel_cap (2)
    # front_left_door_glass, rear_left_door_glass, etc. -> door_glass (5)
    # left_pillar, right_pillar -> pillar (13)
    # rear_left_quarter_panel, rear_right_quarter_panel -> rear_quarter_panel (21)
    # left_side_skirt, right_side_skirt -> side_skirt_rocker_panel (23)
    # front_left_fog_lamp, front_right_fog_lamp -> fog_lamp (24)
    # rear_bumper -> back_bumper (25)
    # left_indicator, right_indicator -> indicator (26)
    # front_license_plate, rear_license_plate -> number_plate (27)
    # rear_glass -> back_glass (28)
    # rear_left_door -> back_left_door (14)
    # rear_right_door -> back_right_door (15)
    # rear_left_light -> back_left_light (19)
    # rear_right_light -> back_right_light (20)

    FINE_TO_CANONICAL = {
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

    for c in raw2_cats:
        cname = c["name"]
        cid = c["id"]
        if cname in unified_names:
            direct_match.append((cid, cname, unified_names[cname]))
        elif cname in FINE_TO_CANONICAL and FINE_TO_CANONICAL[cname] in unified_names:
            target_canonical = FINE_TO_CANONICAL[cname]
            mapped.append((cid, cname, target_canonical, unified_names[target_canonical]))
        else:
            unmapped.append((cid, cname))

    print(f"Direct exact matches ({len(direct_match)}/46):")
    for cid, cname, uid in direct_match:
        print(f"  [EXACT] RAW2 ID {cid:2d}: {cname:<25} == Canonical ID {uid:2d}")

    print(f"\nSub-part / Alias mapped to Canonical ({len(mapped)}/46):")
    for cid, cname, target, uid in mapped:
        print(f"  [MAP]   RAW2 ID {cid:2d}: {cname:<25} -> {target} (Canonical ID {uid:2d})")

    print(f"\nUnmapped ({len(unmapped)}/46):")
    for cid, cname in unmapped:
        print(f"  [UNMAPPED] RAW2 ID {cid:2d}: {cname}")

    if len(unmapped) == 0:
        print("\n>>> ALL 46 RAW_DATASET2 CATEGORIES SUCCESSFULLY MAP 100% TO THE 32 CANONICAL CLASSES! <<<")

if __name__ == "__main__":
    check_and_compare()
