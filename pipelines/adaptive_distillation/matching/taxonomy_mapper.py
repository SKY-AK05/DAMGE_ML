"""
Taxonomy mapper and alignment module.
Provides bidirectional mapping and normalization between:
- 23-class taxonomy (datasets/combined_carparts/data.yaml)
- 32-class taxonomy (unified_categories.yaml / CARPARTS_SEG_CLASSES)
- Target label schemas
"""

from typing import Dict, List, Optional, Tuple


# 23-class taxonomy used by combined_carparts
CLASSES_23 = [
    "back_bumper",            # 0
    "back_door",              # 1
    "back_glass",             # 2
    "back_left_door",         # 3
    "back_left_light",        # 4
    "back_light",             # 5
    "back_right_door",        # 6
    "back_right_light",       # 7
    "front_bumper",           # 8
    "front_door",             # 9
    "front_glass",            # 10
    "front_left_door",        # 11
    "front_left_light",       # 12
    "front_light",            # 13
    "front_right_door",       # 14
    "front_right_light",      # 15
    "hood",                   # 16
    "left_mirror",            # 17
    "object",                 # 18
    "right_mirror",           # 19
    "tailgate",               # 20
    "trunk",                  # 21
    "wheel",                  # 22
]

# 32-class taxonomy used by fine-tuned Mask2Former & unified_categories.yaml
CLASSES_32 = [
    "wheel",                   # 0
    "wheel_cap",               # 1
    "front_left_light",        # 2
    "front_right_light",       # 3
    "door_glass",              # 4
    "left_mirror",             # 5
    "right_mirror",            # 6
    "front_bumper",            # 7
    "grille",                  # 8
    "front_left_door",         # 9
    "front_right_door",        # 10
    "hood",                    # 11
    "pillar",                  # 12
    "back_left_door",          # 13
    "back_right_door",         # 14
    "front_glass",             # 15
    "front_left_fender",       # 16
    "front_right_fender",      # 17
    "back_left_light",         # 18
    "back_right_light",        # 19
    "rear_quarter_panel",      # 20
    "roof",                    # 21
    "side_skirt_rocker_panel", # 22
    "fog_lamp",                # 23
    "back_bumper",             # 24
    "indicator",               # 25
    "number_plate",            # 26
    "back_glass",              # 27
    "tailgate",                # 28
    "fuel_lid",                # 29
    "antenna",                 # 30
    "trunk",                   # 31
]

# Explicit mapping from 32-class names to 23-class canonical names
# None means the class does not exist in 23-class dataset (quarantined or dropped from direct matching)
MAP_32_TO_23_NAMES: Dict[str, Optional[str]] = {
    "wheel": "wheel",
    "wheel_cap": "wheel",                      # Merged into wheel in 23-class or treated as child
    "front_left_light": "front_left_light",
    "front_right_light": "front_right_light",
    "door_glass": None,                        # Not labeled in 23-class
    "left_mirror": "left_mirror",
    "right_mirror": "right_mirror",
    "front_bumper": "front_bumper",
    "grille": None,                            # Not in 23-class
    "front_left_door": "front_left_door",
    "front_right_door": "front_right_door",
    "hood": "hood",
    "pillar": None,
    "back_left_door": "back_left_door",
    "back_right_door": "back_right_door",
    "front_glass": "front_glass",
    "front_left_fender": None,
    "front_right_fender": None,
    "back_left_light": "back_left_light",
    "back_right_light": "back_right_light",
    "rear_quarter_panel": None,
    "roof": None,
    "side_skirt_rocker_panel": None,
    "fog_lamp": "front_light",                 # Or None
    "back_bumper": "back_bumper",
    "indicator": None,
    "number_plate": None,
    "back_glass": "back_glass",
    "tailgate": "tailgate",
    "fuel_lid": None,
    "antenna": None,
    "trunk": "trunk",
}


class TaxonomyMapper:
    """Manages class ID and class name conversions between model spaces and canonical GT space."""

    def __init__(self, canonical_classes: List[str] = CLASSES_23):
        self.canonical_classes = list(canonical_classes)
        self.canonical_name_to_id = {name: i for i, name in enumerate(self.canonical_classes)}
        self.canonical_id_to_name = {i: name for i, name in enumerate(self.canonical_classes)}

    @property
    def num_classes(self) -> int:
        return len(self.canonical_classes)

    def normalize_name(self, name: str) -> str:
        """Normalizes class string representation."""
        clean = name.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "bonnet": "hood",
            "windscreen": "front_glass",
            "rear_windscreen": "back_glass",
            "taillight": "back_light",
            "headlamp": "front_light",
            "orvm": "left_mirror",
            "rear_bumper": "back_bumper",
        }
        return aliases.get(clean, clean)

    def map_teacher_class_to_canonical(
        self, teacher_class_id: int, teacher_classes: List[str]
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Maps a teacher-specific class ID to the canonical class ID and name.
        Returns (canonical_id, canonical_name) or (None, None) if unmapped.
        """
        if teacher_class_id < 0 or teacher_class_id >= len(teacher_classes):
            return None, None

        source_name = teacher_classes[teacher_class_id]
        norm_name = self.normalize_name(source_name)

        # 1. Exact match in canonical space
        if norm_name in self.canonical_name_to_id:
            cid = self.canonical_name_to_id[norm_name]
            return cid, norm_name

        # 2. Check 32-to-23 taxonomy dictionary
        if norm_name in MAP_32_TO_23_NAMES:
            target_name = MAP_32_TO_23_NAMES[norm_name]
            if target_name and target_name in self.canonical_name_to_id:
                cid = self.canonical_name_to_id[target_name]
                return cid, target_name

        return None, None

    def get_teacher_class_list(self, class_map_name: str) -> List[str]:
        """Returns standard class list for a known taxonomy name."""
        if class_map_name in ("23", "23_class", "combined_carparts"):
            return CLASSES_23
        elif class_map_name in ("32", "32_class", "unified_categories"):
            return CLASSES_32
        raise ValueError(f"Unknown taxonomy class map name: {class_map_name}")

