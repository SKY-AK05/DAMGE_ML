#!/usr/bin/env python3
"""
setup_dataset_structure.py
---------------------------
One-time migration script:
  1. Moves deduplicated custom_carparts images/labels -> datasets/raw/
  2. Moves carparts-seg -> datasets/external/carparts-seg/
  3. Moves dsmlr-carparts-split -> datasets/external/dsmlr/
  4. Creates empty matched/ scaffold

Run once.  Safe to re-run (skips already-moved data).
NEVER modifies datasets/raw/ after initial population.
"""

import shutil
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS     = PROJECT_ROOT / "datasets"


# ── helpers ────────────────────────────────────────────────────────────────

def get_base_stem(filename: str) -> str:
    """Strip annotator suffix:  Coupe_000003__Team_D  ->  Coupe_000003"""
    return Path(filename).stem.split("__")[0]


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


# ── Step 1: Populate datasets/raw/ from custom_carparts (dedup) ────────────

def migrate_raw():
    src_root = DATASETS / "custom_carparts"
    dst_root = DATASETS / "raw"

    if not src_root.exists():
        print("[SKIP] datasets/custom_carparts not found — raw/ already migrated or data missing.")
        return

    # Check if raw already has content
    existing = list((dst_root / "images").rglob("*.*")) if (dst_root / "images").exists() else []
    if existing:
        print(f"[SKIP] datasets/raw/ already has {len(existing)} files — skipping migration.")
        return

    print("[*] Migrating custom_carparts -> datasets/raw/ (deduplicating annotator versions)...")

    # Group by base stem across both splits
    groups = defaultdict(list)   # base_stem -> list of (split, img_path)
    for split in ["train", "val"]:
        img_dir = src_root / "images" / split
        if img_dir.exists():
            for f in sorted(img_dir.glob("*.*")):
                groups[get_base_stem(f.name)].append((split, f))

    kept = 0
    discarded = 0

    for base_stem, versions in groups.items():
        for split in ["train", "val"]:
            split_versions = [v for v in versions if v[0] == split]
            if not split_versions:
                continue

            # Prefer the base file (no __ suffix); fall back to first alphabetically
            no_suffix = [v for v in split_versions if "__" not in v[1].stem]
            chosen_split, chosen_img = no_suffix[0] if no_suffix else split_versions[0]

            # Image
            dst_img = dst_root / "images" / chosen_split / chosen_img.name
            copy_file(chosen_img, dst_img)

            # Label
            src_lbl = src_root / "labels" / chosen_split / (chosen_img.stem + ".txt")
            if src_lbl.exists():
                dst_lbl = dst_root / "labels" / chosen_split / src_lbl.name
                copy_file(src_lbl, dst_lbl)

            kept += 1
            discarded += len(split_versions) - 1

    # Copy data.yaml
    src_yaml = src_root / "data.yaml"
    if src_yaml.exists():
        shutil.copy2(src_yaml, dst_root / "data.yaml")

    print(f"[OK] raw/ populated: {kept} base images kept, {discarded} annotator duplicates discarded.")

    # Verify counts
    for split in ["train", "val"]:
        n_img = len(list((dst_root / "images" / split).glob("*.*")))
        n_lbl = len(list((dst_root / "labels" / split).glob("*.txt")))
        print(f"     {split}: {n_img} images, {n_lbl} label files")


# ── Step 2: Move carparts-seg -> external/ ─────────────────────────────────

def migrate_external_carparts():
    src = DATASETS / "carparts-seg"
    if not src.exists() and (PROJECT_ROOT / "carparts-seg").exists():
        src = PROJECT_ROOT / "carparts-seg"
    dst = DATASETS / "external" / "carparts-seg"

    if not src.exists():
        print("[SKIP] datasets/carparts-seg not found.")
        return

    has_imgs = (dst / "images").exists() and any((dst / "images").rglob("*.*"))
    if dst.exists() and has_imgs:
        print(f"[SKIP] datasets/external/carparts-seg/ already populated.")
        return

    print("[*] Moving carparts-seg -> datasets/external/carparts-seg/ ...")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[OK] external/carparts-seg/ ready.")


# ── Step 3: Move dsmlr-carparts-split -> external/dsmlr/ ───────────────────

def migrate_external_dsmlr():
    src = DATASETS / "dsmlr-carparts-split"
    dst = DATASETS / "external" / "dsmlr"

    if not src.exists():
        print("[SKIP] datasets/dsmlr-carparts-split not found.")
        return
    if dst.exists() and any(dst.iterdir()):
        print(f"[SKIP] datasets/external/dsmlr/ already populated.")
        return

    print("[*] Moving dsmlr-carparts-split -> datasets/external/dsmlr/ ...")
    shutil.copytree(src, dst)
    print(f"[OK] external/dsmlr/ ready.")


# ── Step 4: Create matched/ scaffold ───────────────────────────────────────

def create_matched_scaffold():
    for sub in ["carparts-seg", "dsmlr"]:
        for split in ["train", "val", "test"]:
            for kind in ["images", "labels"]:
                (DATASETS / "matched" / sub / kind / split).mkdir(parents=True, exist_ok=True)
    print("[OK] datasets/matched/ scaffold created.")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Dataset Structure Migration")
    print("=" * 60)
    migrate_raw()
    migrate_external_carparts()
    migrate_external_dsmlr()
    create_matched_scaffold()
    print()
    print("Migration complete. New layout:")
    for sub in ["raw", "external/carparts-seg", "external/dsmlr",
                "matched/carparts-seg", "matched/dsmlr"]:
        p = DATASETS / sub
        n = len(list(p.rglob("*.*"))) if p.exists() else 0
        print(f"  datasets/{sub:<30} {n:>6} files")


if __name__ == "__main__":
    main()
