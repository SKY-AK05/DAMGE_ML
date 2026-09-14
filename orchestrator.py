#!/usr/bin/env python3
"""
orchestrator.py -- Interactive Pipeline Orchestrator in Python.

Provides an interactive CLI menu to run data prep, model training (local/Azure ML),
inference, evaluation, model comparison, cleanup, and fast preprocessing.
Target models: YOLO11m-seg, Mask R-CNN, and Mask2Former.

CLI flags (for automation / VM runs):
  --yes-clean-runs   Skip the interactive prompt and DELETE runs/logs/combined/matched/external
  --yes-clean-raw    Skip the interactive prompt and DELETE datasets/raw/ (DANGEROUS -- irreplaceable data!)
"""

import sys
import os
import argparse
import subprocess
import shutil
import datetime
import glob
import json
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.resolve()

# ── CLI args ────
_cli_parser = argparse.ArgumentParser(description="Car-Parts Segmentation -- Interactive Python Orchestrator")
_cli_parser.add_argument("--yes-clean-runs", action="store_true", default=False,
                          help="Non-interactively confirm deletion of runs/logs/combined/matched/external.")
_cli_parser.add_argument("--yes-clean-raw", action="store_true", default=False,
                          help="Non-interactively confirm deletion of datasets/raw/ (DANGEROUS).")
if "-h" in sys.argv or "--help" in sys.argv:
    _cli_parser.print_help()
    sys.exit(0)
_CLI_ARGS, _CLI_UNKNOWN = _cli_parser.parse_known_args()

def prompt(msg):
    try:
        return input(msg).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

def prompt_default(label, default):
    val = prompt(f"  {label} [default: {default}]: ")
    return val if val else str(default)

def prompt_int(label, default):
    val = prompt_default(label, default)
    try:
        return int(val)
    except ValueError:
        return int(default)

def display_main_menu():
    print()
    print("======================================================================")
    print("        Car-Parts Segmentation -- Interactive Python Orchestrator       ")
    print("======================================================================")
    print("  1)  Train Models Locally   -- Prepare dataset, train models & generate comparison")
    print("  2)  Compare Models         -- Compare metrics, curves, plots & multi-sheet Excel")
    print("  3)  Test on Trained Model  -- Run inference on test images using trained weights")
    print("  4)  Auto-Prepare Dataset   -- Match taxonomy, combine sources, generate COCO JSON")
    print("  5)  Dataset Analytics      -- View image counts, total annotations, & class breakdown")
    print("  6)  Export CVAT XML & ZIP  -- Export combined dataset -> CVAT XML 1.1 & ZIP package (~7k images)")
    print("  7)  Preprocess Tools       -- Standalone tools (yolo_to_coco, combine, verify, hash)")
    print("  8)  GPU Capacity Check     -- Stepwise batch size & worker stress tester")
    print("  9)  Train on Azure ML      -- Submit GPU job to Azure ML (Single / All / Dry-run)")
    print("  10) Cleanup Pipeline       -- Wipe stale outputs interactively (safe, keeps raw/)")
    print("  11) Exit")
    print("  12) Best Model Analytics   -- Auto-find best run, deep per-class analysis & export bundle")
    print("======================================================================")


class ModelSelection:
    def __init__(self, yolo=False, maskrcnn=False, mask2former=False):
        self.yolo = yolo
        self.maskrcnn = maskrcnn
        self.mask2former = mask2former

def collect_models():
    print()
    print("======================================================================")
    print(" Select Instance Segmentation Models to Train")
    print("======================================================================")
    print("  1) YOLOv11m-seg only")
    print("  2) Mask R-CNN only")
    print("  3) Mask2Former (Swin Transformer) only")
    print("  4) ALL 3 Models (YOLO11m + Mask R-CNN + Mask2Former) [Recommended]")
    print("======================================================================")
    choice = prompt("Enter choice [1-4] (default 4): ") or "4"
    mapping = {
        "1": ModelSelection(yolo=True),
        "2": ModelSelection(maskrcnn=True),
        "3": ModelSelection(mask2former=True),
        "4": ModelSelection(yolo=True, maskrcnn=True, mask2former=True),
    }
    return mapping.get(choice, ModelSelection(yolo=True, maskrcnn=True, mask2former=True))

def load_gpu_recommendations():
    json_path = PROJECT_ROOT / "runs_comparison" / "gpu_capacity_recommendations.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("models", {})
        except Exception:
            pass
    return {}

class Hyperparams:
    def __init__(self, models=None):
        recs = load_gpu_recommendations()

        def get_default_batch(model_key, fallback):
            if model_key in recs and recs[model_key].get("batch", 0) > 0:
                return recs[model_key]["batch"]
            return fallback

        def get_default_workers(model_key, fallback):
            if model_key in recs and recs[model_key].get("workers", 0) > 0:
                return recs[model_key]["workers"]
            return fallback

        need_yolo     = models is None or models.yolo
        need_maskrcnn = models is None or models.maskrcnn
        need_m2f      = models is None or models.mask2former

        print()
        print("======================================================================")
        if recs:
            print(" Set training hyperparameters (Auto-loaded defaults from latest GPU check)")
        else:
            print(" Set training hyperparameters (press Enter to accept defaults)")
        print("======================================================================")

        # ── YOLO ──────────────────────────────────────────────────────────────
        if need_yolo:
            yolo_default_b = get_default_batch("yolo11m-seg", 8)
            yolo_default_w = get_default_workers("yolo11m-seg", 8)
            print("\n---- YOLOv11m-seg ----")
            self.yolo_epochs       = prompt_int("Epochs", 50)
            self.yolo_batch        = prompt_int("Batch size (-1 = auto)", yolo_default_b)
            self.yolo_workers      = prompt_int("Dataloader workers", yolo_default_w)
            self.yolo_val_interval = prompt_int("Validate every N epochs", 5)
        else:
            self.yolo_epochs       = 50
            self.yolo_batch        = get_default_batch("yolo11m-seg", 8)
            self.yolo_workers      = get_default_workers("yolo11m-seg", 8)
            self.yolo_val_interval = 5

        # ── Mask R-CNN ────────────────────────────────────────────────────────
        if need_maskrcnn:
            mrcnn_default_b = get_default_batch("maskrcnn", 2)
            mrcnn_default_w = get_default_workers("maskrcnn", 4)
            print("\n---- Mask R-CNN ----")
            self.mrcnn_epochs       = prompt_int("Epochs", 30)
            self.mrcnn_batch        = prompt_int("Batch size", mrcnn_default_b)
            self.mrcnn_workers      = prompt_int("Dataloader workers", mrcnn_default_w)
            self.mrcnn_val_interval = prompt_int("Validate every N epochs", 5)
            self.mrcnn_accum_steps  = prompt_int("Gradient accumulation steps", 4)
        else:
            self.mrcnn_epochs       = 30
            self.mrcnn_batch        = get_default_batch("maskrcnn", 2)
            self.mrcnn_workers      = get_default_workers("maskrcnn", 4)
            self.mrcnn_val_interval = 5
            self.mrcnn_accum_steps  = 4

        # ── Mask2Former ───────────────────────────────────────────────────────
        if need_m2f:
            m2f_default_b = get_default_batch("mask2former", 2)
            m2f_default_w = get_default_workers("mask2former", 4)
            print("\n---- Mask2Former (Swin Transformer) ----")
            self.m2f_epochs       = prompt_int("Epochs", 30)
            self.m2f_batch        = prompt_int("Batch size", m2f_default_b)
            self.m2f_workers      = prompt_int("Dataloader workers", m2f_default_w)
            self.m2f_val_interval = prompt_int("Validate every N epochs", 5)
            self.m2f_accum_steps  = prompt_int("Gradient accumulation steps", 4)
            _no_obj               = prompt("  no_object_weight penalty [default: 0.25, range 0.20-0.30]: ") or "0.25"
            try:
                self.m2f_no_obj_weight = float(_no_obj)
            except ValueError:
                self.m2f_no_obj_weight = 0.25
            _compile_ans          = prompt("  Use --compile (torch.compile)? [y/N]: ")
            self.m2f_compile      = _compile_ans.strip().lower() in ("y", "yes")
        else:
            self.m2f_epochs       = 30
            self.m2f_batch        = get_default_batch("mask2former", 2)
            self.m2f_workers      = get_default_workers("mask2former", 4)
            self.m2f_val_interval = 5
            self.m2f_accum_steps  = 4
            self.m2f_no_obj_weight = 0.25
            self.m2f_compile      = False

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n---- Summary ----")
        if need_yolo:
            print(f"  YOLOv11m-seg : epochs={self.yolo_epochs}  batch={self.yolo_batch}  workers={self.yolo_workers}  val_every={self.yolo_val_interval}")
        if need_maskrcnn:
            print(f"  Mask R-CNN   : epochs={self.mrcnn_epochs}  batch={self.mrcnn_batch}  workers={self.mrcnn_workers}  val_every={self.mrcnn_val_interval}  accum={self.mrcnn_accum_steps}")
        if need_m2f:
            print(f"  Mask2Former  : epochs={self.m2f_epochs}  batch={self.m2f_batch}  workers={self.m2f_workers}  val_every={self.m2f_val_interval}  accum={self.m2f_accum_steps}  compile={self.m2f_compile}")
        print("-----------------")

def run_cmd_and_log(cmd, log_path, label):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lf.write(f"=== {label} | {timestamp} ===\n")
        lf.write(f"Command: {' '.join(cmd)}\n\n")

    print(f"\n[*] Running [{label}] ...")
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )

    GPU_REC_MARKER = "[GPU_RECOMMENDATIONS_JSON]"
    captured_recs = None

    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            for line in proc.stdout:
                sys.stdout.write(f"[{label}] {line}")
                sys.stdout.flush()
                lf.write(line)
                if GPU_REC_MARKER in line:
                    try:
                        json_start = line.index(GPU_REC_MARKER) + len(GPU_REC_MARKER)
                        captured_recs = json.loads(line[json_start:].strip())
                    except Exception:
                        pass
        proc.wait()
    except KeyboardInterrupt:
        print(f"\n[INFO] Stopped watching [{label}]. Terminating background process...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print(f"[OK] Task [{label}] canceled. Returning to main menu.")
        return False

    success = proc.returncode == 0
    if success:
        if captured_recs:
            local_rec_dir = PROJECT_ROOT / "runs_comparison"
            os.makedirs(local_rec_dir, exist_ok=True)
            local_rec_path = local_rec_dir / "gpu_capacity_recommendations.json"
            try:
                with open(local_rec_path, "w", encoding="utf-8") as f:
                    json.dump(captured_recs, f, indent=2)
                print(f"[OK] GPU recommendations saved to {local_rec_path}")
            except Exception as e:
                print(f"[WARN] Could not save recommendations: {e}")
        print(f"[OK] [{label}] completed successfully.")
        return True
    else:
        print(f"[FAILED] [{label}] failed with exit code {proc.returncode}. Log: {log_path}")
        return False

def _sizeof_dir(path: Path) -> str:
    if not path.exists():
        return "(not present)"
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if total < 1024**2:
            return f"{total/1024:.1f} KB"
        elif total < 1024**3:
            return f"{total/1024**2:.1f} MB"
        else:
            return f"{total/1024**3:.2f} GB"
    except Exception:
        return "(unknown)"

def interactive_cleanup_prompt():
    datasets_dir = PROJECT_ROOT / "datasets"
    runs_targets = [
        PROJECT_ROOT / "runs_comparison",
        PROJECT_ROOT / "logs",
        datasets_dir / "combined_carparts",
        datasets_dir / "matched",
        datasets_dir / "external",
    ]

    print()
    print("=" * 68)
    print(" Pipeline Startup — Cleanup Check")
    print("=" * 68)

    present_runs = [p for p in runs_targets if p.exists()]
    print()
    print("  [Q1] Delete old run outputs, logs, and derived dataset artifacts?")
    print("       (runs_comparison/, logs/, combined_carparts/, matched/, external/)")
    print("       Default: No")
    print()
    if present_runs:
        print("       Will remove:")
        for p in present_runs:
            size = _sizeof_dir(p)
            try:
                rel = p.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = p
            print(f"         DELETE  {rel}  [{size}]")
    else:
        print("       (nothing to remove — all targets already absent)")
    print()

    clean_runs = False
    if _CLI_ARGS.yes_clean_runs:
        print("       [AUTO] --yes-clean-runs flag set — proceeding with deletion.")
        clean_runs = True
    elif present_runs:
        try:
            ans = input("       Delete run outputs? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        clean_runs = ans in ("y", "yes")

    if clean_runs:
        for p in present_runs:
            print(f"  [DELETE] {p}")
            shutil.rmtree(p, ignore_errors=True)
        print("  [OK] Run outputs cleaned.")

    raw_dir = datasets_dir / "raw"
    raw_size = _sizeof_dir(raw_dir)
    print()
    print("  [Q2] Delete datasets/raw/? (your hand-annotated ground truth — IRREPLACEABLE!)")
    print(f"       Current size: {raw_size}")
    print("       Default: No. You must type 'yes' to confirm.")
    print()

    clean_raw = False
    if _CLI_ARGS.yes_clean_raw:
        print("       [AUTO] --yes-clean-raw flag set — proceeding with deletion.")
        clean_raw = True
    elif raw_dir.exists() and any(raw_dir.rglob("*.*")):
        try:
            ans = input("       Type 'yes' to DELETE datasets/raw/ (or anything else to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans == "yes":
            clean_raw = True

    if clean_raw:
        print(f"  [DELETE] {raw_dir}")
        shutil.rmtree(raw_dir, ignore_errors=True)
        print("  [OK] datasets/raw/ removed.")

    print("=" * 68)
    return clean_runs, clean_raw

def prompt_dataset_sources():
    """
    Ask user which dataset sources to include in training.
    Returns: (use_raw, use_external, combined_data_path)
      - use_raw: bool - include RAW_DATASET (custom carparts)
      - use_external: bool - include external datasets (carparts-seg, dsmlr)
      - combined_data_path: str or None - path to combined_data/ pre-cleaned COCO dir
    """
    print()
    print("======================================================================")
    print(" Select Dataset Sources")
    print("======================================================================")
    print("  1) RAW_DATASET only (your custom annotated data)")
    print("  2) External datasets only (carparts-seg + dsmlr from GitHub)")
    print("  3) BOTH (RAW_DATASET + External datasets) [DEFAULT - RECOMMENDED]")
    print("  4) combined_data  (pre-cleaned COCO, 32 classes — Mask2Former only)")
    print("======================================================================")
    print("  NOTE: Selecting BOTH gives ~5689 images vs ~1455 for RAW only")
    print("        External datasets are cached — no re-download if already present")
    print("  NOTE: combined_data skips dataset prep entirely (already cleaned).")
    print("        YOLO training is skipped for combined_data (COCO-only format).")
    print("======================================================================")
    choice = prompt("Enter choice [1-4] (press Enter for 3 - BOTH): ") or "3"

    if choice == "1":
        print("  [Selected] RAW_DATASET only (1,455 images)")
        return True, False, None
    elif choice == "2":
        print("  [Selected] External datasets only")
        return False, True, None
    elif choice == "4":
        _combined_path = str(PROJECT_ROOT / "RAW_DATASET" / "combined_data")
        print(f"  [Selected] combined_data (pre-cleaned COCO, 32 classes)")
        print(f"             Path: {_combined_path}")
        print("  [NOTE] YOLO training will be SKIPPED for this dataset source.")
        return False, False, _combined_path
    else:  # choice == "3" or default
        print("  [Selected] BOTH datasets (~5,689 images)")
        return True, True, None


def ensure_and_prepare_datasets(logs_dir, use_raw=True, use_external=True):
    """
    Prepare datasets based on user selection.
    
    Args:
        logs_dir: Path to logs directory
        use_raw: Include RAW_DATASET (custom carparts)
        use_external: Include external datasets (carparts-seg, dsmlr)
    """
    datasets_dir = PROJECT_ROOT / "datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "01_smart_dataset_prep.log")

    # Step 1: RAW_DATASET/01_orchvate_custom -> datasets/raw/ (only if selected)
    # RAW_DATASET is the permanent safe storage. We copy/process from it — never modify it.
    if use_raw:
        orchvate_dir = PROJECT_ROOT / "RAW_DATASET" / "01_orchvate_custom"
        raw_images = orchvate_dir / "IMAGES"
        raw_xmls   = orchvate_dir / "XML"
        if raw_images.exists() and any(raw_images.iterdir()) and raw_xmls.exists() and any(raw_xmls.iterdir()):
            print("[*] Found RAW_DATASET/01_orchvate_custom — converting XML annotations to YOLO format...")
            run_cmd_and_log(
                ["python", "scripts/data/prepare_raw_dataset.py",
                 "--raw-dir", str(orchvate_dir)],
                log_path, "raw_prep"
            )
        else:
            print(f"[WARN] RAW_DATASET/01_orchvate_custom not found or empty — skipping custom data.")

        raw_dir = datasets_dir / "raw"
        if not raw_dir.exists() or not any((raw_dir / "images").rglob("*.*") if (raw_dir / "images").exists() else []):
            print("[*] Migrating deduplicated HITL data to datasets/raw/ ...")
            run_cmd_and_log(["python", "scripts/data/setup_dataset_structure.py"], log_path, "setup_structure")
    else:
        print("[*] Skipping RAW_DATASET (user choice: external only)")

    # Step 2: External datasets (carparts-seg, DSMLR) (only if selected)
    if use_external:
        carparts_ext = datasets_dir / "external" / "carparts-seg"
        # Check if it has actual images (not just empty scaffold dirs created by setup_structure)
        def _has_images(path):
            if not path.exists():
                return False
            return any(path.rglob("*.jpg")) or any(path.rglob("*.jpeg")) or any(path.rglob("*.png"))
        if not _has_images(carparts_ext):
            print("\n[*] datasets/external/carparts-seg/ not found — downloading from Ultralytics...")
            try:
                import urllib.request, zipfile, shutil
                zip_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip"
                zip_file = datasets_dir / "carparts-seg.zip"

                # Download with progress output
                def _reporthook(count, block_size, total_size):
                    if total_size > 0:
                        pct = min(count * block_size * 100 // total_size, 100)
                        if pct % 10 == 0:
                            print(f"    Downloading carparts-seg.zip ... {pct}%", flush=True)

                print(f"    URL: {zip_url}")
                urllib.request.urlretrieve(zip_url, zip_file, reporthook=_reporthook)
                print(f"    Downloaded: {zip_file} ({zip_file.stat().st_size // 1024 // 1024} MB)")

                # Extract into a temp dir to avoid interactive overwrite prompts
                # (Python's zipfile module never asks — unlike the unzip CLI)
                temp_dir = datasets_dir / "_carparts_seg_extract_tmp"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True)

                print(f"    Extracting to {temp_dir} ...")
                with zipfile.ZipFile(zip_file, "r") as zf:
                    zf.extractall(temp_dir)
                zip_file.unlink(missing_ok=True)

                # Check if extracted directly into temp_dir or inside a subfolder
                if (temp_dir / "carparts-seg").exists():
                    extracted = temp_dir / "carparts-seg"
                elif (temp_dir / "images").exists() or (temp_dir / "labels").exists():
                    extracted = temp_dir
                else:
                    candidates = [d for d in temp_dir.iterdir() if d.is_dir() and ((d / "images").exists() or (d / "labels").exists())]
                    extracted = candidates[0] if candidates else temp_dir

                # Move to final destination
                print(f"    Moving to datasets/external/carparts-seg/ ...")
                carparts_ext.parent.mkdir(parents=True, exist_ok=True)
                if carparts_ext.exists():
                    shutil.rmtree(carparts_ext)
                shutil.move(str(extracted), str(carparts_ext))
                shutil.rmtree(temp_dir, ignore_errors=True)

                n_imgs = len(list(carparts_ext.rglob("*.jpg"))) + len(list(carparts_ext.rglob("*.png")))
                print(f"    [OK] datasets/external/carparts-seg/ ready ({n_imgs} images)")

            except Exception as e:
                print(f"[WARN] Could not auto-download carparts-seg: {e}")
                import traceback
                traceback.print_exc()

        dsmlr_ext = datasets_dir / "external" / "dsmlr"
        if not dsmlr_ext.exists() or not any(dsmlr_ext.iterdir()):
            dsmlr_raw = datasets_dir / "dsmlr-carparts"
            if not dsmlr_raw.exists() or not any(dsmlr_raw.iterdir()):
                cmd = ["git", "clone", "--depth", "1", "https://github.com/dsmlr/Car-Parts-Segmentation.git", str(dsmlr_raw)]
                run_cmd_and_log(cmd, log_path, "dsmlr_clone")
            if dsmlr_raw.exists() and any(dsmlr_raw.iterdir()):
                run_cmd_and_log(["python", "scripts/data/prepare_dsmlr_split.py"], log_path, "dsmlr_split")
                run_cmd_and_log(["python", "scripts/data/setup_dataset_structure.py"], log_path, "setup_structure")

        # Step 3: Match taxonomy — run matchers if source exists and destination is empty
        matched_cs   = datasets_dir / "matched" / "carparts-seg"
        matched_dsmlr = datasets_dir / "matched" / "dsmlr"

        # carparts-seg: check source exists with images, dest has none
        carparts_ext = datasets_dir / "external" / "carparts-seg"
        carparts_imgs_dir = carparts_ext / "images"
        if carparts_imgs_dir.exists():
            # Count files directly in images/train (fast, no deep scan)
            train_dir = carparts_imgs_dir / "train"
            n_src = len(list(train_dir.iterdir())) if train_dir.exists() else 0
            print(f"[*] external/carparts-seg/images/train: {n_src} files")
            n_dst = len(list((matched_cs / "images" / "train").iterdir())) if (matched_cs / "images" / "train").exists() else 0
            print(f"[*] matched/carparts-seg/images/train: {n_dst} files")
            if n_src > 100 and n_dst == 0:
                run_cmd_and_log(["python", "scripts/data/match_carparts_seg.py"], log_path, "match_carparts_seg")
            elif n_dst > 0:
                print(f"[SKIP] matched/carparts-seg already has {n_dst} images.")
            else:
                print(f"[WARN] external/carparts-seg/images/train has only {n_src} files — skipping.")
        else:
            print(f"[WARN] external/carparts-seg/images/ not found — carparts-seg download may have failed.")

        # Check if dsmlr needs matching (has external source AND matched dir is empty)
        if (datasets_dir / "external" / "dsmlr").exists():
            matched_dsmlr_populated = (matched_dsmlr / "images").exists() and any((matched_dsmlr / "images").rglob("*.*"))
            if not matched_dsmlr_populated:
                run_cmd_and_log(["python", "scripts/data/match_dsmlr.py"], log_path, "match_dsmlr")
            else:
                print("[SKIP] matched/dsmlr already populated.")
    else:
        print("[*] Skipping external datasets (user choice: RAW_DATASET only)")

    # Step 4: Combine datasets (with selected sources)
    # Print source summary so the user can catch missing data before the long combine step
    print("\n[*] Dataset sources ready for combining:")
    for src_name, src_path in [
        ("raw",              datasets_dir / "raw"),
        ("carparts-seg",     datasets_dir / "matched" / "carparts-seg"),
        ("dsmlr",            datasets_dir / "matched" / "dsmlr"),
    ]:
        if src_path.exists():
            n = len(list(src_path.rglob("*.jpg"))) + len(list(src_path.rglob("*.png")))
            print(f"    {src_name:<20}: {n} images")
        else:
            print(f"    {src_name:<20}: NOT FOUND (will be skipped)")

    print("\n[*] Rebuilding combined_carparts from selected sources...")
    cmd = ["python", "scripts/data/combine_datasets.py"]
    if not use_raw:
        cmd.append("--no-raw")
    if not use_external:
        cmd.append("--no-external")
    run_cmd_and_log(cmd, log_path, "combine_datasets")

    # Step 5: Generate COCO JSON
    print("\n[*] Regenerating COCO JSON for Mask R-CNN / Mask2Former...")
    run_cmd_and_log(["python", "scripts/data/yolo_to_coco.py", "--dataset", "combined_carparts"], log_path, "yolo_to_coco")

    # Step 6: Verify labels
    print("\n[*] Verifying labels (corrupt image scan)...")
    run_cmd_and_log(["python", "scripts/data/verify_labels.py", "--dir", str(datasets_dir / "combined_carparts")], log_path, "verify_labels")

    print("\n[OK] Dataset preparation complete. Training set: './datasets/combined_carparts'")
    return "combined_carparts"

def analyze_dataset(dataset_name="combined_carparts", output_dir=None):
    ds_path = PROJECT_ROOT / "datasets" / dataset_name
    if not ds_path.exists():
        print(f"[ERROR] Dataset directory '{ds_path}' does not exist.")
        return

    print()
    print("======================================================================")
    print(f" Dataset Analytics Summary -- '{dataset_name}'")
    print("======================================================================")

    total_images = 0
    total_annotations = 0
    class_counts = {}

    for split in ["train", "val", "test"]:
        coco_path = ds_path / f"coco_{split}.json"
        if coco_path.exists():
            try:
                with open(coco_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                n_img = len(data.get("images", []))
                n_ann = len(data.get("annotations", []))
                total_images += n_img
                total_annotations += n_ann
                cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
                for ann in data.get("annotations", []):
                    cat_id = ann.get("category_id")
                    cat_name = cat_map.get(cat_id, f"class_{cat_id}")
                    class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
                print(f"  Split [{split:<5}]: {n_img:>6} images | {n_ann:>7} annotations")
            except Exception as e:
                print(f"  [WARN] Failed to parse {coco_path.name}: {e}")
        else:
            img_dir = ds_path / "images" / split
            if img_dir.exists():
                n_img = len(list(img_dir.glob("*.*")))
                total_images += n_img
                print(f"  Split [{split:<5}]: {n_img:>6} images (YOLO raw format)")

    print("----------------------------------------------------------------------")
    print(f"  TOTAL     : {total_images:>6} images | {total_annotations:>7} annotations")
    print("======================================================================")

    if class_counts:
        print("\n Annotations Per Label Class:")
        print(" ---------------------------------------------------------------------")
        print(f"  {'Category Name':<28} | {'Annotations':<12} | {'Percentage':<10}")
        print(" ---------------------------------------------------------------------")
        for name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"  {name:<28} | {count:>12} | {pct:>9.2f}%")
        print(" ---------------------------------------------------------------------")

    # Export multi-sheet Excel report
    try:
        from scripts.data.export_dataset_analytics import export_analytics_excel
        excel_path = export_analytics_excel(ds_path, output_path=output_dir)
        if excel_path:
            print(f"\n[OK] Dataset Analytics Excel Report saved to:")
            print(f"     {excel_path}")
    except Exception as e:
        print(f"\n[WARN] Could not export Excel report: {e}")


def _generate_report(run_config_path, logs_dir):
    try:
        cmd = ["python", "scripts/data/generate_pipeline_report.py", "--run_config", run_config_path]
        log_path = os.path.join(logs_dir, "pipeline_report.log")
        run_cmd_and_log(cmd, log_path, "generate_pipeline_report")
    except Exception as e:
        print(f"[WARN] Failed to generate pipeline report: {e}")

def save_run_config(logs_dir, task_name, dataset, models=None, hparams=None, extra=None):
    run_id = os.path.basename(os.path.dirname(logs_dir))
    selected_models = []
    if models:
        if getattr(models, "yolo", False): selected_models.append("yolo11m-seg")
        if getattr(models, "maskrcnn", False): selected_models.append("maskrcnn")
        if getattr(models, "mask2former", False): selected_models.append("mask2former")

    hp_dict = {}
    if hparams:
        hp_dict = {
            "yolo": {
                "epochs": getattr(hparams, "yolo_epochs", None),
                "batch": getattr(hparams, "yolo_batch", None),
                "workers": getattr(hparams, "yolo_workers", None)
            },
            "maskrcnn": {
                "epochs": getattr(hparams, "mrcnn_epochs", None),
                "batch": getattr(hparams, "mrcnn_batch", None),
                "workers": getattr(hparams, "mrcnn_workers", None),
                "accum_steps": getattr(hparams, "mrcnn_accum_steps", 4)
            },
            "mask2former": {
                "epochs": getattr(hparams, "m2f_epochs", None),
                "batch": getattr(hparams, "m2f_batch", None),
                "workers": getattr(hparams, "m2f_workers", None),
                "accum_steps": getattr(hparams, "m2f_accum_steps", 4),
                "compile": getattr(hparams, "m2f_compile", False)
            },
        }
        hp_dict = {k: v for k, v in hp_dict.items() if any(val is not None for val in v.values())}

    config = {
        "run_id": run_id,
        "task": task_name,
        "started_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "models_selected": selected_models,
        "hyperparameters": hp_dict,
    }
    if extra:
        config.update(extra)

    config_path = os.path.join(logs_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\n[OK] Run config saved -> {config_path}")
    return config_path

def check_azure_disk():
    cwd_str = str(PROJECT_ROOT)
    if cwd_str.startswith("/mnt/"):
        print("############################################################", file=sys.stderr)
        print(f"  WARNING: Running from {cwd_str}", file=sys.stderr)
        print("  If this is Azure's temporary resource disk, it will be WIPED", file=sys.stderr)
        print("  when you Stop (Deallocate) the VM. Move to /home first!", file=sys.stderr)
        print("############################################################", file=sys.stderr)
        import time
        time.sleep(3)

def run_evaluation(dataset, base_out_dir, logs_dir):
    print("\n========================================")
    print(" Running Evaluation & Comparison")
    print("========================================")

    yolo_weights = str(Path(base_out_dir) / "yolo11m-seg" / "weights" / "best.pt")
    mrcnn_weights = str(Path(base_out_dir) / "maskrcnn" / "weights" / "best.pt")
    m2f_weights = str(Path(base_out_dir) / "mask2former" / "weights" / "best")

    eval_jobs = [
        ("06a_eval_yolo", ["--model", "yolo", "--weights", yolo_weights, "--dataset", dataset]),
        ("06b_eval_maskrcnn", ["--model", "maskrcnn", "--weights", mrcnn_weights, "--dataset", dataset]),
        ("06c_eval_mask2former", ["--model", "mask2former", "--weights", m2f_weights, "--dataset", dataset]),
    ]

    for label, eval_args in eval_jobs:
        weights_path = Path(eval_args[3])
        if not weights_path.exists():
            continue
        log_path = os.path.join(logs_dir, f"{label}.log")
        cmd = ["python", "scripts/evaluation/evaluate_confusion_matrix.py"] + eval_args
        run_cmd_and_log(cmd, log_path, label)

    # Run unified comparison pipeline
    log_path = os.path.join(logs_dir, "07_compare_models.log")
    cmd = ["python", "scripts/evaluation/compare_models.py", "--run_dir", base_out_dir, "--out_dir", base_out_dir, "--non_interactive"]
    print("[*] Generating Model Comparison Report & Plots...")
    run_cmd_and_log(cmd, log_path, "compare_models")

def main():
    check_azure_disk()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = str(PROJECT_ROOT / "runs_comparison" / f"run_{timestamp}")
    logs_dir = os.path.join(base_out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    with open(os.path.join(logs_dir, "orchestrator.log"), "w", encoding="utf-8") as f:
        f.write(f"Run started at {datetime.datetime.now()}\n")
        f.write(f"Project root: {PROJECT_ROOT}\n")

    def log_selection(label, value):
        """Write user selection to orchestrator.log and print it."""
        msg = f"[USER SELECTION] {label}: {value}"
        print(msg)
        with open(os.path.join(logs_dir, "orchestrator.log"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    display_main_menu()
    choice = prompt("Enter choice [1-12]: ")
    log_selection("Main Menu", f"Option {choice}")

    if choice == "1":
        print("\n[TASK] Train Models Locally (Dataset Prep -> Train -> Eval -> Compare)")
        models = collect_models()
        hparams = Hyperparams(models)
        # Ask user which dataset sources to use
        use_raw, use_external, combined_data_path = prompt_dataset_sources()

        # Log all user selections
        model_list = []
        if models.yolo: model_list.append("yolo11m-seg")
        if models.maskrcnn: model_list.append("maskrcnn")
        if models.mask2former: model_list.append("mask2former")
        log_selection("Models selected", ", ".join(model_list) if model_list else "none")
        log_selection("Dataset sources", f"use_raw={use_raw}, use_external={use_external}, combined_data={combined_data_path}")
        if models.yolo:
            log_selection("YOLO hyperparams", f"epochs={hparams.yolo_epochs}, batch={hparams.yolo_batch}, workers={hparams.yolo_workers}")
        if models.maskrcnn:
            log_selection("MaskRCNN hyperparams", f"epochs={hparams.mrcnn_epochs}, batch={hparams.mrcnn_batch}, workers={hparams.mrcnn_workers}, accum={hparams.mrcnn_accum_steps}")
        if models.mask2former:
            log_selection("Mask2Former hyperparams", f"epochs={hparams.m2f_epochs}, batch={hparams.m2f_batch}, workers={hparams.m2f_workers}, accum={hparams.m2f_accum_steps}")

        if combined_data_path:
            # combined_data mode: skip normal dataset prep entirely
            print(f"\n[*] combined_data mode — skipping dataset prep (already cleaned).")
            dataset = combined_data_path  # used only for eval/logging; Mask2Former uses --combined_data
        else:
            dataset = ensure_and_prepare_datasets(logs_dir, use_raw, use_external)
        run_config_path = save_run_config(logs_dir, "Train Models Locally", dataset, models, hparams)
        _generate_report(run_config_path, logs_dir)

        if models.yolo:
            if combined_data_path:
                print("[INFO] Skipping YOLO training — combined_data is COCO format only "
                      "(no YOLO .txt labels available). Use option 1/2/3 for YOLO training.")
            else:
                log_path = os.path.join(logs_dir, "02_train_yolo.log")
                out_dir = os.path.join(base_out_dir, "yolo11m-seg")
                cmd = ["python", "scripts/training/train_yolo_seg.py", "--model", "yolo11m-seg",
                       "--dataset", dataset, "--epochs", str(hparams.yolo_epochs),
                       "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers),
                       "--val_interval", str(hparams.yolo_val_interval),
                       "--output_dir", out_dir]
                run_cmd_and_log(cmd, log_path, "train_yolo")

        if models.maskrcnn:
            if combined_data_path:
                print("[INFO] Skipping Mask R-CNN training for combined_data mode.")
            else:
                log_path = os.path.join(logs_dir, "03_train_maskrcnn.log")
                out_dir = os.path.join(base_out_dir, "maskrcnn")
                cmd = ["python", "scripts/training/train_maskrcnn.py", "--dataset", dataset,
                       "--epochs", str(hparams.mrcnn_epochs), "--batch", str(hparams.mrcnn_batch),
                       "--num_workers", str(hparams.mrcnn_workers), "--output_dir", out_dir,
                       "--val_interval", str(hparams.mrcnn_val_interval),
                       "--accum_steps", str(hparams.mrcnn_accum_steps)]
                run_cmd_and_log(cmd, log_path, "train_maskrcnn")

        if models.mask2former:
            log_path = os.path.join(logs_dir, "04_train_mask2former.log")
            out_dir = os.path.join(base_out_dir, "mask2former")
            if combined_data_path:
                # Pass combined_data path directly; --dataset is still required by argparse
                # so we pass a dummy value that won't be used when --combined_data is set.
                cmd = ["python", "scripts/training/train_mask2former.py",
                       "--dataset", "combined_data_placeholder",
                       "--combined_data", combined_data_path,
                       "--epochs", str(hparams.m2f_epochs), "--batch", str(hparams.m2f_batch),
                       "--num_workers", str(hparams.m2f_workers), "--output_dir", out_dir,
                       "--val_interval", str(hparams.m2f_val_interval),
                       "--accum_steps", str(hparams.m2f_accum_steps),
                       "--no_object_weight", str(getattr(hparams, "m2f_no_obj_weight", 0.25))]
            else:
                cmd = ["python", "scripts/training/train_mask2former.py", "--dataset", dataset,
                       "--epochs", str(hparams.m2f_epochs), "--batch", str(hparams.m2f_batch),
                       "--num_workers", str(hparams.m2f_workers), "--output_dir", out_dir,
                       "--val_interval", str(hparams.m2f_val_interval),
                       "--accum_steps", str(hparams.m2f_accum_steps),
                       "--no_object_weight", str(getattr(hparams, "m2f_no_obj_weight", 0.25))]
            if hparams.m2f_compile:
                cmd.append("--compile")
            run_cmd_and_log(cmd, log_path, "train_mask2former")

        run_evaluation(dataset, base_out_dir, logs_dir)

    elif choice == "2":
        print("\n[TASK] Compare Models (YOLO11m-seg / Mask R-CNN / Mask2Former)")
        default_test = str(PROJECT_ROOT / "test_data")
        run_test = prompt("  Run test-data inference on images? [y/N] (default: n): ").lower() in ("y", "yes")

        test_images_arg = ""
        conf_arg = "0.40"
        if run_test:
            test_dir_input = prompt_default("  Test images directory", default_test)
            conf_input = prompt_default("  Detection confidence threshold", "0.40")
            test_images_arg = test_dir_input
            conf_arg = conf_input

        cmd = ["python", "scripts/evaluation/compare_models.py", "--out_dir", base_out_dir]
        if run_test and test_images_arg:
            cmd.extend(["--test_images", test_images_arg, "--conf", conf_arg])
        else:
            cmd.append("--skip_test_data")

        log_path = os.path.join(logs_dir, "00_compare_models.log")
        run_cmd_and_log(cmd, log_path, "compare_models")

    elif choice == "3":
        print("\n[TASK] Test on Trained Model (Inference)")

        try:
            from scripts.inference.infer_both_models import (
                _default_yolo_weights,
                _default_mask2former_weights,
                _default_maskrcnn_weights,
            )
            yolo_w = _default_yolo_weights()
            m2f_w = _default_mask2former_weights()
            mrcnn_w = _default_maskrcnn_weights()
        except Exception:
            yolo_w = "runs_comparison/yolo11m-seg/weights/best.pt"
            m2f_w = "runs_comparison/mask2former/weights/best"
            mrcnn_w = "runs_comparison/maskrcnn/weights/best.pt"

        yolo_found = os.path.exists(yolo_w)
        m2f_found = os.path.exists(m2f_w)
        mrcnn_found = os.path.exists(mrcnn_w)

        print("\nDetected Model Weights:")
        print(f"  - YOLOv11m-seg : {yolo_w} [{'FOUND' if yolo_found else 'NOT FOUND'}]")
        print(f"  - Mask2Former  : {m2f_w} [{'FOUND' if m2f_found else 'NOT FOUND'}]")
        print(f"  - Mask R-CNN   : {mrcnn_w} [{'FOUND' if mrcnn_found else 'NOT FOUND'}]")
        print()

        default_test_dir = "./test_data" if os.path.exists("./test_data") else "./test"
        inp = prompt_default("Test images directory", default_test_dir)
        out = prompt_default("Output directory", "./test_result")
        conf = prompt_default("Confidence threshold", "0.55")

        print("\nSelect Model(s) to Run:")
        print("  1) All Available Models (auto-skip missing weights) [Default]")
        print("  2) Mask2Former only")
        print("  3) YOLOv11m-seg only")
        print("  4) Mask R-CNN only")
        print("  5) Custom weights selection")
        m_choice = prompt("Enter choice [1-5] (default 1): ") or "1"

        cmd = ["python", "scripts/inference/infer_both_models.py", "--input", inp, "--output", out, "--conf", conf]

        if m_choice == "2":
            cmd.extend(["--skip_yolo", "--skip_maskrcnn"])
            if m2f_w:
                cmd.extend(["--mask2former_weights", m2f_w])
        elif m_choice == "3":
            cmd.extend(["--skip_mask2former", "--skip_maskrcnn"])
            if yolo_w:
                cmd.extend(["--yolo_weights", yolo_w])
        elif m_choice == "4":
            cmd.extend(["--skip_yolo", "--skip_mask2former"])
            if mrcnn_w:
                cmd.extend(["--maskrcnn_weights", mrcnn_w])
        elif m_choice == "5":
            use_yolo = prompt("  Run YOLO? [y/N]: ").strip().lower() in ("y", "yes")
            if use_yolo:
                yw = prompt_default("  YOLO weights path", yolo_w)
                cmd.extend(["--yolo_weights", yw])
            else:
                cmd.append("--skip_yolo")

            use_m2f = prompt("  Run Mask2Former? [y/N]: ").strip().lower() in ("y", "yes")
            if use_m2f:
                mw = prompt_default("  Mask2Former weights dir", m2f_w)
                cmd.extend(["--mask2former_weights", mw])
            else:
                cmd.append("--skip_mask2former")

            use_mrcnn = prompt("  Run Mask R-CNN? [y/N]: ").strip().lower() in ("y", "yes")
            if use_mrcnn:
                rw = prompt_default("  Mask R-CNN weights path", mrcnn_w)
                cmd.extend(["--maskrcnn_weights", rw])
            else:
                cmd.append("--skip_maskrcnn")
        else:
            if yolo_w:
                cmd.extend(["--yolo_weights", yolo_w])
            if m2f_w:
                cmd.extend(["--mask2former_weights", m2f_w])
            if mrcnn_w:
                cmd.extend(["--maskrcnn_weights", mrcnn_w])

        log_path = os.path.join(logs_dir, "01_inference.log")
        run_cmd_and_log(cmd, log_path, "inference")

    elif choice == "4":
        print("\n[TASK] Auto-Prepare Dataset")
        # Ask user which dataset sources to use
        use_raw, use_external = prompt_dataset_sources()
        combined_name = ensure_and_prepare_datasets(logs_dir, use_raw, use_external)
        run_config_path = save_run_config(logs_dir, "Auto-Prepare Dataset", combined_name)
        _generate_report(run_config_path, logs_dir)
        print(f"\n[OK] Dataset ready in './datasets/{combined_name}'!")
        analyze_dataset(combined_name, output_dir=base_out_dir)

    elif choice == "5":
        print("\n[TASK] Dataset Analytics")
        ds = prompt_default("Dataset name (under ./datasets)", "combined_carparts")
        analyze_dataset(ds, output_dir=base_out_dir)

    elif choice == "6":
        print("\n[TASK] Export CVAT XML & ZIP Package")
        ds = prompt_default("Dataset name (under ./datasets)", "combined_carparts")
        out_dir = prompt_default("Output folder", os.path.join(base_out_dir, f"{ds}_xml"))
        log_path = os.path.join(logs_dir, "export_cvat_xml.log")
        cmd = ["python", "scripts/data/export_dataset_xml.py", "--dataset", ds, "--out_dir", out_dir]
        run_cmd_and_log(cmd, log_path, "export_dataset_xml")
        print(f"\n[OK] CVAT XML export & ZIP package completed!")
        print(f"     Folder: {out_dir}")
        print(f"     ZIP   : {out_dir}.zip")

    elif choice == "7":
        print("\n[TASK] Dataset Preprocessing Tools")
        print("  a) yolo_to_coco    -- Convert YOLO polygons -> COCO JSON (parallel)")
        print("  b) combine         -- Merge multiple YOLO datasets (parallel copy)")
        print("  c) verify_labels   -- Scan for corrupt images (report-only)")
        print("  d) hash_dataset    -- Write blake3 checksums manifest")
        print("  e) auto_annotate   -- Auto-annotate images (Grounding DINO + SAM2)")
        print("  f) export_xml      -- Export combined dataset -> CVAT XML 1.1 & ZIP archive")
        sub = prompt("Enter sub-option [a-f]: ")
        if sub == "a":
            ds = prompt_default("Dataset", "combined_carparts")
            workers = prompt_default("Workers", "8")
            cmd = ["python", "scripts/data/yolo_to_coco.py", "--dataset", ds, "--num_workers", str(workers)]
            run_cmd_and_log(cmd, os.path.join(logs_dir, "preprocess_yolo_to_coco.log"), "yolo_to_coco")
        elif sub == "b":
            dirs_raw = prompt_default("Dataset dirs (comma-separated)", "datasets/carparts-seg,datasets/custom_carparts")
            out = prompt_default("Output dir", "datasets/combined_carparts")
            workers = prompt_default("Workers", "8")
            cmd = ["python", "scripts/data/combine_datasets.py", "--datasets", dirs_raw, "--out_dir", out, "--workers", str(workers)]
            run_cmd_and_log(cmd, os.path.join(logs_dir, "preprocess_combine.log"), "combine_datasets")
        elif sub == "c":
            dir_raw = prompt_default("Dataset directory to verify", "datasets/combined_carparts")
            cmd = ["python", "scripts/data/verify_labels.py", "--dir", dir_raw]
            run_cmd_and_log(cmd, os.path.join(logs_dir, "preprocess_verify.log"), "verify_labels")
        elif sub == "d":
            dir_raw = prompt_default("Directory to hash", "datasets/combined_carparts")
            out_manifest = prompt_default("Output manifest path", f"{dir_raw}/checksums.blake3")
            workers = prompt_default("Workers", "8")
            cmd = ["python", "scripts/data/hash_dataset.py", "--dir", dir_raw, "--out", out_manifest, "--workers", str(workers)]
            run_cmd_and_log(cmd, os.path.join(logs_dir, "preprocess_hash.log"), "hash_dataset")
        elif sub == "e":
            inp = prompt_default("Input images directory", "./RAW_DATASET/IMAGES")
            out = prompt_default("Output directory", "./datasets/auto_annotated")
            log_path = os.path.join(logs_dir, "01_annotation.log")
            cmd = ["python", "scripts/inference/auto_annotate_carparts.py", "--input", inp, "--output", out]
            run_cmd_and_log(cmd, log_path, "annotation")
        elif sub == "f":
            ds = prompt_default("Dataset name (under ./datasets)", "combined_carparts")
            out_dir = prompt_default("Output folder", os.path.join(base_out_dir, f"{ds}_xml"))
            log_path = os.path.join(logs_dir, "preprocess_export_xml.log")
            cmd = ["python", "scripts/data/export_dataset_xml.py", "--dataset", ds, "--out_dir", out_dir]
            run_cmd_and_log(cmd, log_path, "export_dataset_xml")
            print(f"\n[OK] CVAT XML export & ZIP package completed!")
            print(f"     Folder: {out_dir}")
            print(f"     ZIP   : {out_dir}.zip")
        else:
            print(f"[ERROR] Unknown sub-option '{sub}'")

    elif choice == "8":
        print("\n[TASK] GPU Capacity Check")
        mode = prompt_default("Run mode (local / azure)", "local")
        ds = prompt_default("Dataset path", "./datasets/combined_carparts")
        log_path = os.path.join(logs_dir, "00_capacity_check.log")
        cmd = ["python", "scripts/training/capacity_check.py", "--dataset", ds, "--mode", mode]
        run_cmd_and_log(cmd, log_path, "capacity_check")

    elif choice == "9":
        print("\n[TASK] Train on Azure ML")
        is_dryrun = prompt("  Run 3-epoch quick pipeline check (dry run)? [y/N] (default: n): ").lower() in ("y", "yes")
        local_dir = prompt_default("Local dataset directory", "./datasets/combined_carparts")

        if is_dryrun:
            cmd = ["python", "scripts/training/azure_train.py", "--model", "all",
                   "--local_dataset_dir", local_dir, "--epochs", "3",
                   "--batch", "-1", "--workers", "8", "--auto_upload"]
            run_cmd_and_log(cmd, os.path.join(logs_dir, "00_azure_dryrun.log"), "azure_dryrun")
        else:
            models = collect_models()
            hparams = Hyperparams(models)
            all_selected = (models.yolo and models.maskrcnn and models.mask2former)
            if all_selected:
                log_path = os.path.join(logs_dir, "00_azure_train_all.log")
                cmd = ["python", "scripts/training/azure_train.py", "--model", "all",
                       "--local_dataset_dir", local_dir, "--epochs", str(hparams.yolo_epochs),
                       "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers), "--auto_upload"]
                run_cmd_and_log(cmd, log_path, "azure_train_all")
            else:
                if models.yolo:
                    cmd = ["python", "scripts/training/azure_train.py", "--model", "yolo11m-seg",
                           "--local_dataset_dir", local_dir, "--epochs", str(hparams.yolo_epochs),
                           "--batch", str(hparams.yolo_batch), "--workers", str(hparams.yolo_workers), "--auto_upload"]
                    run_cmd_and_log(cmd, os.path.join(logs_dir, "00_azure_train_yolo.log"), "azure_train_yolo")
                if models.maskrcnn:
                    cmd = ["python", "scripts/training/azure_train.py", "--model", "maskrcnn",
                           "--local_dataset_dir", local_dir, "--epochs", str(hparams.mrcnn_epochs),
                           "--batch", str(hparams.mrcnn_batch), "--workers", str(hparams.mrcnn_workers), "--auto_upload"]
                    run_cmd_and_log(cmd, os.path.join(logs_dir, "00_azure_train_maskrcnn.log"), "azure_train_maskrcnn")
                if models.mask2former:
                    cmd = ["python", "scripts/training/azure_train.py", "--model", "mask2former",
                           "--local_dataset_dir", local_dir, "--epochs", str(hparams.m2f_epochs),
                           "--batch", str(hparams.m2f_batch), "--workers", str(hparams.m2f_workers), "--auto_upload"]
                    run_cmd_and_log(cmd, os.path.join(logs_dir, "00_azure_train_mask2former.log"), "azure_train_mask2former")

    elif choice == "10":
        print("\n[TASK] Cleanup Pipeline")
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "cleanup_pipeline",
            str(PROJECT_ROOT / "scripts" / "data" / "cleanup_pipeline.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.main()

    elif choice == "11":
        print("Exiting.")
        sys.exit(0)

    elif choice == "12":
        print()
        print("======================================================================")
        print(" Best Model Analytics -- Deep Per-Class Analysis & Export Bundle")
        print("======================================================================")
        print("  Scans all runs_comparison/ folders, finds the best trained model")
        print("  by highest mask mAP50, then generates:")
        print("    * Per-part mAP50 bar chart  (colour-coded green/amber/red tiers)")
        print("    * Radar chart               (precision vs recall per class)")
        print("    * Precision/Recall heatmap  (grid: class x metric)")
        print("    * Score distribution chart  (mAP50 vs mAP50-95 per part)")
        print("    * Ranked parts CSV          (sorted by mAP50)")
        print("    * Test inference results    (annotated images from test_data/)")
        print("    * analytics_summary.xlsx    (4-sheet Excel workbook)")
        print("    * Full export bundle        (weights + plots + comparison Excel)")
        print("======================================================================")
        print()

        # ── Run selection ──────────────────────────────────────────────────────
        runs_dir = PROJECT_ROOT / "runs_comparison"
        print("  [A] Run selection:")
        print("    1) Auto-detect best run across ALL runs_comparison/ folders [Default]")
        print("    2) Pick a specific run folder manually")
        run_sel = prompt("  Enter choice [1-2] (default 1): ") or "1"

        target_run_dir  = None
        target_model_dir = None

        if run_sel == "2":
            # List available runs
            avail_runs = sorted(
                [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            ) if runs_dir.exists() else []
            if not avail_runs:
                print("[ERROR] No run folders found in runs_comparison/")
            else:
                print()
                print("  Available runs:")
                for idx, rd in enumerate(avail_runs, 1):
                    # Try to read best score for display
                    score_str = ""
                    for md in rd.iterdir():
                        bm = md / "best_metrics.json" if md.is_dir() else None
                        if bm and bm.exists():
                            try:
                                with open(bm, "r", encoding="utf-8") as _f:
                                    _d = json.load(_f)
                                vm = _d.get("val_metrics", {})
                                s = vm.get("map50_mask", vm.get("map50_box",
                                    _d.get("best_mask_map50", 0.0)))
                                score_str = f"  [mAP50={s:.4f}  model={md.name}]"
                            except Exception:
                                pass
                    print(f"    {idx:>2}) {rd.name}{score_str}")
                print()
                run_idx_str = prompt(f"  Enter run number [1-{len(avail_runs)}]: ")
                try:
                    run_idx = int(run_idx_str) - 1
                    if 0 <= run_idx < len(avail_runs):
                        target_run_dir = avail_runs[run_idx]
                        print(f"  [Selected] {target_run_dir.name}")
                    else:
                        print("[WARN] Invalid selection — using auto-detect.")
                except ValueError:
                    print("[WARN] Invalid input — using auto-detect.")

        # ── Inference options ──────────────────────────────────────────────────
        default_test = str(PROJECT_ROOT / "test_data")
        print()
        skip_inf_ans = prompt("  Run test-image inference? [Y/n] (default: y): ").strip().lower()
        skip_inference = skip_inf_ans in ("n", "no")

        test_images_path = default_test
        conf_val = "0.35"
        if not skip_inference:
            test_images_path = prompt_default("  Test images directory", default_test)
            conf_val = prompt_default("  Inference confidence threshold", "0.35")

        # ── Output directory ───────────────────────────────────────────────────
        if target_run_dir:
            default_out = str(target_run_dir / "analytics_export")
        else:
            default_out = "<auto — inside best run folder>/analytics_export"
        print()
        print(f"  Output will be saved to: {default_out}")
        custom_out = prompt("  Custom output dir? (press Enter to use default): ").strip()

        # ── Build command ──────────────────────────────────────────────────────
        analytics_script = PROJECT_ROOT / "scripts" / "evaluation" / "best_model_analytics.py"
        cmd = [sys.executable, str(analytics_script)]

        if target_run_dir:
            cmd += ["--run_dir", str(target_run_dir)]
        else:
            cmd.append("--auto")

        if not skip_inference:
            cmd += ["--test_images", test_images_path, "--conf", conf_val]
        else:
            cmd.append("--skip_inference")

        if custom_out:
            cmd += ["--out_dir", custom_out]

        log_path = os.path.join(logs_dir, "12_best_model_analytics.log")
        log_selection("Best Model Analytics",
                      f"run={target_run_dir.name if target_run_dir else 'auto'}  "
                      f"skip_inference={skip_inference}  conf={conf_val}")
        ok = run_cmd_and_log(cmd, log_path, "best_model_analytics")

        if ok:
            if custom_out:
                bundle_path = custom_out
            elif target_run_dir:
                bundle_path = str(target_run_dir / "analytics_export")
            else:
                bundle_path = str(runs_dir / "<best_run>" / "analytics_export")
            print()
            print("  ╔══════════════════════════════════════════════════════════╗")
            print("  ║  EXPORT BUNDLE READY                                     ║")
            print("  ╠══════════════════════════════════════════════════════════╣")
            print(f"  ║  {bundle_path:<56}║")
            print("  ╠══════════════════════════════════════════════════════════╣")
            print("  ║  Contents:                                               ║")
            print("  ║    • per_class_analytics/  (4 charts + CSV)              ║")
            print("  ║    • test_results/          (annotated inference images)  ║")
            print("  ║    • analytics_summary.xlsx (4-sheet workbook)            ║")
            print("  ║    • model_comparison_report.xlsx                        ║")
            print("  ║    • best_model/             (weights copy)               ║")
            print("  ║    • comparison_plots/       (learning curves)            ║")
            print("  ║    • README_export.txt       (bundle index)               ║")
            print("  ╚══════════════════════════════════════════════════════════╝")

    else:
        print("[ERROR] Invalid choice.")

    print("\n==============================================")
    print(" TASK COMPLETE")
    print(f" Outputs : {base_out_dir}")
    print(f" Logs    : {logs_dir}")
    print("==============================================")

if __name__ == "__main__":
    main()
