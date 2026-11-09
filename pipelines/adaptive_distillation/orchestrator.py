#!/usr/bin/env python3
"""
Adaptive Multi-Teacher Distillation Orchestrator.
Master interactive entrypoint for the new pipeline:
- Teacher model discovery and selection from runs_comparison/
- Offline Phase A cache preparation
- Baseline E2 training
- Distillation training (E3, E4, E5, E6)
- Controlled ablation execution (C1, C2, C3)
- Final hypothesis evaluation and reporting
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.adaptive_distillation.config import DistillationPipelineConfig
from pipelines.adaptive_distillation.teachers.teacher_scanner import TeacherScanner


DEFAULT_CONFIG = PROJECT_ROOT / "pipelines" / "adaptive_distillation" / "configs" / "default.yaml"


def prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)


def run_command(cmd: list) -> int:
    print(f"\n[*] Executing: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd)
    return proc.returncode


def display_menu():
    print("\n" + "=" * 70)
    print(" GROUND-TRUTH-GUIDED ADAPTIVE MULTI-TEACHER DISTILLATION PIPELINE")
    print("=" * 70)
    print("  1) Interactive Unified Training Workflow (run_training.py) [Recommended]")
    print("  2) Select Teachers from runs_comparison/ & Prepare Cache (Phase A)")
    print("  3) Train GT-Only Scratch Student C (Baseline E2)")
    print("  4) Train Single-Teacher Distillation (E3 / E4)")
    print("  5) Train Fixed Multi-Teacher Distillation (E5)")
    print("  6) Train Adaptive Multi-Teacher Distillation (E6)")
    print("  7) Run Experimental Controls (C1 Shuffled / C2 Uniform / C3 Inverse)")
    print("  8) Evaluate Models & Validate Pre-Registered Hypotheses")
    print("  9) Validate Teacher Prediction Cache Integrity")
    print("  10) View Available Teacher Candidates in runs_comparison/")
    print("  0) Exit")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Adaptive Multi-Teacher Distillation Orchestrator")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to pipeline configuration YAML")
    parser.add_argument("--auto_run", choices=["cache", "e2", "e5", "e6", "eval"], default=None, help="Direct non-interactive run mode")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[!] Config not found at {cfg_path}, creating default.")
        DistillationPipelineConfig().save_yaml(cfg_path)

    if args.auto_run:
        if args.auto_run == "cache":
            run_command(["python3", "pipelines/adaptive_distillation/scripts/prepare_teacher_cache.py", "--config", str(cfg_path)])
        elif args.auto_run == "e2":
            run_command(["python3", "pipelines/adaptive_distillation/scripts/train_student.py", "--config", str(cfg_path), "--no_distill"])
        elif args.auto_run == "e5":
            run_command(["python3", "pipelines/adaptive_distillation/scripts/train_student.py", "--config", str(cfg_path), "--mode", "fixed", "--exp_name", "e5_fixed_multiteacher"])
        elif args.auto_run == "e6":
            run_command(["python3", "pipelines/adaptive_distillation/scripts/train_student.py", "--config", str(cfg_path), "--mode", "adaptive", "--exp_name", "e6_adaptive_multiteacher"])
        elif args.auto_run == "eval":
            run_command(["python3", "pipelines/adaptive_distillation/scripts/evaluate_experiments.py", "--config", str(cfg_path)])
        return

    scanner = TeacherScanner()

    while True:
        display_menu()
        choice = prompt("Select an option [0-9]: ")

        if choice == "0":
            print("Exiting.")
            break

        elif choice == "1":
            print("\n[TASK] Launching Interactive Unified Training Workflow")
            run_command([
                "python3", "pipelines/adaptive_distillation/run_training.py",
                "--config", str(cfg_path)
            ])

        elif choice == "2":
            print("\n[TASK] Select Teachers and Prepare Prediction Cache (Phase A)")
            run_command([
                "python3", "pipelines/adaptive_distillation/scripts/prepare_teacher_cache.py",
                "--config", str(cfg_path),
                "--select_teachers"
            ])

        elif choice == "3":
            print("\n[TASK] Train GT-Only Scratch Student C Baseline (E2)")
            epochs = prompt("Enter number of epochs (default from config): ")
            cmd = ["python3", "pipelines/adaptive_distillation/scripts/train_student.py", "--config", str(cfg_path), "--no_distill"]
            if epochs:
                cmd.extend(["--epochs", epochs])
            run_command(cmd)

        elif choice == "4":
            print("\n[TASK] Train Single-Teacher Distillation (E3 / E4)")
            t_name = prompt("Enter teacher name to distill from [Teacher_A / Teacher_B]: ") or "Teacher_A"
            t_canonical = "teacher_a_m2f" if "a" in t_name.lower() else "teacher_b_m2f"
            run_command([
                "python3", "pipelines/adaptive_distillation/scripts/train_student.py",
                "--config", str(cfg_path),
                "--mode", "fixed",
                "--teachers", t_canonical,
                "--exp_name", f"single_{t_name.lower()}"
            ])

        elif choice == "5":
            print("\n[TASK] Train Fixed Multi-Teacher Distillation (E5)")
            run_command([
                "python3", "pipelines/adaptive_distillation/scripts/train_student.py",
                "--config", str(cfg_path),
                "--mode", "fixed",
                "--exp_name", "e5_fixed_multiteacher"
            ])

        elif choice == "6":
            print("\n[TASK] Train Adaptive Multi-Teacher Distillation (E6)")
            run_command([
                "python3", "pipelines/adaptive_distillation/scripts/train_student.py",
                "--config", str(cfg_path),
                "--mode", "adaptive",
                "--exp_name", "e6_adaptive_multiteacher"
            ])

        elif choice == "7":
            print("\n[TASK] Run Experimental Controls")
            print("  1) C1: Shuffled Reliability Weights")
            print("  2) C2: Uniform within Gate")
            print("  3) C3: Inverse Reliability Weights")
            ctrl = prompt("Select control [1-3]: ")
            mode_map = {"1": "shuffled", "2": "uniform_in_gate", "3": "inverse"}
            c_mode = mode_map.get(ctrl, "shuffled")
            run_command([
                "python3", "pipelines/adaptive_distillation/scripts/train_student.py",
                "--config", str(cfg_path),
                "--mode", c_mode,
                "--exp_name", f"control_{c_mode}"
            ])

        elif choice == "8":
            print("\n[TASK] Evaluate Models and Validate Hypotheses")
            split = prompt("Evaluate on which split? [val/test] (default: val): ") or "val"
            run_command([
                "python3", "pipelines/adaptive_distillation/scripts/evaluate_experiments.py",
                "--config", str(cfg_path),
                "--split", split
            ])

        elif choice == "9":
            print("\n[TASK] Validate Cache Integrity")
            cfg = DistillationPipelineConfig.from_yaml(cfg_path)
            from pipelines.adaptive_distillation.cache.cache_validator import CacheValidator
            import json
            with open(cfg.dataset.train_json) as f:
                cdata = json.load(f)
            e_ids = [str(img["id"]) for img in cdata.get("images", [])]
            validator = CacheValidator(cfg.cache.dir, cfg)
            ok, errs, summary = validator.validate_cache(e_ids)
            if ok:
                print("\n[OK] Cache is fully valid and complete!")
            else:
                print(f"\n[ERROR] Found {len(errs)} cache validation errors:")
                for e in errs:
                    print(f"  - {e}")

        elif choice == "10":
            cands = scanner.scan()
            scanner.display_candidates(cands)


if __name__ == "__main__":
    main()

