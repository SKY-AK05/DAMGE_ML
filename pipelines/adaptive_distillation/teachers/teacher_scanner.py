"""
Teacher Scanner module for discovering and selecting candidate teacher models
directly from runs_comparison/ and associated directories.
Allows user to choose who will be teacher and how many teachers to use.
"""

import os
import csv
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from pipelines.adaptive_distillation.config import TeacherConfig
from pipelines.adaptive_distillation.teachers.teacher_loader import compute_path_sha256


@dataclass
class DiscoveredTeacher:
    index: int
    name: str
    run_id: str
    arch: str                              # "mask2former", "yolo", "maskrcnn"
    weights_path: str
    best_map50: float
    best_epoch: Any
    total_epochs: int
    taxonomy: str                          # "23_class" or "32_class"
    raw_stats: Dict[str, Any]
    checkpoint_hash: str = ""
    num_classes: int = 32
    is_compatible: bool = True
    compatibility_note: str = ""


class TeacherScanner:
    """Scans runs_comparison and provides both interactive and programmatic selection of teachers."""

    def __init__(self, runs_comparison_dir: str = "/home/orchvate/CAR_TRAIN1/runs_comparison"):
        self.runs_comparison_dir = Path(runs_comparison_dir)
        self._hash_cache: Dict[str, str] = {}

    def _get_checkpoint_hash(self, path: Path | str) -> str:
        p_str = str(path)
        if p_str not in self._hash_cache:
            try:
                self._hash_cache[p_str] = compute_path_sha256(path)
            except Exception:
                self._hash_cache[p_str] = "hash_unavailable"
        return self._hash_cache[p_str]

    def _read_csv_best_map(self, csv_path: Path) -> Tuple[float, Any, int]:
        if not csv_path.exists():
            return 0.0, "?", 0
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
            if not reader:
                return 0.0, "?", 0
            best_map = -1.0
            best_ep = 1
            for row in reader:
                for k in ["best_mask_map50", "map50_mask", "mAP50", "metrics/mAP50(M)", "val/mAP50"]:
                    if k in row and row[k]:
                        try:
                            v = float(row[k])
                            if v > best_map:
                                best_map = v
                                best_ep = row.get("epoch", 1)
                        except (ValueError, TypeError):
                            pass
            return max(best_map, 0.0), best_ep, len(reader)
        except Exception:
            return 0.0, "?", 0

    def _find_best_metrics(self, weights_dir: Path) -> Tuple[float, Any, int]:
        """Searches for metrics.csv or best_metrics.json in parent directories."""
        # 1. Try best_metrics.json
        for search_dir in [weights_dir, weights_dir.parent, weights_dir.parent.parent]:
            bm_json = search_dir / "best_metrics.json"
            if bm_json.exists():
                try:
                    with open(bm_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    val_map = (
                        data.get("best_mask_map50")
                        or data.get("best_map50")
                        or data.get("best_val_map50")
                        or data.get("map50_mask")
                        or (data.get("val_metrics", {}).get("map50_mask") if isinstance(data.get("val_metrics"), dict) else 0.0)
                        or 0.0
                    )
                    val_ep = data.get("best_epoch") or data.get("epoch") or "?"
                    return float(val_map), val_ep, data.get("total_epochs", 0)
                except Exception:
                    pass

        # 2. Try metrics.csv or results.csv
        for search_dir in [weights_dir, weights_dir.parent, weights_dir.parent.parent]:
            for name in ["metrics.csv", "results.csv"]:
                csv_p = search_dir / name
                if csv_p.exists():
                    b_map, b_ep, total_ep = self._read_csv_best_map(csv_p)
                    if b_map > 0.0 or total_ep > 0:
                        return b_map, b_ep, total_ep

        return 0.0, "?", 0

    def scan(self, target_taxonomy: str = "32_class") -> List[DiscoveredTeacher]:
        """Scans runs_comparison directory for all available models, deduplicating by run_id & arch."""
        candidates: List[DiscoveredTeacher] = []
        if not self.runs_comparison_dir.exists():
            return candidates

        # 1. Mask2Former models
        m2f_best_models = list(self.runs_comparison_dir.glob("**/mask2former*/weights/best/model.safetensors"))
        if not m2f_best_models:
            m2f_best_models = list(self.runs_comparison_dir.glob("**/mask2former*/best_model/model.safetensors"))

        seen_runs = set()
        for p in sorted(m2f_best_models, key=lambda x: str(x)):
            weights_dir = p.parent
            run_id = "unknown"
            for part in weights_dir.parts:
                if part.startswith("run_"):
                    run_id = part
                    break
            
            run_key = ("mask2former", run_id)
            if run_key in seen_runs:
                continue
            seen_runs.add(run_key)

            # Taxonomy detection
            cfg_path = weights_dir / "config.json"
            taxonomy = "32_class"
            num_classes = 32
            if cfg_path.exists():
                try:
                    with open(cfg_path) as f:
                        cfg = json.load(f)
                    id2label = cfg.get("id2label", {})
                    if len(id2label) == 23:
                        taxonomy = "23_class"
                        num_classes = 23
                    elif len(id2label) > 0:
                        num_classes = len(id2label)
                except Exception:
                    pass

            b_map, b_ep, total_ep = self._find_best_metrics(weights_dir)
            ckpt_hash = self._get_checkpoint_hash(weights_dir)

            candidates.append(DiscoveredTeacher(
                index=0,
                name=f"Mask2Former_{run_id}",
                run_id=run_id,
                arch="mask2former",
                weights_path=str(weights_dir),
                best_map50=b_map,
                best_epoch=b_ep,
                total_epochs=total_ep,
                taxonomy=taxonomy,
                num_classes=num_classes,
                checkpoint_hash=ckpt_hash,
                raw_stats={"config_path": str(cfg_path)}
            ))

        # 2. YOLO models
        yolo_pts = list(self.runs_comparison_dir.glob("**/yolo11*-seg*/weights/best.pt")) + \
                   list(self.runs_comparison_dir.glob("**/yolo11*-seg*/best.pt"))
        
        for pt in sorted(yolo_pts, key=lambda x: str(x)):
            run_id = "unknown"
            for part in pt.parts:
                if part.startswith("run_"):
                    run_id = part
                    break
            if run_id == "unknown":
                run_id = pt.parent.parent.name

            run_key = ("yolo", run_id)
            if run_key in seen_runs:
                continue
            seen_runs.add(run_key)

            b_map, b_ep, total_ep = self._find_best_metrics(pt.parent)
            ckpt_hash = self._get_checkpoint_hash(pt)

            candidates.append(DiscoveredTeacher(
                index=0,
                name=f"YOLO11m_{run_id}",
                run_id=run_id,
                arch="yolo",
                weights_path=str(pt),
                best_map50=b_map,
                best_epoch=b_ep,
                total_epochs=total_ep,
                taxonomy="23_class",
                num_classes=23,
                checkpoint_hash=ckpt_hash,
                raw_stats={"weights_path": str(pt)}
            ))

        # Sort candidates primarily by best_map50 descending
        candidates.sort(key=lambda c: c.best_map50, reverse=True)
        # Re-index 1..N and evaluate compatibility against target dataset taxonomy
        target_tax = target_taxonomy or "32_class"
        for i, c in enumerate(candidates, 1):
            c.index = i
            if c.taxonomy == target_tax:
                c.is_compatible = True
                c.compatibility_note = f"Native {target_tax} match"
            elif target_tax == "32_class" and c.taxonomy == "23_class":
                c.is_compatible = False
                c.compatibility_note = "Requires 23->32 class mapping"
            else:
                c.is_compatible = True
                c.compatibility_note = f"Taxonomy {c.taxonomy}"

        return candidates

    def display_candidates(self, candidates: List[DiscoveredTeacher], target_taxonomy: str = "32_class") -> None:
        """Pretty-prints candidate teachers with rich metadata, hashes, and compatibility."""
        print("\n" + "=" * 105)
        print(" CANDIDATE TEACHER MODELS DISCOVERED IN runs_comparison/")
        print("=" * 105)
        print(f"{'#':<3} | {'Architecture':<12} | {'Run ID':<24} | {'mAP50':<7} | {'Epoch':<6} | {'Classes':<7} | {'SHA-256 (prefix)':<16} | {'Compatibility'}")
        print("-" * 105)
        for c in candidates:
            hash_display = c.checkpoint_hash[:14] + ".." if len(c.checkpoint_hash) > 16 else c.checkpoint_hash
            compat_str = "[OK] Compatible" if c.is_compatible else f"[!] {c.compatibility_note}"
            print(f"[{c.index:<2}] | {c.arch:<12} | {c.run_id:<24} | {c.best_map50:<7.4f} | {str(c.best_epoch):<6} | {c.num_classes} ({c.taxonomy.split('_')[0]}) | {hash_display:<16} | {compat_str}")
            print(f"      Weights: {c.weights_path}")
        print("=" * 105)

    def interactive_select(self, target_taxonomy: str = "32_class") -> List[TeacherConfig]:
        """Interactive CLI prompt allowing user to choose number of teachers and which models."""
        candidates = self.scan(target_taxonomy=target_taxonomy)
        if not candidates:
            raise RuntimeError(f"No candidate teacher models found in {self.runs_comparison_dir}")

        self.display_candidates(candidates, target_taxonomy=target_taxonomy)

        print("\nOptions for selection:")
        print("  - Enter comma-separated indices (e.g. '1,2' for top 2 teachers, or '1' for single teacher)")
        print("  - Enter 'all' to select all compatible teachers")
        print("  - Enter a direct directory or checkpoint path")

        selected_configs: List[TeacherConfig] = []
        while not selected_configs:
            raw_input = input(f"\nSelect teacher(s) [default: 1,2]: ").strip()
            if not raw_input:
                raw_input = "1,2"

            if raw_input.lower() == "all":
                chosen_list = [c for c in candidates if c.is_compatible]
                if not chosen_list:
                    chosen_list = candidates
                for i, c in enumerate(chosen_list, 1):
                    selected_configs.append(TeacherConfig(
                        name=f"Teacher_{chr(64 + i)}",
                        arch=c.arch,
                        checkpoint_path=c.weights_path,
                        run_id=c.run_id,
                        class_map=c.taxonomy
                    ))
                break

            # Check if direct path
            if Path(raw_input).exists():
                p = Path(raw_input)
                arch = "mask2former" if (p / "model.safetensors").exists() or p.is_dir() else "yolo"
                selected_configs.append(TeacherConfig(
                    name="Teacher_A",
                    arch=arch,
                    checkpoint_path=str(p.resolve()),
                    run_id=p.name,
                    class_map=target_taxonomy
                ))
                break

            # Comma-separated indices
            tokens = [t.strip() for t in raw_input.replace(" ", ",").split(",") if t.strip()]
            valid = True
            temp_configs = []
            for i, tok in enumerate(tokens, 1):
                if not tok.isdigit():
                    print(f"[!] Unrecognized selection token: {tok}")
                    valid = False
                    break
                sel_idx = int(tok)
                chosen = next((c for c in candidates if c.index == sel_idx), None)
                if not chosen:
                    print(f"[!] Index {sel_idx} not found in candidate list.")
                    valid = False
                    break
                if not chosen.is_compatible:
                    print(f"[!] Warning: Teacher #{sel_idx} ({chosen.name}) has note: {chosen.compatibility_note}")
                label = f"Teacher_{chr(64 + i)}"
                temp_configs.append(TeacherConfig(
                    name=label,
                    arch=chosen.arch,
                    checkpoint_path=chosen.weights_path,
                    run_id=chosen.run_id,
                    class_map=chosen.taxonomy
                ))
            if valid and temp_configs:
                selected_configs = temp_configs

        print("\nSelected Teacher Team:")
        for t in selected_configs:
            print(f"  - {t.name}: {t.arch} ({t.checkpoint_path}) [Run: {t.run_id}]")

        return selected_configs

    def select_by_spec(self, specs: List[str]) -> List[TeacherConfig]:
        """
        Programmatic selection from string identifiers (index, run_id, or direct weight path).
        Supports:
          specs = ["1", "2"] (indices)
          specs = ["run_20261031_150325", "run_20260909_134656"] (run_ids)
          specs = ["/path/to/best.pt", "/path/to/m2f/best"] (direct paths)
        """
        candidates = self.scan()
        selected: List[TeacherConfig] = []

        for i, s in enumerate(specs, 1):
            label = f"Teacher_{chr(64 + i)}"
            matched_candidate = None

            # Try by index
            if s.isdigit():
                idx = int(s)
                matched_candidate = next((c for c in candidates if c.index == idx), None)

            # Try by run_id
            if matched_candidate is None:
                matched_candidate = next((c for c in candidates if s in c.run_id or s in c.name), None)

            # Try by path
            if matched_candidate is None and Path(s).exists():
                p = Path(s)
                arch = "mask2former" if (p / "model.safetensors").exists() or p.is_dir() else "yolo"
                matched_candidate = DiscoveredTeacher(
                    index=999,
                    name=f"Custom_{label}",
                    run_id="custom",
                    arch=arch,
                    weights_path=str(p.resolve()),
                    best_map50=0.0,
                    best_epoch=1,
                    total_epochs=1,
                    taxonomy="23_class",
                    raw_stats={}
                )

            if matched_candidate is None:
                raise ValueError(f"Teacher specification '{s}' could not be resolved from runs_comparison or disk.")

            selected.append(TeacherConfig(
                name=label,
                arch=matched_candidate.arch,
                checkpoint_path=matched_candidate.weights_path,
                run_id=matched_candidate.run_id,
                class_map=matched_candidate.taxonomy
            ))

        return selected
