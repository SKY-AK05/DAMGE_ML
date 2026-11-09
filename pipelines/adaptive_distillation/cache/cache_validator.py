"""
Cache Validator module.
Executes hard gate validation checks on persistent teacher prediction caches
before student training can begin.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from pipelines.adaptive_distillation.config import DistillationPipelineConfig
from pipelines.adaptive_distillation.cache.cache_manifest import CacheManifestManager
from pipelines.adaptive_distillation.cache.cache_reader import CacheReader


class CacheValidator:
    """Verifies cache completeness, schema integrity, and absence of numerical corruption."""

    def __init__(self, cache_dir: Path | str, cfg: DistillationPipelineConfig):
        self.cache_dir = Path(cache_dir)
        self.cfg = cfg

    def validate_cache(
        self, expected_image_ids: List[str]
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Runs comprehensive validation suite.
        Returns:
            - is_valid: bool
            - errors: list of error descriptions
            - summary: dict of summary metrics (instance counts, empty counts, memory)
        """
        errors: List[str] = []
        summary: Dict[str, Any] = {"teachers": {}}

        # 1. Validate manifest against current config
        is_manifest_valid, manifest_errs = CacheManifestManager.validate_manifest_against_config(
            self.cache_dir, self.cfg
        )
        if not is_manifest_valid:
            errors.extend(manifest_errs)

        # 2. Check each teacher cache
        expected_set = set(expected_image_ids)

        for teacher_cfg in self.cfg.teachers:
            t_name = teacher_cfg.name
            t_summary = {
                "total_images": 0,
                "empty_images": 0,
                "total_instances": 0,
                "nan_box_errors": 0,
                "out_of_bounds_errors": 0
            }

            try:
                reader = CacheReader(self.cache_dir, t_name)
            except Exception as e:
                errors.append(f"Cannot initialize CacheReader for teacher '{t_name}': {e}")
                continue

            cached_ids = set(reader.index_map.keys())

            # Check for missing images (Never allow missing images to be silently ignored)
            missing_ids = expected_set - cached_ids
            if missing_ids:
                sample_missing = list(missing_ids)[:5]
                errors.append(
                    f"Teacher '{t_name}' cache is incomplete: {len(missing_ids)} expected images are missing. "
                    f"Sample: {sample_missing}"
                )

            # Sample spot-check (check first 200 images for numerical validity)
            check_sample_ids = expected_image_ids[:200]
            for iid in check_sample_ids:
                if iid not in cached_ids:
                    continue
                record = reader.read_image(iid)
                t_summary["total_images"] += 1
                if record.is_empty:
                    t_summary["empty_images"] += 1
                    continue

                for inst in record.instances:
                    t_summary["total_instances"] += 1
                    b = inst.box
                    # NaN / Inf checks
                    if any(np.isnan(x) or np.isinf(x) for x in b):
                        errors.append(f"NaN or Inf detected in bounding box for image '{iid}' in teacher '{t_name}'.")
                        t_summary["nan_box_errors"] += 1

                    # Bounds check (allow 10% slack for slight boundary annotations)
                    x1, y1, x2, y2 = b
                    if x1 < -10 or y1 < -10 or x2 > record.width + 10 or y2 > record.height + 10:
                        t_summary["out_of_bounds_errors"] += 1

                    # Mask dimension check
                    rle_size = inst.mask_rle.get("size", [])
                    if len(rle_size) == 2:
                        if rle_size[0] != record.height or rle_size[1] != record.width:
                            errors.append(
                                f"Mask size mismatch in image '{iid}': rle={rle_size} vs image=({record.height}, {record.width})"
                            )

            summary["teachers"][t_name] = t_summary

        is_valid = (len(errors) == 0)
        return is_valid, errors, summary

