"""
Cache Builder module.
Writes persistent, sharded, indexed disk caches of teacher predictions and metadata.
Guarantees distinct empty-detection records and atomic shard generation.
"""

import os
import json
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from pipelines.adaptive_distillation.teachers.teacher_inference import ImageTeacherOutput, TeacherInstance
from pipelines.adaptive_distillation.matching.instance_matcher import MatchResult
from pipelines.adaptive_distillation.matching.trust_weighting import InstanceWeightResult


class CacheBuilder:
    """Builds sharded .npz prediction archives and JSON index mappings for a given teacher."""

    def __init__(self, cache_dir: Path | str, teacher_name: str, shard_size: int = 500):
        self.cache_dir = Path(cache_dir)
        self.teacher_name = teacher_name
        self.shard_size = shard_size

        self.teacher_dir = self.cache_dir / teacher_name
        self.teacher_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / f"index_{teacher_name}.json"

    def write_records(
        self,
        records: List[Tuple[ImageTeacherOutput, Optional[List[MatchResult]], Optional[List[InstanceWeightResult]]]]
    ) -> Dict[str, Any]:
        """
        Writes a collection of image records into numbered shards.
        records: list of tuples: (output, optional_matches, optional_weights)
        Returns the constructed index dictionary.
        """
        index_map: Dict[str, Dict[str, Any]] = {}
        total_images = len(records)
        num_shards = (total_images + self.shard_size - 1) // max(1, self.shard_size)

        for s_idx in range(num_shards):
            start = s_idx * self.shard_size
            end = min(total_images, start + self.shard_size)
            shard_records = records[start:end]

            shard_name = f"shard_{s_idx:05d}.npz"
            shard_path = self.teacher_dir / shard_name

            # Pack shard data
            image_ids = []
            widths = []
            heights = []
            is_empty_list = []
            instance_counts = []

            # Flattened instance arrays
            boxes_all = []
            classes_all = []
            confs_all = []
            rle_sizes_all = []
            rle_counts_all = []
            downscaled_masks = []
            omegas_all = []
            statuses_all = []

            for offset_in_shard, item in enumerate(shard_records):
                out = item[0]
                matches = item[1]
                weights = item[2]

                image_ids.append(out.image_id)
                widths.append(out.width)
                heights.append(out.height)
                is_empty_list.append(int(out.is_empty))
                instance_counts.append(len(out.instances))

                index_map[out.image_id] = {
                    "shard": shard_name,
                    "offset": offset_in_shard,
                    "n_instances": len(out.instances),
                    "is_empty": out.is_empty
                }

                for inst_idx, inst in enumerate(out.instances):
                    boxes_all.append(inst.box)
                    classes_all.append(inst.canonical_class_id if inst.canonical_class_id is not None else -1)
                    confs_all.append(inst.confidence)
                    
                    # RLE encoding
                    rle_size = inst.mask_rle.get("size", [out.height, out.width])
                    rle_counts = inst.mask_rle.get("counts", "")
                    rle_sizes_all.append(rle_size)
                    rle_counts_all.append(rle_counts)

                    # Downscaled mask
                    if inst.mask_prob_downscaled is not None:
                        downscaled_masks.append(inst.mask_prob_downscaled)
                    else:
                        downscaled_masks.append(np.zeros((1, 1), dtype=np.uint8))

                    # Weight and status
                    omega_val = 1.0
                    status_val = "unmatched"
                    if weights and inst_idx < len(weights):
                        w_item = weights[inst_idx]
                        omega_val = getattr(w_item, "final_weight_w", getattr(w_item, "instance_omega", 1.0))
                    if matches and inst_idx < len(matches):
                        status_val = matches[inst_idx].status
                        # Strict GT veto safeguard
                        if status_val in ("FP_class", "FP_loc", "FP_unmatched"):
                            omega_val = 0.0

                    omegas_all.append(omega_val)
                    statuses_all.append(status_val)

            # Convert to numpy arrays
            save_dict = {
                "image_ids": np.array(image_ids, dtype=object),
                "widths": np.array(widths, dtype=np.int32),
                "heights": np.array(heights, dtype=np.int32),
                "is_empty": np.array(is_empty_list, dtype=np.int8),
                "instance_counts": np.array(instance_counts, dtype=np.int32),
                "boxes": np.array(boxes_all, dtype=np.float32) if boxes_all else np.zeros((0, 4), dtype=np.float32),
                "canonical_classes": np.array(classes_all, dtype=np.int32) if classes_all else np.zeros((0,), dtype=np.int32),
                "confidences": np.array(confs_all, dtype=np.float32) if confs_all else np.zeros((0,), dtype=np.float32),
                "omegas": np.array(omegas_all, dtype=np.float32) if omegas_all else np.zeros((0,), dtype=np.float32),
                "statuses": np.array(statuses_all, dtype=object) if statuses_all else np.zeros((0,), dtype=object),
                "rle_sizes": np.array(rle_sizes_all, dtype=object) if rle_sizes_all else np.zeros((0, 2), dtype=object),
                "rle_counts": np.array(rle_counts_all, dtype=object) if rle_counts_all else np.zeros((0,), dtype=object),
                "downscaled_masks": np.array(downscaled_masks, dtype=object) if downscaled_masks else np.zeros((0,), dtype=object)
            }

            # Atomic write
            temp_path = self.teacher_dir / f"{shard_name}.tmp.npz"
            np.savez_compressed(temp_path, **save_dict)
            temp_path.replace(shard_path)

        # Write index file atomically
        temp_index = self.index_path.with_suffix(".tmp")
        with open(temp_index, "w", encoding="utf-8") as f:
            json.dump(index_map, f, indent=2)
        temp_index.replace(self.index_path)

        return index_map

    def append_records(
        self,
        records: List[Tuple[ImageTeacherOutput, Optional[List[MatchResult]], Optional[List[InstanceWeightResult]]]]
    ) -> Dict[str, Any]:
        """
        Incrementally appends records into new numbered shards and updates the index map.
        Guarantees that existing valid shards remain completely intact.
        """
        if not records:
            if self.index_path.exists():
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

        # 1. Load existing index
        index_map: Dict[str, Dict[str, Any]] = {}
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    index_map = json.load(f)
            except Exception:
                index_map = {}

        # 2. Determine next available shard index
        existing_shards = sorted(self.teacher_dir.glob("shard_*.npz"))
        start_shard_idx = 0
        if existing_shards:
            max_idx = -1
            for s_path in existing_shards:
                stem = s_path.stem
                if stem.startswith("shard_"):
                    try:
                        idx_num = int(stem.split("_")[1])
                        if idx_num > max_idx:
                            max_idx = idx_num
                    except (ValueError, IndexError):
                        pass
            start_shard_idx = max_idx + 1

        total_images = len(records)
        num_shards = (total_images + self.shard_size - 1) // max(1, self.shard_size)

        for s_idx in range(num_shards):
            curr_shard_idx = start_shard_idx + s_idx
            start = s_idx * self.shard_size
            end = min(total_images, start + self.shard_size)
            shard_records = records[start:end]

            shard_name = f"shard_{curr_shard_idx:05d}.npz"
            shard_path = self.teacher_dir / shard_name

            image_ids = []
            widths = []
            heights = []
            is_empty_list = []
            instance_counts = []

            boxes_all = []
            classes_all = []
            confs_all = []
            rle_sizes_all = []
            rle_counts_all = []
            downscaled_masks = []
            omegas_all = []
            statuses_all = []

            for offset_in_shard, item in enumerate(shard_records):
                out = item[0]
                matches = item[1]
                weights = item[2]

                image_ids.append(out.image_id)
                widths.append(out.width)
                heights.append(out.height)
                is_empty_list.append(int(out.is_empty))
                instance_counts.append(len(out.instances))

                # Update index map (atomic update for this image)
                entry = {
                    "shard": shard_name,
                    "offset": offset_in_shard,
                    "n_instances": len(out.instances),
                    "is_empty": out.is_empty
                }
                if hasattr(out, "ann_hash") and out.ann_hash:
                    entry["ann_hash"] = out.ann_hash
                if hasattr(out, "img_hash") and out.img_hash:
                    entry["img_hash"] = out.img_hash

                index_map[out.image_id] = entry

                for inst_idx, inst in enumerate(out.instances):
                    boxes_all.append(inst.box)
                    classes_all.append(inst.canonical_class_id if inst.canonical_class_id is not None else -1)
                    confs_all.append(inst.confidence)

                    rle_size = inst.mask_rle.get("size", [out.height, out.width])
                    rle_counts = inst.mask_rle.get("counts", "")
                    rle_sizes_all.append(rle_size)
                    rle_counts_all.append(rle_counts)

                    if inst.mask_prob_downscaled is not None:
                        downscaled_masks.append(inst.mask_prob_downscaled)
                    else:
                        downscaled_masks.append(np.zeros((1, 1), dtype=np.uint8))

                    omega_val = 1.0
                    status_val = "unmatched"
                    if weights and inst_idx < len(weights):
                        w_item = weights[inst_idx]
                        omega_val = getattr(w_item, "final_weight_w", getattr(w_item, "instance_omega", 1.0))
                    if matches and inst_idx < len(matches):
                        status_val = matches[inst_idx].status
                        if status_val in ("FP_class", "FP_loc", "FP_unmatched"):
                            omega_val = 0.0

                    omegas_all.append(omega_val)
                    statuses_all.append(status_val)

            save_dict = {
                "image_ids": np.array(image_ids, dtype=object),
                "widths": np.array(widths, dtype=np.int32),
                "heights": np.array(heights, dtype=np.int32),
                "is_empty": np.array(is_empty_list, dtype=np.int8),
                "instance_counts": np.array(instance_counts, dtype=np.int32),
                "boxes": np.array(boxes_all, dtype=np.float32) if boxes_all else np.zeros((0, 4), dtype=np.float32),
                "canonical_classes": np.array(classes_all, dtype=np.int32) if classes_all else np.zeros((0,), dtype=np.int32),
                "confidences": np.array(confs_all, dtype=np.float32) if confs_all else np.zeros((0,), dtype=np.float32),
                "omegas": np.array(omegas_all, dtype=np.float32) if omegas_all else np.zeros((0,), dtype=np.float32),
                "statuses": np.array(statuses_all, dtype=object) if statuses_all else np.zeros((0,), dtype=object),
                "rle_sizes": np.array(rle_sizes_all, dtype=object) if rle_sizes_all else np.zeros((0, 2), dtype=object),
                "rle_counts": np.array(rle_counts_all, dtype=object) if rle_counts_all else np.zeros((0,), dtype=object),
                "downscaled_masks": np.array(downscaled_masks, dtype=object) if downscaled_masks else np.zeros((0,), dtype=object)
            }

            temp_path = self.teacher_dir / f"{shard_name}.tmp.npz"
            np.savez_compressed(temp_path, **save_dict)
            temp_path.replace(shard_path)

        # Write index file atomically
        temp_index = self.index_path.with_suffix(".tmp")
        with open(temp_index, "w", encoding="utf-8") as f:
            json.dump(index_map, f, indent=2)
        temp_index.replace(self.index_path)

        return index_map
