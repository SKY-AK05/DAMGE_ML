"""
Cache Reader module.
Performs streaming, random-access, batch-level retrieval of cached teacher predictions.
Loads only the current batch's records to prevent RAM exhaustion.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class CachedInstanceRecord:
    canonical_class_id: int
    confidence: float
    box: List[float]                       # [x1, y1, x2, y2]
    mask_rle: Dict[str, Any]
    mask_prob_downscaled: Optional[np.ndarray]
    omega: float
    status: str


@dataclass
class CachedImageRecord:
    image_id: str
    width: int
    height: int
    is_empty: bool
    instances: List[CachedInstanceRecord]


class CacheReader:
    """Reads sharded teacher caches by image ID with on-demand shard caching."""

    def __init__(self, cache_dir: Path | str, teacher_name: str, max_loaded_shards: int = 4):
        self.cache_dir = Path(cache_dir)
        self.teacher_name = teacher_name
        self.teacher_dir = self.cache_dir / teacher_name
        self.index_path = self.cache_dir / f"index_{teacher_name}.json"
        self.max_loaded_shards = max_loaded_shards

        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing cache index for teacher '{teacher_name}' at: {self.index_path}")

        with open(self.index_path, "r", encoding="utf-8") as f:
            self.index_map: Dict[str, Dict[str, Any]] = json.load(f)

        # Simple LRU shard cache in worker process
        self._shard_cache: Dict[str, Dict[str, Any]] = {}
        self._shard_lru_keys: List[str] = []

    def _load_shard(self, shard_name: str) -> Dict[str, Any]:
        if shard_name in self._shard_cache:
            # Move to end of LRU
            self._shard_lru_keys.remove(shard_name)
            self._shard_lru_keys.append(shard_name)
            return self._shard_cache[shard_name]

        shard_path = self.teacher_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file: {shard_path}")

        with np.load(shard_path, allow_pickle=True) as data:
            loaded_data = {k: data[k] for k in data.files}

        # Evict oldest if exceeding capacity
        if len(self._shard_cache) >= self.max_loaded_shards:
            oldest = self._shard_lru_keys.pop(0)
            del self._shard_cache[oldest]

        self._shard_cache[shard_name] = loaded_data
        self._shard_lru_keys.append(shard_name)
        return loaded_data

    def read_image(self, image_id: str) -> CachedImageRecord:
        """Reads prediction record for a single image ID."""
        if image_id not in self.index_map:
            raise KeyError(
                f"Image ID '{image_id}' not found in cache index for teacher '{self.teacher_name}'. "
                f"Verify cache alignment and completeness."
            )

        meta = self.index_map[image_id]
        shard_name = meta["shard"]
        offset = meta["offset"]
        n_instances = meta["n_instances"]
        is_empty = bool(meta.get("is_empty", False))

        shard_data = self._load_shard(shard_name)

        w = int(shard_data["widths"][offset])
        h = int(shard_data["heights"][offset])

        if is_empty or n_instances == 0:
            return CachedImageRecord(
                image_id=image_id,
                width=w,
                height=h,
                is_empty=True,
                instances=[]
            )

        # Calculate prefix sum index for instances in shard
        prefix_counts = np.cumsum(shard_data["instance_counts"])
        inst_start = 0 if offset == 0 else prefix_counts[offset - 1]
        inst_end = inst_start + n_instances

        boxes = shard_data["boxes"][inst_start:inst_end]
        classes = shard_data["canonical_classes"][inst_start:inst_end]
        confs = shard_data["confidences"][inst_start:inst_end]
        omegas = shard_data["omegas"][inst_start:inst_end]
        statuses = shard_data["statuses"][inst_start:inst_end]
        rle_sizes = shard_data["rle_sizes"][inst_start:inst_end]
        rle_counts = shard_data["rle_counts"][inst_start:inst_end]
        downscaled = shard_data["downscaled_masks"][inst_start:inst_end]

        instances = []
        for i in range(n_instances):
            cid = int(classes[i])
            rle_dict = {
                "size": list(rle_sizes[i]),
                "counts": rle_counts[i]
            }
            down_mask = downscaled[i] if downscaled[i].shape != (1, 1) else None

            instances.append(CachedInstanceRecord(
                canonical_class_id=cid,
                confidence=float(confs[i]),
                box=[float(b) for b in boxes[i]],
                mask_rle=rle_dict,
                mask_prob_downscaled=down_mask,
                omega=float(omegas[i]),
                status=str(statuses[i])
            ))

        return CachedImageRecord(
            image_id=image_id,
            width=w,
            height=h,
            is_empty=False,
            instances=instances
        )

    def read_batch(self, image_ids: List[str]) -> List[CachedImageRecord]:
        """Reads a mini-batch of cached records by image IDs."""
        return [self.read_image(iid) for iid in image_ids]

