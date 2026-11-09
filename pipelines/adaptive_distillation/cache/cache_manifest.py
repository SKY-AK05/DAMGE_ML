"""
Cache Manifest module.
Generates, validates, and verifies cryptographic SHA-256 cache manifests
to prevent stale caches and silent reproducibility bugs.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from pipelines.adaptive_distillation.config import DistillationPipelineConfig
from pipelines.adaptive_distillation.teachers.teacher_loader import compute_path_sha256


class CacheManifestManager:
    """Manages cache_manifest.json creation, serialization, and validation."""

    SCHEMA_VERSION = "2.0"

    @staticmethod
    def generate_manifest(
        cfg: DistillationPipelineConfig,
        n_images: int,
        failed_image_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        failed = failed_image_ids or []
        
        # 1. Compute dataset manifest hash
        dataset_hash = "unknown"
        if Path(cfg.dataset.train_json).exists():
            dataset_hash = compute_path_sha256(cfg.dataset.train_json)

        # 2. Compute teacher checkpoint hashes
        teacher_metadata = {}
        for t in cfg.teachers:
            t_hash = compute_path_sha256(t.checkpoint_path)
            teacher_metadata[t.name] = {
                "checkpoint_path": str(t.checkpoint_path),
                "checkpoint_sha256": t_hash,
                "arch": t.arch,
                "tier": cfg.cache.tier,
                "conf_floor": t.conf_floor,
                "nms_iou": t.nms_iou,
                "tta": t.tta,
                "run_id": t.run_id
            }

        preprocess_meta = {
            "imgsz": cfg.dataset.imgsz,
            "canonical_taxonomy": cfg.dataset.canonical_taxonomy,
            "mask_scale": cfg.cache.mask_scale
        }

        # Deterministic cache key
        key_content = json.dumps({
            "schema": CacheManifestManager.SCHEMA_VERSION,
            "dataset_hash": dataset_hash,
            "teachers": teacher_metadata,
            "preprocess": preprocess_meta
        }, sort_keys=True)
        cache_key = hashlib.sha256(key_content.encode("utf-8")).hexdigest()

        manifest = {
            "schema_version": CacheManifestManager.SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cache_key": cache_key,
            "dataset_manifest_sha256": dataset_hash,
            "canonical_taxonomy": cfg.dataset.canonical_taxonomy,
            "preprocess": preprocess_meta,
            "teachers": teacher_metadata,
            "n_images": n_images,
            "failed_image_ids": failed
        }
        return manifest

    @staticmethod
    def save_manifest(manifest: Dict[str, Any], cache_dir: Path | str) -> Path:
        out_path = Path(cache_dir) / "cache_manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return out_path

    @staticmethod
    def load_manifest(cache_dir: Path | str) -> Dict[str, Any]:
        p = Path(cache_dir) / "cache_manifest.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing cache manifest at: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def validate_manifest_against_config(
        cache_dir: Path | str, cfg: DistillationPipelineConfig
    ) -> Tuple[bool, List[str]]:
        """
        Validates whether stored cache manifest strictly matches current config and checkpoints.
        Returns (is_valid, list_of_errors).
        """
        errors = []
        try:
            manifest = CacheManifestManager.load_manifest(cache_dir)
        except Exception as e:
            return False, [f"Failed to load cache_manifest.json: {e}"]

        if manifest.get("schema_version") != CacheManifestManager.SCHEMA_VERSION:
            errors.append(
                f"Schema version mismatch: manifest={manifest.get('schema_version')} vs required={CacheManifestManager.SCHEMA_VERSION}"
            )

        # Check teacher checkpoints
        manifest_teachers = manifest.get("teachers", {})
        for t in cfg.teachers:
            if t.name not in manifest_teachers:
                errors.append(f"Teacher '{t.name}' missing from cache manifest.")
                continue
            meta = manifest_teachers[t.name]
            current_hash = compute_path_sha256(t.checkpoint_path)
            if current_hash != meta.get("checkpoint_sha256"):
                errors.append(
                    f"Checkpoint hash mismatch for {t.name}: stored={meta.get('checkpoint_sha256')} vs current={current_hash}"
                )

        # Check preprocessing metadata
        prep = manifest.get("preprocess", {})
        if prep.get("imgsz") != cfg.dataset.imgsz:
            errors.append(f"Image size mismatch: manifest={prep.get('imgsz')} vs config={cfg.dataset.imgsz}")
        if prep.get("mask_scale") != cfg.cache.mask_scale:
            errors.append(f"Mask scale mismatch: manifest={prep.get('mask_scale')} vs config={cfg.cache.mask_scale}")
        if manifest.get("canonical_taxonomy") != cfg.dataset.canonical_taxonomy:
            errors.append(
                f"Taxonomy mismatch: manifest={manifest.get('canonical_taxonomy')} vs config={cfg.dataset.canonical_taxonomy}"
            )

        # If user explicitly required a specific cache key
        if cfg.cache.require_key and cfg.cache.require_key != manifest.get("cache_key"):
            errors.append(f"Cache key mismatch: expected {cfg.cache.require_key} but found {manifest.get('cache_key')}")

        return (len(errors) == 0), errors
