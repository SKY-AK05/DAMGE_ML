"""
Configuration system for Adaptive Multi-Teacher Distillation Pipeline.
Provides structured dataclasses and YAML serialization with schema validation.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml
import hashlib
import json


@dataclass
class TeacherConfig:
    name: str                                # e.g. "Teacher_A"
    arch: str                                # "yolo", "mask2former", "maskrcnn"
    checkpoint_path: str                     # Path to .pt or safetensors directory
    conf_floor: float = 0.05                 # Low conf floor for full calibration curve
    nms_iou: float = 0.70                    # Permissive NMS at cache generation time
    tta: bool = False                        # Test-Time Augmentation
    class_map: Optional[str] = None          # Optional taxonomy override ("23_class", "32_class")
    run_id: Optional[str] = None             # Originating run_id from runs_comparison


@dataclass
class DatasetConfig:
    name: str = "RAW_DATASET2"
    root: str = "/home/orchvate/CAR_TRAIN1/RAW_DATASET2"
    train_images: str = "/home/orchvate/CAR_TRAIN1/RAW_DATASET2/images"
    val_images: str = "/home/orchvate/CAR_TRAIN1/RAW_DATASET2/images"
    test_images: str = "/home/orchvate/CAR_TRAIN1/EVA/GROUND/images"
    train_json: str = "/home/orchvate/CAR_TRAIN1/RAW_DATASET2/annotations/train_5212.json"
    calib_json: str = "/home/orchvate/CAR_TRAIN1/RAW_DATASET2/annotations/calib_652.json"
    val_json: str = "/home/orchvate/CAR_TRAIN1/RAW_DATASET2/annotations/val_652.json"
    test_json: str = "/home/orchvate/CAR_TRAIN1/EVA/GROUND/coco_annotations.json"
    canonical_taxonomy: str = "32_class"     # Primary evaluation & student class space (32 classes)
    imgsz: int = 640


@dataclass
class MatchingConfig:
    tau_match: float = 0.50                  # Hungarian matching IoU threshold
    tau_low: float = 0.10                    # Poor localization threshold (below this is FP_loc)
    metric: str = "mask_iou"                 # "mask_iou" or "box_iou"
    class_agnostic: bool = True              # Class-agnostic matching to detect class errors


@dataclass
class ReliabilityConfig:
    shrinkage_m: float = 20.0                # Bayesian pseudo-count for rare class smoothing
    n_min: int = 10                          # Minimum instances to trust per-class estimate
    metric_type: str = "recall_aware_iou"    # r[k,c] = recall * mean_tp_iou
    bootstrap_iterations: int = 500          # Image-level bootstrap for confidence intervals
    conf_calibration: str = "isotonic"       # "isotonic", "platt", or "none"


@dataclass
class TrustWeightingConfig:
    mode: str = "adaptive"                   # "adaptive", "fixed", "shuffled", "inverse", "uniform_in_gate"
    gate_type: str = "smooth"                # "smooth" or "hard"
    gate_margin: float = 0.02                # Margin above r_ref required to activate gate
    g_min: float = 0.0                       # Minimum gate floor
    share_type: str = "softmax"              # "softmax" or "proportional"
    temperature: float = 0.50                # Softmax temperature (T -> 0: winner-take-all, T -> inf: uniform)
    s_min: float = 0.10                      # Floor to preserve teacher diversity when both are usable
    instance_enabled: bool = True            # Apply per-instance omega (GT veto & quality)
    omega_min: float = 0.10                  # Minimum weight for a TP instance match
    output_type_split: bool = False          # Separate weights for cls, box, and mask


@dataclass
class FusionConfig:
    tau_cluster: float = 0.60                # Mask IoU threshold for multi-teacher clustering
    solo_penalty: float = 0.80               # Penalty kappa applied when only 1 teacher sees object
    disagreement_ratio: float = 2.0          # Margin ratio required when teachers disagree without GT
    box_fusion_mode: str = "wbf"             # "wbf" (Weighted Box Fusion) or "highest_conf"


@dataclass
class CacheConfig:
    dir: str = "/home/orchvate/CAR_TRAIN1/teacher_cache"
    tier: int = 2                            # Tier 1 (instances), Tier 2 (soft targets), Tier 3 (dense logits)
    shard_size: int = 500                    # Images per shard file
    storage_format: str = "npz"              # "npz" or "pt"
    require_key: Optional[str] = None        # Expected cache hash for strict reproducibility
    mask_scale: float = 0.25                 # Mask resolution scale in Tier 2 (1/4 canonical)


@dataclass
class StudentConfig:
    arch: str = "mask2former"                # "mask2former" (Swin-Tiny, 47.4M params) — ONLY SUPPORTED VALUE
    init: str = "scratch"                    # "scratch": Mask2FormerConfig-only init, NO from_pretrained() on model
    imgsz: int = 640
    epochs: int = 50
    batch_size: int = 4
    workers: int = 4
    lr0: float = 0.0001                      # AdamW head LR; backbone gets 0.1× (1e-5)
    weight_decay: float = 0.0001
    momentum: float = 0.9
    optimizer: str = "adamw"                 # "adamw" (recommended) or "sgd"
    amp: bool = True


@dataclass
class DistillationConfig:
    enabled: bool = True
    lambda_schedule: str = "warmup"          # "constant", "warmup", "decay"
    lambda_max: float = 1.0                  # Distillation loss multiplier
    warmup_epochs: int = 10
    temperature: float = 2.0                 # KD soft-target temperature
    loss_reduction: str = "weighted_mean"    # "weighted_mean" or "sum"
    ignore_vetoed: bool = True               # Exclude vetoed instances from KD loss
    ignore_absence: bool = True              # Teacher absence is never background
    quarantine_unmatched: bool = True        # FP_unmatched instances quarantined


@dataclass
class AugmentationConfig:
    level: str = "L0"                        # "L0" (none), "L1" (photometric), "L2" (geometric), "L3" (mosaic/mixup)
    hflip_prob: float = 0.5
    scale_range: tuple = (0.8, 1.2)


@dataclass
class EvaluationConfig:
    conf: float = 0.001                      # COCO evaluation confidence threshold
    nms_iou: float = 0.70                    # Evaluation NMS threshold
    primary_metric: str = "mask_mAP50-95"    # Pre-registered primary evaluation metric
    seed: int = 42
    num_seeds: int = 3                       # Seeds for statistical confidence testing


@dataclass
class DistillationPipelineConfig:
    experiment_name: str = "distillation_exp"
    output_dir: str = "/home/orchvate/CAR_TRAIN1/runs/distillation"
    teachers: List[TeacherConfig] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    weights: TrustWeightingConfig = field(default_factory=TrustWeightingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    distill: DistillationConfig = field(default_factory=DistillationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DistillationPipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Convert nested dictionaries back to dataclass objects
        teachers = [TeacherConfig(**t) for t in data.get("teachers", [])]
        dataset = DatasetConfig(**data.get("dataset", {}))
        matching = MatchingConfig(**data.get("matching", {}))
        reliability = ReliabilityConfig(**data.get("reliability", {}))
        weights = TrustWeightingConfig(**data.get("weights", {}))
        fusion = FusionConfig(**data.get("fusion", {}))
        cache = CacheConfig(**data.get("cache", {}))
        student = StudentConfig(**data.get("student", {}))
        distill = DistillationConfig(**data.get("distill", {}))
        augmentation = AugmentationConfig(**data.get("augmentation", {}))
        evaluation = EvaluationConfig(**data.get("evaluation", {}))

        return cls(
            experiment_name=data.get("experiment_name", "distillation_exp"),
            output_dir=data.get("output_dir", "/home/orchvate/CAR_TRAIN1/runs/distillation"),
            teachers=teachers,
            dataset=dataset,
            matching=matching,
            reliability=reliability,
            weights=weights,
            fusion=fusion,
            cache=cache,
            student=student,
            distill=distill,
            augmentation=augmentation,
            evaluation=evaluation
        )

    def compute_config_hash(self) -> str:
        """Computes deterministic SHA-256 hash of entire configuration state."""
        dict_repr = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(dict_repr.encode("utf-8")).hexdigest()

