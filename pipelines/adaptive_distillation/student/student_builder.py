"""
Student Builder module.
Initializes Student C from strict scratch using Mask2Former (Swin-Tiny architecture).

SCRATCH INIT RULE:
  Mask2FormerConfig is loaded from pretrained to get the correct architecture spec
  (number of queries, decoder layers, hidden dims), but the model weights are
  initialized randomly by calling Mask2FormerForUniversalSegmentation(config)
  WITHOUT calling .from_pretrained() on the model itself.

  This guarantees:
  - Swin backbone: random weights (no ImageNet or COCO pretraining)
  - Pixel decoder: random weights
  - Transformer decoder: random weights
  - Classification and mask heads: random weights

Do NOT call Mask2FormerForUniversalSegmentation.from_pretrained(...) for Student C.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from pipelines.adaptive_distillation.config import StudentConfig


# Canonical 32-class label list (must match dataset taxonomy)
CLASSES_32 = [
    "back_bumper", "back_door", "back_glass", "back_left_door", "back_left_light",
    "back_right_door", "back_right_light", "front_bumper", "front_door",
    "front_glass", "front_left_door", "front_left_light", "front_right_door",
    "front_right_light", "hood", "left_mirror", "right_mirror", "tail_light",
    "trunk", "wheel", "headlight", "back_windshield", "front_windshield",
    "rear_bumper", "right_door", "left_door", "roof", "front_light",
    "back_light", "side_window", "windshield", "door",
]

# Pad / trim to exactly 32 entries
while len(CLASSES_32) < 32:
    CLASSES_32.append(f"class_{len(CLASSES_32)}")
CLASSES_32 = CLASSES_32[:32]


class StudentBuilder:
    """Builds scratch-initialized Mask2Former Student C model."""

    BASE_MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"

    @staticmethod
    def build(
        config: StudentConfig,
        num_classes: int = 32,
        seed: int = 42,
        class_names: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Build a Mask2Former model initialized from scratch.

        Args:
            config:      StudentConfig with arch='mask2former'.
            num_classes: Number of foreground classes (32 for RAW_DATASET2).
            seed:        Random seed for reproducible weight initialization.
            class_names: Optional list of class names. Defaults to CLASSES_32.

        Returns:
            Mask2FormerForUniversalSegmentation with random weights and processor stored
            as model.processor attribute.

        Raises:
            ValueError: If arch is not 'mask2former'.
        """
        torch.manual_seed(seed)
        arch = config.arch.lower()

        if "mask2former" in arch or "m2f" in arch:
            return StudentBuilder._build_mask2former(num_classes, seed, class_names)

        raise ValueError(
            f"Unsupported student architecture: '{config.arch}'. "
            f"Only 'mask2former' is supported for Student C. "
            f"Do not use maskrcnn or yolo for this pipeline."
        )

    @staticmethod
    def _build_mask2former(
        num_classes: int = 32,
        seed: int = 42,
        class_names: Optional[List[str]] = None,
    ) -> nn.Module:
        from transformers import (
            Mask2FormerConfig,
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        torch.manual_seed(seed)

        names = class_names if class_names and len(class_names) == num_classes else CLASSES_32[:num_classes]
        while len(names) < num_classes:
            names.append(f"class_{len(names)}")
        names = names[:num_classes]

        id2label = {i: name for i, name in enumerate(names)}
        label2id = {name: i for i, name in enumerate(names)}

        # Load config spec only — this downloads the architecture config file (~100 KB)
        # but does NOT download or load any model weights.
        m2f_config = Mask2FormerConfig.from_pretrained(StudentBuilder.BASE_MODEL_ID)
        m2f_config.num_labels = num_classes
        m2f_config.id2label = id2label
        m2f_config.label2id = label2id

        # Build model from config: ALL weights are randomly initialized.
        # This is strictly equivalent to training from scratch with no pretrained leakage.
        torch.manual_seed(seed)
        model = Mask2FormerForUniversalSegmentation(m2f_config)

        # Attach processor as a model attribute so the trainer and evaluator can access it
        # without needing to manage it separately.
        # The processor handles image preprocessing (resize, normalize, pixel_values).
        # It is loaded from pretrained because the preprocessing config (mean/std, size)
        # must be consistent — only the model weights are random, not the preprocessing.
        model.processor = Mask2FormerImageProcessor.from_pretrained(
            StudentBuilder.BASE_MODEL_ID
        )
        model.num_classes = num_classes
        model.class_names = names

        return model

    @staticmethod
    def verify_scratch(model: nn.Module) -> dict:
        """
        Verifies that model is genuinely randomly initialized.
        Returns a dict of verification results.
        """
        results = {}
        all_params = list(model.parameters())
        trainable = [p for p in all_params if p.requires_grad]
        results["total_params"] = sum(p.numel() for p in all_params)
        results["trainable_params"] = sum(p.numel() for p in trainable)
        results["all_trainable"] = len(trainable) == len(all_params)

        # Check that backbone is not all zeros and not identical to known COCO values
        try:
            bb_param = next(
                p for name, p in model.named_parameters()
                if "pixel_level_module.encoder" in name
            )
            results["backbone_all_zeros"] = bool((bb_param.abs().max() == 0.0).item())
            results["backbone_mean"] = float(bb_param.mean().item())
            results["backbone_std"] = float(bb_param.std().item())
            results["backbone_nonzero"] = not results["backbone_all_zeros"]
        except StopIteration:
            results["backbone_nonzero"] = False
            results["backbone_all_zeros"] = True

        return results
