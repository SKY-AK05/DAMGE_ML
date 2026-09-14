#!/usr/bin/env python3
"""
train_mask2former.py -- Mask2Former (Swin Transformer) Instance Segmentation Training.

Fine-tunes HuggingFace Mask2Former (facebook/mask2former-swin-tiny-coco-instance)
on the COCO-formatted car-parts dataset.
"""

import os
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path
import numpy as np
import torch

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
from PIL import Image as PILImage
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# pycocotools: imported at module level so DataLoader workers (which use
# multiprocessing 'spawn' on Windows) don't re-import it on every sample.
from pycocotools import mask as mask_utils

from scripts.evaluation.unified_evaluator import UnifiedEvaluator
from scripts.evaluation.unified_logger import UnifiedLogger

try:
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
except ImportError:
    print("[ERROR] HuggingFace transformers is required for Mask2Former training. Install with: pip install transformers")
    sys.exit(1)


class COCOMask2FormerDataset(Dataset):
    def __init__(self, images_dir, json_file, processor):
        self.images_dir = Path(images_dir)
        with open(json_file, "r") as f:
            self.coco_data = json.load(f)
        
        self.processor = processor
        self.images_by_id = {img["id"]: img for img in self.coco_data.get("images", [])}

        self.anns_by_image = {}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.anns_by_image:
                self.anns_by_image[img_id] = []
            self.anns_by_image[img_id].append(ann)

        self.cat_id_to_label_idx = {c["id"]: i for i, c in enumerate(self.coco_data.get("categories", []))}

        # Only include images that have at least one annotation.
        # Matches CocoMaskRCNNDataset behaviour and prevents the empty-image
        # phantom-instance fallback from polluting training batches.
        self.image_ids = [img_id for img_id in self.images_by_id
                          if img_id in self.anns_by_image]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_by_id[img_id]
        img_path = self.images_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        anns = self.anns_by_image.get(img_id, [])
        instance_masks = []
        class_labels = []

        for ann in anns:
            seg = ann.get("segmentation")
            if not seg:
                continue
            raw_cat_id = ann["category_id"]
            cat_id = self.cat_id_to_label_idx.get(raw_cat_id, raw_cat_id)
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                rle  = mask_utils.merge(rles)
            elif isinstance(seg, dict):
                rle = seg
            else:
                continue
            m = mask_utils.decode(rle)
            if m.sum() > 0:
                instance_masks.append(m)
                class_labels.append(cat_id)

        # If all annotations decoded to zero area (degenerate polygons), we
        # should not inject a phantom instance — that would pollute the loss
        # with a fake cat_id=0 mask.  Instead we return empty tensors here;
        # the model's criterion handles N=0 gracefully.
        # Note: images with NO annotations at all are excluded in __init__,
        # so this branch only fires for images whose annotations are all
        # zero-area after decoding (an uncommon but valid COCO edge case).
        if len(instance_masks) == 0:
            pass  # mask_tensors / label_tensors stay empty; handled below

        # ── Build pixel_values / pixel_mask via processor (image only) ───────
        # We deliberately do NOT pass segmentation_maps to the processor because
        # its internal instance_id_to_semantic_id lookup compares np.uint8 keys
        # against our Python-int dict and raises KeyError on older transformers.
        # Instead we construct mask_labels and class_labels tensors manually,
        # which is exactly what the model's forward() / criterion expects.
        img_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].squeeze(0)          # (3, H', W')
        pixel_mask   = img_inputs.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.squeeze(0)                        # (H', W')

        # Processor may resize; get the processed spatial size for mask resizing
        _, proc_h, proc_w = pixel_values.shape

        # ── Build mask_labels (N, H', W') float32 and class_labels (N,) int64 ─
        mask_tensors  = []
        label_tensors = []
        for m, cat_id in zip(instance_masks, class_labels):
            # Resize binary mask to match processed image size using nearest interpolation
            m_pil     = PILImage.fromarray(m.astype(np.uint8) * 255).resize(
                            (proc_w, proc_h), PILImage.NEAREST)
            m_resized = (np.array(m_pil) > 128).astype(np.float32)
            mask_tensors.append(torch.from_numpy(m_resized))
            label_tensors.append(torch.tensor(cat_id, dtype=torch.long))

        # Handle the (rare) case where mask_tensors is empty — torch.stack([])
        # raises RuntimeError.  Use zero-shaped tensors instead so the model
        # criterion sees N=0 and skips this image's mask loss correctly.
        if mask_tensors:
            stacked_masks  = torch.stack(mask_tensors)   # (N, H', W')
            stacked_labels = torch.stack(label_tensors)  # (N,)
        else:
            stacked_masks  = torch.zeros((0, proc_h, proc_w), dtype=torch.float32)
            stacked_labels = torch.zeros((0,),                 dtype=torch.long)

        result = {
            "pixel_values": pixel_values,
            "mask_labels":  stacked_masks,
            "class_labels": stacked_labels,
            "image_id":     img_id,
            "orig_size":    (h, w),                           # actual image dims for post-processing
        }
        if pixel_mask is not None:
            result["pixel_mask"] = pixel_mask
        return result


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask   = torch.stack([b["pixel_mask"] for b in batch]) if "pixel_mask" in batch[0] else None

    # mask_labels: list of (N_i, H', W') tensors — N_i varies per image
    # class_labels: list of (N_i,) tensors
    # Keep as lists; the model's criterion handles variable N per image.
    mask_labels  = [b["mask_labels"]  for b in batch]
    class_labels = [b["class_labels"] for b in batch]
    image_ids    = [b["image_id"]     for b in batch]
    # orig_size: list of (h, w) tuples — actual image dimensions before processor resize.
    # Used in the val loop so post_process_instance_segmentation rescales masks to the
    # correct resolution rather than the previously hardcoded (640, 640).
    orig_sizes   = [b["orig_size"]    for b in batch]

    res = {
        "pixel_values": pixel_values,
        "mask_labels":  mask_labels,
        "class_labels": class_labels,
        "image_ids":    image_ids,
        "orig_sizes":   orig_sizes,
    }
    if pixel_mask is not None:
        res["pixel_mask"] = pixel_mask
    return res



def _split_coco_json_inplace(coco_data, train_ratio=0.8, seed=42):
    """
    Split a single COCO JSON dict into train/val in-memory (no disk I/O).

    Returns (train_dict, val_dict) — both are plain dicts with
    'info', 'licenses', 'categories', 'images', 'annotations' keys.
    The split is deterministic given the same seed.
    """
    all_images = coco_data.get("images", [])
    rng = random.Random(seed)
    shuffled = list(all_images)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_ratio))
    train_imgs = shuffled[:n_train]
    val_imgs   = shuffled[n_train:]

    train_ids = {img["id"] for img in train_imgs}
    val_ids   = {img["id"] for img in val_imgs}

    all_anns = coco_data.get("annotations", [])
    train_anns = [a for a in all_anns if a["image_id"] in train_ids]
    val_anns   = [a for a in all_anns if a["image_id"] in val_ids]

    base = {
        "info":       coco_data.get("info", {}),
        "licenses":   coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
    }
    train_dict = {**base, "images": train_imgs, "annotations": train_anns}
    val_dict   = {**base, "images": val_imgs,   "annotations": val_anns}
    return train_dict, val_dict


def main():
    parser = argparse.ArgumentParser(description="Mask2Former Instance Segmentation Training")
    parser.add_argument("--dataset", required=True, help="Dataset name or path to dataset directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers. Default lowered 8→4 for Windows: 'spawn' "
                             "multiprocessing means each worker boots a fresh interpreter, so "
                             "8 workers consume ~3-4 GB RAM before a single batch is loaded. "
                             "4 workers with prefetch_factor=4 is more efficient on Windows.")
    parser.add_argument("--output_dir", default="runs_comparison/mask2former")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to train for quick capacity testing.")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Run validation + COCO eval every N epochs (default: 5). "
                             "Use 1 to validate every epoch.")
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4). "
                             "Effective batch = batch * accum_steps. "
                             "With batch=2 and accum_steps=4 the optimizer sees an effective "
                             "batch of 8 without holding 8 images in VRAM simultaneously.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (number of validation epochs without improvement).")
    parser.add_argument("--conf_thresh", type=float, default=0.55,
                        help="Validation confidence threshold for post-processing predictions.")
    parser.add_argument("--no_object_weight", type=float, default=0.25,
                        help="Loss penalty weight for no-object / background queries (default: 0.25, range: 0.20-0.30 to fix false positives).")
    parser.add_argument("--num_queries", type=int, default=100,
                        help="Number of query slots in Mask2Former decoder (default: 100).")
    parser.add_argument("--class_weight", type=float, default=2.0,
                        help="Loss weight for classification loss (default: 2.0).")
    parser.add_argument("--mask_weight", type=float, default=5.0,
                        help="Loss weight for binary focal mask loss (default: 5.0).")
    parser.add_argument("--dice_weight", type=float, default=5.0,
                        help="Loss weight for Dice mask loss (default: 5.0).")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Wrap the model with torch.compile(backend='inductor') before "
                             "training.  Benchmark (Stage 5) found 9 graph breaks in the "
                             "default config, all caused by Tensor.item() in "
                             "multi_scale_deformable_attention.  Setting "
                             "capture_scalar_outputs=True (done automatically here) resolves "
                             "all of them and gives inductor a single clean graph covering the "
                             "full Swin backbone + decoder.  Expected 10-30%% throughput gain "
                             "on GPU; not beneficial on CPU.  Disable if you hit compile "
                             "errors on a new torch/transformers version.")
    # ── W&B / combined_data args (new — do not conflict with any existing args) ─────
    parser.add_argument("--wandb_project", default="car-damage-parts-seg",
                        help="W&B project name (default: 'car-damage-parts-seg').")
    parser.add_argument("--wandb_run_name", default=None,
                        help="Optional W&B run display name. Auto-generated if not set.")
    parser.add_argument("--no_wandb", action="store_true", default=False,
                        help="Disable W&B logging entirely.")
    parser.add_argument("--combined_data", default=None,
                        help="Path to combined_data/ dir (pre-cleaned COCO, 32 classes). "
                             "Overrides --dataset. Performs in-memory 80/20 train/val split "
                             "on 'annotations/instances_unified.json'. Images must be in "
                             "'images/' subdir (flat, no train/val subdirs).")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{'OK' if device == 'cuda' else 'WARNING'}] Using device: {device}")

    # ── combined_data path (pre-cleaned COCO, overrides --dataset) ───────────
    _use_combined_data = args.combined_data is not None
    if _use_combined_data:
        cd_root      = Path(args.combined_data).resolve()
        cd_json_path = cd_root / "annotations" / "instances_unified.json"
        cd_images    = cd_root / "images"  # flat dir — same dir used for train + val
        print(f"[*] combined_data mode: {cd_root}")
        print(f"    Annotation file: {cd_json_path}")
        if not cd_json_path.exists():
            print(f"[ERROR] combined_data annotation not found: {cd_json_path}")
            sys.exit(1)
        print("[*] Loading combined_data COCO JSON (may take a moment for large files)...")
        with open(cd_json_path, "r") as f:
            _combined_coco = json.load(f)
        coco_train_data, _coco_val_data = _split_coco_json_inplace(
            _combined_coco, train_ratio=0.8, seed=42
        )
        print(f"    Split: {len(coco_train_data['images'])} train / "
              f"{len(_coco_val_data['images'])} val images")
        print(f"    Categories: {len(coco_train_data['categories'])}")
        # train_images == val_images (flat dir); Dataset resolves filenames from JSON
        train_images = cd_images
        val_images   = cd_images
        # Store val data so the Dataset class can use it
        _val_coco_data_combined = _coco_val_data
    else:
        ds_path = Path(args.dataset)
        if ds_path.is_dir():
            base = ds_path if (ds_path / "coco_train.json").exists() else ds_path / "combined_carparts"
            train_images = base / "images" / "train"
            train_json   = base / "coco_train.json"
            val_images   = base / "images" / "val"
            val_json     = base / "coco_val.json"
        else:
            train_images = PROJECT_ROOT / "datasets" / args.dataset / "images" / "train"
            train_json   = PROJECT_ROOT / "datasets" / args.dataset / "coco_train.json"
            val_images   = PROJECT_ROOT / "datasets" / args.dataset / "images" / "val"
            val_json     = PROJECT_ROOT / "datasets" / args.dataset / "coco_val.json"

        with open(train_json, "r") as f:
            coco_train_data = json.load(f)

    num_classes = len(coco_train_data["categories"])
    class_names = [c["name"] for c in coco_train_data["categories"]]

    # Build name→COCO-category-id lookup from the training JSON.
    # Used later to remap model label_ids (0-indexed) → COCO category_ids (1-indexed).
    coco_name_to_cat_id = {c["name"]: c["id"] for c in coco_train_data["categories"]}

    id2label = {i: c["name"] for i, c in enumerate(coco_train_data["categories"])}
    label2id = {c["name"]: i for i, c in enumerate(coco_train_data["categories"])}

    model_id = "facebook/mask2former-swin-tiny-coco-instance"
    print(f"[*] Loading pretrained Mask2Former ({model_id})...")
    processor = Mask2FormerImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.no_object_weight = args.no_object_weight
    model.config.num_queries = args.num_queries
    model.config.class_weight = args.class_weight
    model.config.mask_weight = args.mask_weight
    model.config.dice_weight = args.dice_weight

    # Ensure internal criterion (if initialized) uses the updated weights
    if hasattr(model, "criterion") and model.criterion is not None:
        if hasattr(model.criterion, "no_object_weight"):
            model.criterion.no_object_weight = args.no_object_weight
        if hasattr(model.criterion, "num_queries"):
            model.criterion.num_queries = args.num_queries
        if hasattr(model.criterion, "weight_dict") and isinstance(model.criterion.weight_dict, dict):
            model.criterion.weight_dict["loss_ce"] = args.class_weight
            model.criterion.weight_dict["loss_mask"] = args.mask_weight
            model.criterion.weight_dict["loss_dice"] = args.dice_weight

    print(f"[OK] Loss weights configured: no_object_weight={args.no_object_weight} | "
          f"class_weight={args.class_weight} | mask_weight={args.mask_weight} | "
          f"dice_weight={args.dice_weight} | num_queries={args.num_queries}")
    model.to(device)

    # ── W&B init (after all hparams are known, before training starts) ───────
    _use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if _use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": "mask2former-swin-tiny-coco-instance",
                "dataset": args.combined_data if _use_combined_data else args.dataset,
                "num_classes": num_classes,
                "class_names": class_names,
                "epochs": args.epochs,
                "batch_size": args.batch,
                "effective_batch": args.batch * args.accum_steps,
                "lr": args.lr,
                "backbone_lr": args.lr * 0.1,
                "num_workers": args.num_workers,
                "accum_steps": args.accum_steps,
                "val_interval": args.val_interval,
                "patience": args.patience,
                "conf_thresh": args.conf_thresh,
                "no_object_weight": args.no_object_weight,
                "num_queries": args.num_queries,
                "class_weight": args.class_weight,
                "mask_weight": args.mask_weight,
                "dice_weight": args.dice_weight,
                "compile": args.compile,
                "optimizer": "AdamW",
                "weight_decay": 1e-4,
                "scheduler": "cosine_with_linear_warmup",
                "warmup_fraction": 0.10,
                "device": device,
                "output_dir": args.output_dir,
                "combined_data_mode": _use_combined_data,
            },
        )
        print(f"[OK] W&B run initialised: {wandb.run.url}")
    elif not _WANDB_AVAILABLE and not args.no_wandb:
        print("[WARNING] wandb not installed — W&B logging disabled. "
              "Install with: pip install wandb")

    # ── torch.compile (optional, GPU-only) ───────────────────────────────────
    # Three dynamo issues specific to Mask2Former on this VM:
    #
    # 1. capture_scalar_outputs=True  — resolves 9 graph breaks in
    #    multi_scale_deformable_attention caused by Tensor.item() calls.
    #
    # 2. suppress_errors=True  — two environment-level failures cause dynamo
    #    to crash the process rather than falling back gracefully:
    #
    #    a) onnxruntime built against NumPy 1.x fails to import under NumPy 2.x
    #       with SystemError("__ARRAY_API not found") during dynamo's lazy
    #       backend registration.  This fires on the FIRST FORWARD PASS, not at
    #       torch.compile() call time, so try/except around torch.compile() does
    #       not help.  Permanent fix: pip install --upgrade onnxruntime-gpu
    #
    #    b) linear_sum_assignment (Mask2Former's Hungarian matcher) is a compiled
    #       C extension that dynamo partially inlines.  When it reaches the
    #       .numpy() call inside it, it fails with:
    #       "TorchRuntimeError: .numpy() is not supported for tensor subclasses"
    #       allow_in_graph() does NOT prevent inlining for partially-traceable
    #       C extensions — suppress_errors is the correct escape hatch here.
    #
    #    suppress_errors=True tells dynamo to silently fall back to eager for any
    #    subgraph that fails compilation.  The backbone and decoder still get
    #    compiled; only the criterion (loss) runs in eager.  This is a net win
    #    over no compile at all.
    #
    #    Once onnxruntime is upgraded (pip install --upgrade onnxruntime-gpu),
    #    the onnxrt crash disappears and suppress_errors becomes a belt-and-
    #    suspenders safety net rather than the primary workaround.
    if args.compile and device == "cuda":
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        # cache_size_limit: dynamo caches one compiled graph per unique input
        # signature.  The val loop switches grad_mode (train→eval), which looks
        # like a new signature.  Default limit of 8 fills up across train/val
        # transitions and dynamo falls back to eager permanently for that frame.
        # 64 gives enough headroom for all grad_mode / AMP / batch-size variants
        # without unbounded memory growth.
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model, backend="inductor")
        print("[OK] torch.compile enabled (inductor backend, capture_scalar_outputs=True, suppress_errors=True, cache_size_limit=64)")
        print("     Note: suppress_errors=True means dynamo falls back to eager for any")
        print("     subgraph that cannot be compiled (criterion/loss runs in eager mode).")
        print("     To resolve permanently: pip install --upgrade onnxruntime-gpu")
    elif args.compile:
        print("[WARNING] --compile requested but device is CPU — skipping torch.compile")

    # ── Stage-2 audit: log point-sampling config so every run is self-documenting ──
    # Per the Mask2Former paper, loss is computed on randomly sampled points rather
    # than the full mask, reducing training memory ~3×.  These three values control
    # that sampling.  If train_num_points were 0 or absent, point sampling would be
    # off and we'd be paying full-resolution mask loss cost.
    cfg = model.config
    _npts   = getattr(cfg, "train_num_points",        "MISSING")
    _over   = getattr(cfg, "oversample_ratio",         "MISSING")
    _imp    = getattr(cfg, "importance_sample_ratio",  "MISSING")
    print(f"\n[Mask2Former config — point-based mask loss sampling]")
    print(f"  train_num_points        = {_npts}   (official default: 12544)")
    print(f"  oversample_ratio        = {_over}  (official default: 3.0)")
    print(f"  importance_sample_ratio = {_imp}  (official default: 0.75)")
    if _npts == "MISSING" or _npts == 0:
        print(f"  [WARNING] Point sampling is OFF — full-resolution mask loss is active.")
        print(f"            Set model.config.train_num_points = 12544 before training.")
    else:
        print(f"  [OK] Point sampling active — expected ~3× memory reduction vs full-mask loss.")
    print()

    # ── Dataset construction — handles both normal and combined_data paths ────
    # For combined_data mode the Dataset is initialised directly from the
    # in-memory dicts rather than from file paths.  COCOMask2FormerDataset
    # still reads from disk for images; only the JSON is pre-loaded.
    if _use_combined_data:
        # Temporarily write split JSONs to a temp location so the existing
        # Dataset class can read them via file path.
        import tempfile, json as _json
        _tmp_dir = Path(tempfile.mkdtemp(prefix="m2f_combined_data_"))
        _train_json_tmp = _tmp_dir / "coco_train_tmp.json"
        _val_json_tmp   = _tmp_dir / "coco_val_tmp.json"
        with open(_train_json_tmp, "w") as _f:
            _json.dump(coco_train_data, _f)
        with open(_val_json_tmp, "w") as _f:
            _json.dump(_val_coco_data_combined, _f)
        train_ds = COCOMask2FormerDataset(train_images, str(_train_json_tmp), processor)
        val_ds   = COCOMask2FormerDataset(val_images,   str(_val_json_tmp),   processor)
    else:
        train_ds = COCOMask2FormerDataset(train_images, train_json, processor)
        val_ds   = COCOMask2FormerDataset(val_images,   val_json,   processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)

    # ── Param groups: backbone at 0.1× LR, head/decoder at full LR ─────────
    # The Swin backbone is pretrained; applying the same LR as the randomly-
    # initialised segmentation head destabilises learned features in early epochs.
    # Standard fine-tuning practice is backbone ~10× lower than the head.
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # pixel_level_module.encoder is the Swin backbone
        if "pixel_level_module.encoder" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    backbone_lr = args.lr * 0.1   # e.g. 1e-5 when lr=1e-4
    head_lr     = args.lr         # e.g. 1e-4

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params,     "lr": head_lr,     "name": "head"},
        ],
        weight_decay=1e-4,
    )
    print(f"[OK] AdamW param groups: backbone lr={backbone_lr:.2e}  |  head lr={head_lr:.2e}")
    print(f"     backbone params: {len(backbone_params)}  |  head params: {len(head_params)}")

    # ── Cosine LR schedule with linear warmup ────────────────────────────────
    # Warmup over 10% of total optimizer steps prevents early gradient explosion
    # when the head is randomly initialised.  After warmup, cosine decay brings
    # LR smoothly to near-zero, avoiding the stall that a flat LR causes in
    # later epochs.
    #
    # Total optimizer steps = ceil(batches_per_epoch / accum_steps) * epochs
    batches_per_epoch  = len(train_loader)
    steps_per_epoch    = max(1, batches_per_epoch // args.accum_steps)
    total_steps        = steps_per_epoch * args.epochs
    warmup_steps       = max(1, int(total_steps * 0.10))

    def warmup_cosine_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

    print(f"[OK] LR schedule: linear warmup ({warmup_steps} steps) → cosine decay ({total_steps} total steps)")
    print(f"     steps_per_epoch={steps_per_epoch}  |  accum_steps={args.accum_steps}  |  "
          f"effective batch={args.batch * args.accum_steps}")

    # AMP: mixed precision for faster training and lower VRAM usage
    # (mirrors train_maskrcnn.py which fixed NaN/overflow with float() casting)
    use_amp = device == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    target_out_dir = Path(args.output_dir).resolve()
    os.makedirs(target_out_dir, exist_ok=True)

    # ── Bug 1 fix: build label_id → COCO category_id mapping ────────────────
    # model.config.id2label maps 0-indexed model output indices → class names.
    # COCO ground-truth uses category_ids from the JSON (typically 1-indexed).
    # Without this map every prediction gets the wrong category_id and COCO
    # eval finds zero matching GT annotations → AP = 0 for all classes.
    id2label = model.config.id2label  # {0: "back_bumper", 1: "back_left_door", ...}
    label_id_to_cat_id = {}
    for label_id, name in id2label.items():
        label_id_int = int(label_id) if isinstance(label_id, (int, str)) and str(label_id).isdigit() else label_id
        if name in coco_name_to_cat_id:
            label_id_to_cat_id[label_id_int] = coco_name_to_cat_id[name]
        elif isinstance(label_id_int, int) and label_id_int < len(class_names):
            label_id_to_cat_id[label_id_int] = coco_name_to_cat_id.get(class_names[label_id_int], label_id_int)

    missing = [name for name in id2label.values() if name not in coco_name_to_cat_id]
    if missing:
        print(f"[WARNING] {len(missing)} model label(s) not found in COCO categories: {missing[:5]}")
    print(f"[OK] label_id→category_id map built: {len(label_id_to_cat_id)}/{len(id2label)} labels mapped")

    if _use_combined_data:
        _logger_train_json = str(_train_json_tmp)
        _logger_val_json   = str(_val_json_tmp)
        _evaluator_val_json = str(_val_json_tmp)
    else:
        _logger_train_json  = str(train_json)
        _logger_val_json    = str(val_json)
        _evaluator_val_json = str(val_json)

    logger = UnifiedLogger(str(target_out_dir), "mask2former", patience=args.patience)
    logger.print_dataset_health(_logger_train_json, _logger_val_json)
    evaluator = UnifiedEvaluator(_evaluator_val_json, str(val_images), class_names, str(target_out_dir))

    print(f"[*] Starting Mask2Former Training for {args.epochs} epochs...")

    global_step = 0  # counts optimizer steps (not batch steps) for scheduler
    _best_val_metric_str = "N/A"  # tracked for wandb completion alert

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            start_time = time.time()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)  # reset at epoch start
            num_images = 0

            print(f"\n[Epoch {epoch}/{args.epochs}] Training...")

            for batch_idx, batch in enumerate(train_loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                pixel_values = batch["pixel_values"].to(device)
                mask_labels  = [m.to(device) for m in batch["mask_labels"]]
                class_labels = [c.to(device) for c in batch["class_labels"]]
                num_images += len(pixel_values)

                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    progress = (batch_idx + 1) / len(train_loader) * 100
                    avg_loss = running_loss / max(1, batch_idx + 1)
                    print(f"  Batch {batch_idx + 1}/{len(train_loader)} ({progress:.1f}%) | Loss: {avg_loss:.4f}")
                    sys.stdout.flush()  # Force output to appear immediately

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(
                        pixel_values=pixel_values,
                        mask_labels=mask_labels,
                        class_labels=class_labels
                    )
                    # Scale loss by 1/accum_steps so that gradients accumulated over
                    # accum_steps micro-batches equal a true mean over the full batch.
                    loss = outputs.loss / args.accum_steps

                scaler.scale(loss).backward()

                # Only step the optimizer every accum_steps batches (or at epoch end)
                is_last_batch = (batch_idx + 1) == len(train_loader)
                if (batch_idx + 1) % args.accum_steps == 0 or is_last_batch:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    if scale_before <= scale_after:
                        global_step += 1
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # ── LR diagnostic: print per-group LRs at first two optimizer steps
                    #    of epoch 1 so we can confirm warmup + param groups are wired correctly.
                    if epoch == 1 and global_step <= 2:
                        lrs = {pg["name"]: pg["lr"] for pg in optimizer.param_groups}
                        print(f"    [LR check] optimizer step {global_step}: "
                              + "  ".join(f"{k}={v:.3e}" for k, v in lrs.items()))

                # Accumulate unscaled loss for logging (multiply back by accum_steps)
                running_loss += loss.item() * args.accum_steps

            epoch_time = time.time() - start_time
            avg_train_loss = running_loss / max(len(train_loader), 1)
            images_sec = num_images / epoch_time if epoch_time > 0 else 0

            print(f"\n[Epoch {epoch}/{args.epochs}] Complete:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Time: {epoch_time:.1f}s | {images_sec:.2f} img/s")
            print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            sys.stdout.flush()

            # Evaluation
            val_metrics = None
            run_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)
            if run_val:
                print(f"\n[Epoch {epoch}] Running validation (every {args.val_interval} epochs)...")
                model.eval()
                predictions_by_image = {}
                running_val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        pixel_values = batch["pixel_values"].to(device)
                        mask_labels  = [m.to(device) for m in batch["mask_labels"]]
                        class_labels = [c.to(device) for c in batch["class_labels"]]

                        # autocast in val matches the precision used during training,
                        # halving VRAM usage and speeding up val forward passes.
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            outputs = model(
                                pixel_values=pixel_values,
                                mask_labels=mask_labels,
                                class_labels=class_labels
                            )
                            if outputs.loss is not None:
                                running_val_loss += outputs.loss.item()
                                val_batches += 1

                        # Use actual image dimensions (h, w) per sample so that
                        # post_process_instance_segmentation rescales prediction masks
                        # back to the true image resolution.
                        results = processor.post_process_instance_segmentation(
                            outputs, target_sizes=batch["orig_sizes"], threshold=args.conf_thresh
                        )

                        for img_id, res in zip(batch["image_ids"], results):
                            preds = []
                            masks = res["segmentation"].cpu().numpy()
                            labels = res["segments_info"]
                            for info in labels:
                                # info["label_id"] is a 0-indexed model output index
                                label_id = info["label_id"]
                                cid = label_id_to_cat_id.get(label_id, -1)
                                if cid == -1:
                                    continue  # unknown label — skip rather than corrupt eval
                                score = info["score"]
                                m = (masks == info["id"])
                                if not m.any():
                                    continue
                                ys, xs = np.where(m)
                                x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                                preds.append({
                                    "category_id": cid,
                                    "score": float(score),
                                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                                    "segmentation": m
                                })
                            predictions_by_image[img_id] = preds

                avg_val_loss = running_val_loss / max(1, val_batches) if val_batches > 0 else float('nan')
                val_metrics = evaluator.evaluate(epoch, predictions_by_image)
            else:
                avg_val_loss = float('nan')
                print(f"\n[Epoch {epoch}] Skipping validation (next at epoch {epoch + (args.val_interval - epoch % args.val_interval)})")

            train_stats = {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[1]["lr"],  # log head LR (index 1 = head group)
                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                "epoch_time_sec": epoch_time,
                "images_sec": len(train_ds) / max(epoch_time, 1e-3)
            }

            should_stop, is_best = logger.log_epoch(epoch, args.epochs, train_stats, val_metrics)

            # ── W&B per-epoch logging ────────────────────────────────────────
            if _use_wandb:
                _wandb_log = {
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "train/lr_head":     optimizer.param_groups[1]["lr"],
                    "train/lr_backbone": optimizer.param_groups[0]["lr"],
                    "train/gpu_mem_gb":  torch.cuda.max_memory_allocated() / 1e9
                                         if torch.cuda.is_available() else 0,
                    "train/epoch_time_sec": epoch_time,
                    "train/images_per_sec": images_sec,
                }
                if run_val:
                    _wandb_log["val/loss"] = avg_val_loss
                    if val_metrics:
                        # val_metrics is a dict from UnifiedEvaluator.evaluate()
                        # e.g. {"AP": 0.xx, "AP50": 0.xx, ...}
                        for _k, _v in val_metrics.items():
                            try:
                                _wandb_log[f"val/{_k}"] = float(_v)
                            except (TypeError, ValueError):
                                pass
                        # Track best metric string for completion alert
                        _ap = val_metrics.get("AP", val_metrics.get("mAP", None))
                        if _ap is not None:
                            _best_val_metric_str = f"AP={float(_ap):.4f}"
                wandb.log(_wandb_log, step=epoch)

            unwrapped_model = getattr(model, "_orig_mod", model)
            weights_dir = target_out_dir / "weights"
            os.makedirs(weights_dir, exist_ok=True)
            last_dir = weights_dir / "last"
            unwrapped_model.save_pretrained(str(last_dir))
            processor.save_pretrained(str(last_dir))
            legacy_last = target_out_dir / "last_model"
            unwrapped_model.save_pretrained(str(legacy_last))
            processor.save_pretrained(str(legacy_last))

            if is_best:
                best_dir = weights_dir / "best"
                unwrapped_model.save_pretrained(str(best_dir))
                processor.save_pretrained(str(best_dir))  # saves preprocessor_config.json needed for inference
                # Also legacy best_model location for backward compat
                legacy_best = target_out_dir / "best_model"
                unwrapped_model.save_pretrained(str(legacy_best))
                processor.save_pretrained(str(legacy_best))

            if should_stop:
                print("[!] Early stopping triggered.")
                break

    except Exception as _exc:
        print(f"[ERROR] Training crashed: {_exc}")
        if _use_wandb:
            wandb.alert(
                title="Mask2Former Training Crashed",
                text=f"Exception: {type(_exc).__name__}: {_exc}",
                level=wandb.AlertLevel.ERROR,
            )
            wandb.finish(exit_code=1)
        raise


    logger.print_final_summary()
    print(f"[OK] Mask2Former training completed!")

    # ── W&B completion alert + finish ────────────────────────────────────────
    if _use_wandb:
        wandb.alert(
            title="Mask2Former Training Complete",
            text=(
                f"Training finished after {args.epochs} epoch(s).\n"
                f"Best validation metric: {_best_val_metric_str}\n"
                f"Output dir: {target_out_dir}"
            ),
        )
        wandb.finish()

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[CUDA Peak VRAM] {peak_vram:.2f} GB")


if __name__ == "__main__":
    main()
