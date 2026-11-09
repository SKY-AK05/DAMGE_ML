import json
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools.mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    COCO = None
    COCOeval = None
    mask_utils = None

# Pairs we specifically care about mixing up for Left/Right confusion
CONFUSION_PAIRS_TO_WATCH = [
    ("front_left_door", "back_left_door"),
    ("front_right_door", "back_right_door"),
    ("front_left_door", "front_right_door"),
    ("back_left_door", "back_right_door"),
    ("front_left_light", "front_right_light"),
    ("back_left_light", "back_right_light"),
    ("left_mirror", "right_mirror"),
]

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Computes boundary of a mask.
    dilation_ratio is relative to the diagonal of the bounding box.
    """
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros_like(mask)
    
    bbox_diag = np.sqrt((xs.max() - xs.min())**2 + (ys.max() - ys.min())**2)
    dilation = int(round(dilation_ratio * bbox_diag))
    if dilation < 1:
        dilation = 1
        
    kernel = np.ones((3, 3), dtype=np.uint8)
    # Pad to avoid boundary issues
    mask_pad = np.pad(mask, ((1, 1), (1, 1)), mode='constant')
    mask_erode = cv2.erode(mask_pad.astype(np.uint8), kernel, iterations=dilation)
    mask_dilate = cv2.dilate(mask_pad.astype(np.uint8), kernel, iterations=dilation)
    boundary = (mask_dilate - mask_erode)[1:-1, 1:-1]
    return boundary > 0

def compute_boundary_iou(gt_mask, pred_mask, dilation_ratio=0.02):
    gt_boundary = mask_to_boundary(gt_mask, dilation_ratio)
    pred_boundary = mask_to_boundary(pred_mask, dilation_ratio)
    
    intersection = np.logical_and(gt_boundary, pred_boundary).sum()
    union = np.logical_or(gt_boundary, pred_boundary).sum()
    if union == 0:
        return 0.0
    return intersection / union

def box_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0

class UnifiedEvaluator:
    def __init__(self, val_json_path, val_images_dir, class_names, output_dir):
        self.val_json_path = val_json_path
        self.val_images_dir = Path(val_images_dir)
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        
        if PYCOCOTOOLS_AVAILABLE:
            self.coco_gt = COCO(val_json_path)
            img_ids = sorted(self.coco_gt.getImgIds())
            self.fixed_vis_img_ids = img_ids[:10]
            self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco_gt.loadCats(self.coco_gt.getCatIds())}
            self.name_to_cat_id = {name: id for id, name in self.cat_id_to_name.items()}
        else:
            print("[WARNING] pycocotools is not installed. UnifiedEvaluator will run in fallback mode.")
            self.coco_gt = None
            self.fixed_vis_img_ids = []
            self.cat_id_to_name = {}
            self.name_to_cat_id = {}

    def evaluate(self, epoch, predictions_by_image_id):
        """
        predictions_by_image_id: dict mapping image_id -> list of predictions
        Prediction format: {'category_id': int, 'score': float, 'bbox': [x,y,w,h], 'segmentation': np.ndarray (H, W, bool)}
        """
        if not PYCOCOTOOLS_AVAILABLE or self.coco_gt is None:
            return self._empty_metrics()
        # 1. Prepare predictions for COCOEval
        coco_res = []
        has_masks = False
        for img_id, preds in predictions_by_image_id.items():
            for p in preds:
                entry = {
                    "image_id": img_id,
                    "category_id": p['category_id'],
                    "bbox": p['bbox'],  # [x, y, w, h]
                    "score": p['score'],
                }
                if p.get('segmentation') is not None:
                    has_masks = True
                    mask_bool = np.asfortranarray(p['segmentation'].astype(np.uint8))
                    rle = mask_utils.encode(mask_bool)
                    rle['counts'] = rle['counts'].decode('utf-8')
                    entry["segmentation"] = rle
                
                coco_res.append(entry)
        
        # If no predictions, return zeros
        if len(coco_res) == 0:
            return self._empty_metrics()

        # Load predictions into COCO
        coco_dt = self.coco_gt.loadRes(coco_res)

        # 2. Run COCOeval for Box
        coco_eval_box = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval_box.evaluate()
        coco_eval_box.accumulate()
        coco_eval_box.summarize()
        
        # 3. Run COCOeval for Mask (only if model predicts masks)
        if has_masks:
            coco_eval_mask = COCOeval(self.coco_gt, coco_dt, 'segm')
            coco_eval_mask.evaluate()
            coco_eval_mask.accumulate()
            coco_eval_mask.summarize()
        else:
            class DummyEval:
                stats = [0.0] * 12
            coco_eval_mask = DummyEval()

        # 4. Custom Metrics (Boundary IoU, Dice, Mask IoU, Left/Right Mixup)
        # We need to match predictions to ground truth
        total_mask_iou = 0
        total_dice = 0
        total_boundary_iou = 0
        matched_instances = 0
        
        n_classes = len(self.cat_id_to_name)
        # Matrix size + 1 for background/missed
        confusion_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)
        
        # Create an index mapping for the matrix
        sorted_cat_ids = sorted(list(self.cat_id_to_name.keys()))
        cat_id_to_idx = {cid: i for i, cid in enumerate(sorted_cat_ids)}
        bg_idx = n_classes

        for img_id in self.coco_gt.getImgIds():
            gt_ann_ids = self.coco_gt.getAnnIds(imgIds=[img_id])
            gt_anns = self.coco_gt.loadAnns(gt_ann_ids)
            preds = predictions_by_image_id.get(img_id, [])
            
            matched_pred = set()
            for gt in gt_anns:
                gt_bbox = [gt['bbox'][0], gt['bbox'][1], gt['bbox'][0]+gt['bbox'][2], gt['bbox'][1]+gt['bbox'][3]]
                gt_mask = self.coco_gt.annToMask(gt)
                
                best_iou = 0
                best_pred_idx = -1
                
                for i, pred in enumerate(preds):
                    if i in matched_pred:
                        continue
                    pred_bbox = [pred['bbox'][0], pred['bbox'][1], pred['bbox'][0]+pred['bbox'][2], pred['bbox'][1]+pred['bbox'][3]]
                    iou = box_iou(gt_bbox, pred_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = i
                
                gt_idx = cat_id_to_idx[gt['category_id']]
                
                if best_iou >= 0.5 and best_pred_idx != -1:
                    matched_pred.add(best_pred_idx)
                    best_pred = preds[best_pred_idx]
                    pred_idx = cat_id_to_idx[best_pred['category_id']]
                    confusion_matrix[gt_idx, pred_idx] += 1
                    
                    # Compute pixel metrics only for matches if prediction has masks
                    p_mask = best_pred.get('segmentation')
                    if p_mask is not None:
                        # gt_mask is at native image resolution (from annToMask).
                        # p_mask from Mask R-CNN is at the model's fixed input size (e.g. 640x640).
                        # Resize prediction mask to ground-truth resolution before comparison.
                        p_mask_arr = np.asarray(p_mask)
                        if gt_mask.shape != p_mask_arr.shape:
                            p_mask_arr = cv2.resize(
                                p_mask_arr.astype(np.uint8),
                                (gt_mask.shape[1], gt_mask.shape[0]),  # cv2 takes (W, H)
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        intersection = np.logical_and(gt_mask, p_mask_arr).sum()
                        union = np.logical_or(gt_mask, p_mask_arr).sum()
                        if union > 0:
                            m_iou = intersection / union
                            dice = (2 * intersection) / (gt_mask.sum() + p_mask_arr.sum())
                            b_iou = compute_boundary_iou(gt_mask, p_mask_arr)
                            
                            total_mask_iou += m_iou
                            total_dice += dice
                            total_boundary_iou += b_iou
                            matched_instances += 1
                else:
                    confusion_matrix[gt_idx, bg_idx] += 1
                    
            for i, pred in enumerate(preds):
                if i not in matched_pred:
                    pred_idx = cat_id_to_idx[pred['category_id']]
                    confusion_matrix[bg_idx, pred_idx] += 1

        # Calculate Left/Right Mixup Rate
        mixup_total = 0
        mixup_errors = 0
        for a, b in CONFUSION_PAIRS_TO_WATCH:
            if a not in self.name_to_cat_id or b not in self.name_to_cat_id:
                continue
            ia = cat_id_to_idx[self.name_to_cat_id[a]]
            ib = cat_id_to_idx[self.name_to_cat_id[b]]
            
            a_total = confusion_matrix[ia, :].sum()
            b_total = confusion_matrix[ib, :].sum()
            a_to_b = confusion_matrix[ia, ib]
            b_to_a = confusion_matrix[ib, ia]
            
            mixup_total += (a_total + b_total)
            mixup_errors += (a_to_b + b_to_a)
            
        lr_mixup_rate = mixup_errors / mixup_total if mixup_total > 0 else 0.0

        # Calculate Per-class AP
        # COCOeval stores eval['precision'] with shape [T, R, K, A, M] and
        # eval['recall'] with shape [T, K, A, M].  The K dimension is indexed by
        # the position of category_id within coco_eval_mask.params.catIds — which
        # is NOT necessarily the same order as sorted_cat_ids.  We must build a
        # cat_id → K-index map from the actual COCOeval params to avoid reading
        # the wrong column and getting all-zero / misaligned per-class numbers.
        per_class_metrics = {}
        has_eval_dict = hasattr(coco_eval_mask, 'eval') and coco_eval_mask.eval
        if has_eval_dict:
            coco_cat_ids = list(coco_eval_mask.params.catIds)  # COCOeval's canonical order
            cat_id_to_k = {cid: k for k, cid in enumerate(coco_cat_ids)}
        for cat_id in sorted_cat_ids:
            c_name = self.cat_id_to_name[cat_id]
            if not has_eval_dict or cat_id not in cat_id_to_k:
                per_class_metrics[c_name] = {'map50': 0, 'map5095': 0, 'precision': 0, 'recall': 0}
                continue
            k_idx = cat_id_to_k[cat_id]
            # precision shape: [iouThrs, recThrs, catIds, areaRng, maxDets]
            # iouThrs index 0 = IoU 0.50; areaRng index 0 = all; maxDets index 2 = 100
            prec = coco_eval_mask.eval['precision'][0, :, k_idx, 0, 2]
            ap50 = float(np.mean(prec[prec > -1])) if (prec > -1).any() else 0.0

            prec_all = coco_eval_mask.eval['precision'][:, :, k_idx, 0, 2]
            ap5095 = float(np.mean(prec_all[prec_all > -1])) if (prec_all > -1).any() else 0.0

            # recall shape: [iouThrs, catIds, areaRng, maxDets]
            rec_val = coco_eval_mask.eval['recall'][0, k_idx, 0, 2]
            recall = float(rec_val) if rec_val > -1 else 0.0

            # Per-class precision: the precision value at the highest achieved recall
            # on the IoU=0.50 P-R curve. In pycocotools, unreached recall thresholds
            # remain 0.0, so the rightmost positive value corresponds to the achieved operating point.
            pos_prec = np.where(prec > 0)[0]
            if len(pos_prec) > 0:
                precision_at_op = float(prec[pos_prec[-1]])
            else:
                precision_at_op = 0.0

            per_class_metrics[c_name] = {
                'map50': ap50,
                'map5095': ap5095,
                'precision': precision_at_op,  # real per-class precision from P-R curve
                'recall': recall,
            }

        # Save per-class metrics JSON
        per_class_path = self.output_dir / "per_class_metrics.json"
        try:
            with open(per_class_path, "w", encoding="utf-8") as f:
                json.dump(per_class_metrics, f, indent=2)
        except Exception as _e:
            print(f"[!] Warning: Could not save per_class_metrics.json: {_e}")

        # Highlight low precision (< 0.40) and low recall (< 0.40) classes
        weak_recall = [name for name, m in per_class_metrics.items() if m.get("recall", 0) < 0.40]
        low_precision = [name for name, m in per_class_metrics.items() if m.get("precision", 0) < 0.40]
        if weak_recall or low_precision:
            print("\n[Per-Class Health Check]")
            if weak_recall:
                print(f"  [!] Weak Recall (<0.40): {', '.join(weak_recall[:8])}" + (f" (+{len(weak_recall)-8} more)" if len(weak_recall) > 8 else ""))
            if low_precision:
                print(f"  [!] Low Precision (<0.40): {', '.join(low_precision[:8])}" + (f" (+{len(low_precision)-8} more)" if len(low_precision) > 8 else ""))

        # Save Visualizations
        self._save_visualizations(epoch, predictions_by_image_id)

        avg_mask_iou = total_mask_iou / matched_instances if matched_instances > 0 else 0
        avg_dice = total_dice / matched_instances if matched_instances > 0 else 0
        avg_boundary_iou = total_boundary_iou / matched_instances if matched_instances > 0 else 0

        stats_box = coco_eval_box.stats
        stats_mask = coco_eval_mask.stats

        # COCO stats[] layout:
        #  [0] AP @ IoU 0.50:0.95 (all)   → mAP50-95
        #  [1] AP @ IoU 0.50               → mAP50
        #  [2] AP @ IoU 0.75
        #  [8] AR @ maxDets=100            → mean recall
        def _get_op_precision(coco_eval, k):
            if not (hasattr(coco_eval, 'eval') and coco_eval.eval):
                return 0.0
            p = coco_eval.eval['precision'][0, :, k, 0, 2]
            nz = np.where(p > 0)[0]
            return float(p[nz[-1]]) if len(nz) > 0 else 0.0

        global_prec_box = float(np.mean([
            _get_op_precision(coco_eval_box, cat_id_to_k[cid])
            for cid in sorted_cat_ids if cid in cat_id_to_k
        ])) if sorted_cat_ids and has_eval_dict else stats_box[0]

        global_prec_mask = float(np.mean([
            m['precision'] for m in per_class_metrics.values()
        ])) if per_class_metrics else stats_mask[0]

        metrics = {
            'precision_box': global_prec_box,
            'recall_box': stats_box[8],    # AR @ maxDets=100
            'map50_box': stats_box[1],
            'map5095_box': stats_box[0],   # fixed: was also stats_box[0] but now correctly labelled

            'precision_mask': global_prec_mask,
            'recall_mask': stats_mask[8],
            'map50_mask': stats_mask[1],
            'map5095_mask': stats_mask[0],
            
            'mask_iou': avg_mask_iou,
            'dice_f1': avg_dice,
            'boundary_iou': avg_boundary_iou,
            'left_right_mixup': lr_mixup_rate,
            'per_class': per_class_metrics
        }
        return metrics

    def _empty_metrics(self):
        return {
            'precision_box': 0, 'recall_box': 0, 'map50_box': 0, 'map5095_box': 0,
            'precision_mask': 0, 'recall_mask': 0, 'map50_mask': 0, 'map5095_mask': 0,
            'mask_iou': 0, 'dice_f1': 0, 'boundary_iou': 0, 'left_right_mixup': 0,
            'per_class': {c: {'map50': 0, 'map5095': 0, 'precision': 0, 'recall': 0} for c in self.cat_id_to_name.values()}
        }

    def _save_visualizations(self, epoch, predictions_by_image_id):
        epoch_dir = self.output_dir / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        for img_id in self.fixed_vis_img_ids:
            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = self.val_images_dir / img_info['file_name']
            
            if not img_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            gt_ann_ids = self.coco_gt.getAnnIds(imgIds=[img_id])
            gt_anns = self.coco_gt.loadAnns(gt_ann_ids)
            
            preds = predictions_by_image_id.get(img_id, [])
            
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            axs[0].imshow(img)
            axs[0].set_title("Original")
            axs[0].axis('off')
            
            # Ground Truth
            gt_img = img.copy()
            for ann in gt_anns:
                mask = self.coco_gt.annToMask(ann)
                color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                gt_img[mask > 0] = gt_img[mask > 0] * 0.5 + color * 0.5
            axs[1].imshow(gt_img.astype(np.uint8))
            axs[1].set_title("Ground Truth")
            axs[1].axis('off')
            
            # Prediction
            pred_img = img.copy()
            img_h, img_w = img.shape[:2]
            for pred in preds:
                if pred['score'] < 0.25:
                    continue
                mask = np.asarray(pred['segmentation'])
                # Resize prediction mask to display image resolution if needed
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                pred_img[mask > 0] = pred_img[mask > 0] * 0.5 + color * 0.5
            axs[2].imshow(pred_img.astype(np.uint8))
            axs[2].set_title("Prediction")
            axs[2].axis('off')
            
            # Overlay (Side by Side) - same as Prediction for now
            axs[3].imshow(pred_img.astype(np.uint8))
            axs[3].set_title("Overlay")
            axs[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(epoch_dir / f"{img_info['file_name']}")
            plt.close(fig)
