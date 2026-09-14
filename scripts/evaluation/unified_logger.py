import sys

# Ensure UTF-8 output on Windows terminals
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import csv
import json
import os
import time
from pathlib import Path

class DualWriter:
    """Mirrors a stream to both console and a log file simultaneously."""
    def __init__(self, stream, file_path):
        self.stream = stream
        self.file = open(file_path, "a", encoding="utf-8", errors="replace")
        self._is_tee = True

    def write(self, data):
        try:
            self.stream.write(data)
            self.stream.flush()
        except Exception:
            pass
        try:
            self.file.write(data)
            self.file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.stream, 'isatty', lambda: False)()

    def fileno(self):
        return self.stream.fileno()

class UnifiedLogger:
    def __init__(self, output_dir, model_name, patience=15):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.patience = patience
        self.metrics_csv = self.output_dir / "metrics.csv"
        self.log_file = self.output_dir / "train.log"
        self.best_mask_map50 = -1.0
        self.best_epoch = -1
        self.patience_counter = 0
        self.start_time = time.time()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mirror console output AND stderr to <output_dir>/train.log so all
        # output (including Python tracebacks) is always saved to file.
        if not getattr(sys.stdout, "_is_tee", False):
            sys.stdout = DualWriter(sys.stdout, self.log_file)
        if not getattr(sys.stderr, "_is_tee", False):
            sys.stderr = DualWriter(sys.stderr, self.log_file)
        
        # Initialize CSV
        if not self.metrics_csv.exists():
            with open(self.metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "model", "epoch", "train_loss", "val_loss", "learning_rate", "gpu_memory_gb", "epoch_time_sec",
                    "precision_box", "recall_box", "map50_box", "map5095_box",
                    "precision_mask", "recall_mask", "map50_mask", "map5095_mask",
                    "mask_iou", "dice_f1", "boundary_iou", "left_right_mixup",
                    "best_mask_map50", "patience_counter", "is_best"
                ])

    def print_dataset_health(self, train_json_path, val_json_path):
        def get_stats(json_path):
            with open(json_path) as f:
                coco = json.load(f)
            imgs = len(coco.get("images", []))
            anns = len(coco.get("annotations", []))
            cats = len(coco.get("categories", []))
            
            img_with_anns = set(a["image_id"] for a in coco.get("annotations", []))
            empty_imgs = imgs - len(img_with_anns)
            return imgs, anns, cats, empty_imgs

        train_imgs, train_anns, cats, train_empty = get_stats(train_json_path)
        val_imgs, val_anns, _, val_empty = get_stats(val_json_path)

        print("\n========================================================")
        print(f"Dataset Health Statistics")
        print("========================================================")
        print(f"Classes: {cats}")
        print("Images:")
        print(f"  Train: {train_imgs} ({train_empty} empty)")
        print(f"  Val:   {val_imgs} ({val_empty} empty)")
        print("Annotations:")
        print(f"  Train: {train_anns}")
        print(f"  Val:   {val_anns}")
        print("========================================================\n")
        
        self.dataset_stats = {
            "train_imgs": train_imgs,
            "val_imgs": val_imgs,
            "cats": cats
        }

    def log_epoch(self, epoch, total_epochs, train_stats, val_metrics=None):
        """
        train_stats: dict with train_loss, val_loss, lr, gpu_mem_gb, epoch_time_sec, images_sec,
                     and optionally specific losses (e.g. classification_loss).
        val_metrics: dict with the evaluation metrics from UnifiedEvaluator. Can be None if skipped.
        Returns:
            should_stop (bool), is_best (bool)
        """
        print(f"\nEpoch {epoch}/{total_epochs}")
        print(f"|-- Train Loss: {train_stats.get('train_loss', 0):.4f}")
        # val_loss is float('nan') for inference-only validation (Mask2Former)
        _val_loss = train_stats.get('val_loss', float('nan'))
        _val_loss_str = f"{_val_loss:.4f}" if _val_loss == _val_loss else "N/A"  # nan != nan
        print(f"|-- Val Loss:   {_val_loss_str}")
        for k, v in train_stats.items():
            if k not in ['train_loss', 'val_loss', 'lr', 'gpu_mem_gb', 'epoch_time_sec', 'images_sec']:
                print(f"|-- {k}: {v:.4f}")
        print(f"|-- Learning Rate: {train_stats.get('lr', 0):.6f}")
        print(f"|-- GPU Memory: {train_stats.get('gpu_mem_gb', 0):.2f} GB")
        print(f"|-- Epoch Time: {train_stats.get('epoch_time_sec', 0):.1f}s")
        print(f"|__ Images/sec: {train_stats.get('images_sec', 0):.1f}")

        is_best = False
        should_stop = False
        
        if val_metrics:
            print("\nValidation Metrics:")
            print(f"|-- Box Precision:   {val_metrics.get('precision_box', 0):.4f}")
            print(f"|-- Box Recall:      {val_metrics.get('recall_box', 0):.4f}")
            print(f"|-- Box mAP50:       {val_metrics.get('map50_box', 0):.4f}")
            print(f"|-- Box mAP50-95:    {val_metrics.get('map5095_box', 0):.4f}")
            print(f"|-- Mask Precision:  {val_metrics.get('precision_mask', 0):.4f}")
            print(f"|-- Mask Recall:     {val_metrics.get('recall_mask', 0):.4f}")
            print(f"|-- Mask mAP50:      {val_metrics.get('map50_mask', 0):.4f}")
            print(f"|-- Mask mAP50-95:   {val_metrics.get('map5095_mask', 0):.4f}")
            print(f"|-- Mask IoU:        {val_metrics.get('mask_iou', 0):.4f}")
            print(f"|-- Dice/F1:         {val_metrics.get('dice_f1', 0):.4f}")
            print(f"|-- Boundary IoU:    {val_metrics.get('boundary_iou', 0):.4f}")
            print(f"|__ Left/Right Mixup: {val_metrics.get('left_right_mixup', 0):.4f}")
            
            if 'per_class' in val_metrics:
                print("\nPer-class Metrics (Mask):")
                print(f"{'CLASS':<22} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'mAP50-95':<10}")
                print("-" * 65)
                for c_name, c_metrics in val_metrics['per_class'].items():
                    print(f"{c_name:<22} {c_metrics.get('precision', 0):.4f}     {c_metrics.get('recall', 0):.4f}     {c_metrics.get('map50', 0):.4f}     {c_metrics.get('map5095', 0):.4f}")

            # Early stopping and best model tracking based on mask mAP50
            current_mask_map50 = val_metrics.get('map50_mask', 0)
            if current_mask_map50 > self.best_mask_map50:
                self.best_mask_map50 = current_mask_map50
                self.best_epoch = epoch
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                should_stop = True
                
            print(f"\nCurrent Mask mAP50: {current_mask_map50:.4f}")
            print(f"Best Mask mAP50:    {self.best_mask_map50:.4f} (Epoch {self.best_epoch})")
            print(f"Patience:           {self.patience_counter}/{self.patience}")

        with open(self.metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.model_name,
                epoch,
                train_stats.get('train_loss', ''),
                train_stats.get('val_loss', ''),
                train_stats.get('lr', ''),
                train_stats.get('gpu_mem_gb', ''),
                train_stats.get('epoch_time_sec', ''),
                val_metrics.get('precision_box', '') if val_metrics else '',
                val_metrics.get('recall_box', '') if val_metrics else '',
                val_metrics.get('map50_box', '') if val_metrics else '',
                val_metrics.get('map5095_box', '') if val_metrics else '',
                val_metrics.get('precision_mask', '') if val_metrics else '',
                val_metrics.get('recall_mask', '') if val_metrics else '',
                val_metrics.get('map50_mask', '') if val_metrics else '',
                val_metrics.get('map5095_mask', '') if val_metrics else '',
                val_metrics.get('mask_iou', '') if val_metrics else '',
                val_metrics.get('dice_f1', '') if val_metrics else '',
                val_metrics.get('boundary_iou', '') if val_metrics else '',
                val_metrics.get('left_right_mixup', '') if val_metrics else '',
                self.best_mask_map50,
                self.patience_counter,
                1 if is_best else 0
            ])
            
        json_path = self.output_dir / f"metrics_epoch_{epoch}.json"
        epoch_data = {
            "model_name": self.model_name,
            "epoch": epoch,
            "train_stats": train_stats,
            "val_metrics": val_metrics,
            "best_mask_map50": self.best_mask_map50,
            "is_best": is_best
        }
        with open(json_path, "w") as f:
            json.dump(epoch_data, f, indent=4)

        if is_best:
            best_json_path = self.output_dir / "best_metrics.json"
            with open(best_json_path, "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "best_epoch": self.best_epoch,
                    "best_mask_map50": self.best_mask_map50,
                    "train_stats": train_stats,
                    "val_metrics": val_metrics
                }, f, indent=4)
            
        self.last_val_metrics = val_metrics if val_metrics else getattr(self, 'last_val_metrics', None)
        self.last_train_stats = train_stats
        self.last_epoch = epoch

        return should_stop, is_best

    def print_final_summary(self):
        total_time_min = (time.time() - self.start_time) / 60
        
        vm = self.last_val_metrics or {}
        ts = self.last_train_stats or {}
        
        print("\n========================================================")
        print(f"MODEL: {self.model_name.upper()}")
        print("========================================================")
        print("\nDataset")
        print(f"  Train images:          {self.dataset_stats.get('train_imgs', 0)}")
        print(f"  Validation images:     {self.dataset_stats.get('val_imgs', 0)}")
        print(f"  Classes:               {self.dataset_stats.get('cats', 0)}")
        
        print("\nTraining")
        print(f"  Epochs:                {self.last_epoch}")
        print(f"  Best epoch:            {self.best_epoch}")
        print(f"  Total training time:   {total_time_min:.1f} min")
        print(f"  Final train loss:      {ts.get('train_loss', 0):.4f}")
        _vl = ts.get('val_loss', float('nan'))
        _vl_str = f"{_vl:.4f}" if _vl == _vl else "N/A"
        print(f"  Final validation loss: {_vl_str}")
        
        print("\nDetection")
        print(f"  Precision:             {vm.get('precision_box', 0):.4f}")
        print(f"  Recall:                {vm.get('recall_box', 0):.4f}")
        print(f"  mAP50:                 {vm.get('map50_box', 0):.4f}")
        print(f"  mAP50-95:              {vm.get('map5095_box', 0):.4f}")

        print("\nSegmentation")
        print(f"  Mask Precision:        {vm.get('precision_mask', 0):.4f}")
        print(f"  Mask Recall:           {vm.get('recall_mask', 0):.4f}")
        print(f"  Mask mAP50:            {vm.get('map50_mask', 0):.4f}")
        print(f"  Mask mAP50-95:         {vm.get('map5095_mask', 0):.4f}")
        print(f"  Mask IoU:              {vm.get('mask_iou', 0):.4f}")
        print(f"  Boundary IoU:          {vm.get('boundary_iou', 0):.4f}")
        print(f"  Dice/F1:               {vm.get('dice_f1', 0):.4f}")

        print("\nCar-part identity")
        print(f"  Left/right mixup:      {vm.get('left_right_mixup', 0):.4f}")
        
        print("\nBest checkpoint")
        print(f"  Epoch:                 {self.best_epoch}")
        print(f"  Mask mAP50:            {self.best_mask_map50:.4f}")
        print("========================================================\n")
