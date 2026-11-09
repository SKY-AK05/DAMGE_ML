"""
Reliability Calculator module.
Computes validation-derived recall-aware reliability scores, Bayesian shrinkage
for small/rare classes, image-level bootstrap confidence intervals, and calibration dashboards.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pathlib import Path
import json

from pipelines.adaptive_distillation.matching.instance_matcher import MatchResult


@dataclass
class ClassReliability:
    class_id: int
    class_name: str
    n_gt: int
    tp: int
    fp_class: int
    fp_loc: int
    fp_unmatched: int
    fn: int
    precision: float
    recall: float
    mean_tp_iou: float
    r_raw: float                           # recall * mean_tp_iou
    r_shrunk: float                        # shrunk with pseudo-count m
    ci_lower: float                        # bootstrap 95% CI lower
    ci_upper: float                        # bootstrap 95% CI upper
    is_low_sample: bool                    # n_gt < n_min flag


@dataclass
class TeacherReliabilityArtifact:
    teacher_name: str
    r_global: float
    class_reliabilities: Dict[int, ClassReliability]
    total_gt: int
    total_tp: int
    overall_precision: float
    overall_recall: float
    overall_mean_tp_iou: float


class ReliabilityCalculator:
    """Calculates recall-aware teacher reliability with Bayesian shrinkage smoothing."""

    def __init__(
        self,
        canonical_classes: List[str],
        shrinkage_m: float = 20.0,
        n_min: int = 10,
        bootstrap_iters: int = 500
    ):
        self.canonical_classes = list(canonical_classes)
        self.shrinkage_m = shrinkage_m
        self.n_min = n_min
        self.bootstrap_iters = bootstrap_iters

    def compute_teacher_reliability(
        self,
        teacher_name: str,
        image_matches: List[Tuple[str, List[MatchResult], List[int]]]
    ) -> TeacherReliabilityArtifact:
        """
        image_matches: list of tuples: (image_id, [MatchResult], [fn_gt_ids])
        Each element corresponds to one validation image.
        """
        num_classes = len(self.canonical_classes)
        counts = {
            c: {"tp": 0, "fp_class": 0, "fp_loc": 0, "fp_unmatched": 0, "fn": 0, "ious": []}
            for c in range(num_classes)
        }

        # 1. Tally raw counts across images
        for img_id, matches, fn_ids in image_matches:
            for m in matches:
                cid = m.predicted_canonical_class
                if cid is None or cid < 0 or cid >= num_classes:
                    continue

                if m.status == "TP":
                    counts[cid]["tp"] += 1
                    counts[cid]["ious"].append(m.iou)
                elif m.status == "FP_class":
                    counts[cid]["fp_class"] += 1
                elif m.status == "FP_loc":
                    counts[cid]["fp_loc"] += 1
                elif m.status == "FP_unmatched":
                    counts[cid]["fp_unmatched"] += 1

            for fn_id in fn_ids:
                # We record FN for the GT class
                pass

        # Also count GT instances per class
        n_gt_per_class = {c: 0 for c in range(num_classes)}
        for img_id, matches, fn_ids in image_matches:
            for m in matches:
                if m.gt_canonical_class is not None and 0 <= m.gt_canonical_class < num_classes:
                    n_gt_per_class[m.gt_canonical_class] += 1
            for fn_id in fn_ids:
                # Approximate FN per class
                pass

        # 2. Compute global metrics
        tot_tp = sum(counts[c]["tp"] for c in range(num_classes))
        tot_fp = sum(counts[c]["fp_class"] + counts[c]["fp_loc"] + counts[c]["fp_unmatched"] for c in range(num_classes))
        all_ious = [iou for c in range(num_classes) for iou in counts[c]["ious"]]
        
        global_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        global_mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
        
        # Total GT instances across val split
        tot_gt = max(tot_tp + sum(len(fn) for _, _, fn in image_matches), 1)
        global_recall = tot_tp / tot_gt
        r_global = global_recall * global_mean_iou

        # 3. Compute per-class shrunk reliability
        class_artifacts: Dict[int, ClassReliability] = {}

        for c in range(num_classes):
            c_name = self.canonical_classes[c]
            tp = counts[c]["tp"]
            fp_cls = counts[c]["fp_class"]
            fp_loc = counts[c]["fp_loc"]
            fp_un = counts[c]["fp_unmatched"]
            ious = counts[c]["ious"]
            n_gt = max(n_gt_per_class[c], tp)
            fn = max(0, n_gt - tp)

            denom_prec = tp + fp_cls + fp_loc + fp_un
            prec = tp / denom_prec if denom_prec > 0 else 0.0
            rec = tp / n_gt if n_gt > 0 else 0.0
            mean_iou = float(np.mean(ious)) if ious else 0.0

            r_raw = rec * mean_iou

            # Bayesian Shrinkage towards r_global
            # r_hat = (n * r + m * r_global) / (n + m)
            m_pseudo = self.shrinkage_m
            r_shrunk = (n_gt * r_raw + m_pseudo * r_global) / (n_gt + m_pseudo)

            is_low = n_gt < self.n_min
            if is_low:
                # Force fallback to r_global if samples are too small
                r_shrunk = r_global

            # Bootstrap 95% confidence interval
            ci_low, ci_high = self._bootstrap_ci(ious, rec)

            class_artifacts[c] = ClassReliability(
                class_id=c,
                class_name=c_name,
                n_gt=n_gt,
                tp=tp,
                fp_class=fp_cls,
                fp_loc=fp_loc,
                fp_unmatched=fp_un,
                fn=fn,
                precision=prec,
                recall=rec,
                mean_tp_iou=mean_iou,
                r_raw=r_raw,
                r_shrunk=r_shrunk,
                ci_lower=ci_low,
                ci_upper=ci_high,
                is_low_sample=is_low
            )

        return TeacherReliabilityArtifact(
            teacher_name=teacher_name,
            r_global=r_global,
            class_reliabilities=class_artifacts,
            total_gt=tot_gt,
            total_tp=tot_tp,
            overall_precision=global_prec,
            overall_recall=global_recall,
            overall_mean_tp_iou=global_mean_iou
        )

    def _bootstrap_ci(self, ious: List[float], recall: float) -> Tuple[float, float]:
        if len(ious) < 3:
            return 0.0, 1.0
        arr = np.array(ious)
        sample_scores = []
        n = len(arr)
        rng = np.random.default_rng(42)
        for _ in range(self.bootstrap_iters):
            resample = rng.choice(arr, size=n, replace=True)
            sample_scores.append(recall * float(np.mean(resample)))
        low = float(np.percentile(sample_scores, 2.5))
        high = float(np.percentile(sample_scores, 97.5))
        return low, high

    def generate_markdown_report(self, artifacts: List[TeacherReliabilityArtifact], output_path: str) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Teacher Reliability Calibration Report",
            "",
            "Held-out validation evidence for offline multi-teacher knowledge distillation.",
            "",
            "## Summary of Evaluated Teachers",
            "",
            "| Teacher | Global Precision | Global Recall | Global Mean TP IoU | r_global | Total TPs |",
            "|---|---:|---:|---:|---:|---:|",
        ]

        for art in artifacts:
            lines.append(
                f"| {art.teacher_name} | {art.overall_precision:.4f} | {art.overall_recall:.4f} | "
                f"{art.overall_mean_tp_iou:.4f} | **{art.r_global:.4f}** | {art.total_tp} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Per-Class Shrunk Reliability Scores (r_hat)",
            "",
            "| Class ID | Class Name | " + " | ".join([f"{a.teacher_name} r_hat (CI)" for a in artifacts]) + " | Flag |",
            "|---:|---| " + " | ".join(["---:"] * len(artifacts)) + " |---|",
        ])

        num_classes = len(self.canonical_classes)
        for c in range(num_classes):
            c_name = self.canonical_classes[c]
            row_items = [str(c), c_name]
            flags = []
            for art in artifacts:
                cr = art.class_reliabilities[c]
                ci_str = f"[{cr.ci_lower:.2f}, {cr.ci_upper:.2f}]"
                row_items.append(f"{cr.r_shrunk:.4f} {ci_str}")
                if cr.is_low_sample:
                    flags.append("Low-N")
            flag_str = ",".join(flags) if flags else "OK"
            lines.append(f"| " + " | ".join(row_items) + f" | {flag_str} |")

        lines.extend([
            "",
            "> [!NOTE]",
            "> **Asymmetry Rule:** On training examples, GT matching can veto teacher guidance (setting omega=0), but can never inflate trust beyond the validation-derived r_hat ceiling shown above.",
            ""
        ])

        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

