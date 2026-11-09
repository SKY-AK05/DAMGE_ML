"""
Trust Weighting module.
Implements:
1. Absolute trust gate g[c]
2. Relative teacher shares s[k,c] (supporting arbitrary N >= 1 teachers)
3. Instance-level GT veto and quality weighting omega[k,i,j]
4. Experimental controls: shuffled, uniform_in_gate, inverse
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from pipelines.adaptive_distillation.config import TrustWeightingConfig
from pipelines.adaptive_distillation.matching.reliability_calculator import TeacherReliabilityArtifact


@dataclass
class InstanceWeightResult:
    teacher_name: str
    class_id: int
    gate_g: float
    share_s: float
    instance_omega: float
    final_weight_w: float
    is_vetoed: bool


class TrustWeighting:
    """Computes decomposition w = g[c] * s[k,c] * omega[k,i,j] under the strict asymmetry rule."""

    def __init__(self, config: TrustWeightingConfig):
        self.config = config

    def compute_gate(
        self,
        class_id: int,
        r_hat_teachers: Dict[str, float],
        r_ref: float,
        r_max: float = 1.0
    ) -> float:
        """
        Absolute trust gate: g[c] in [0, 1].
        If no teacher beats r_ref + margin, g[c] = 0.
        """
        if not r_hat_teachers:
            return 0.0

        r_best = max(r_hat_teachers.values())
        threshold = r_ref + self.config.gate_margin

        if r_best < threshold:
            return 0.0

        if self.config.gate_type == "hard":
            return 1.0

        # Smooth gate
        denom = max(r_max - r_ref, 1e-6)
        raw_g = (r_best - r_ref) / denom
        return float(np.clip(raw_g, self.config.g_min, 1.0))

    def compute_shares(
        self,
        class_id: int,
        r_hat_teachers: Dict[str, float],
        r_ref: float
    ) -> Dict[str, float]:
        """
        Relative teacher shares s[k, c] summing to 1 over usable teachers.
        Supports arbitrary N >= 1 teachers.
        """
        teachers = list(r_hat_teachers.keys())
        n_teachers = len(teachers)

        if n_teachers == 0:
            return {}
        if n_teachers == 1:
            return {teachers[0]: 1.0}

        # Check usability per teacher
        usable = {t: r_hat_teachers[t] >= r_ref for t in teachers}
        usable_teachers = [t for t in teachers if usable[t]]

        if not usable_teachers:
            # Fall back to uniform among all or 0
            return {t: 0.0 for t in teachers}

        if len(usable_teachers) == 1:
            res = {t: 0.0 for t in teachers}
            res[usable_teachers[0]] = 1.0
            return res

        # Control C2: Uniform within gate
        if self.config.mode == "uniform_in_gate":
            val = 1.0 / len(usable_teachers)
            return {t: (val if usable[t] else 0.0) for t in teachers}

        # Control C3: Inverse reliability (weight worse teacher more)
        if self.config.mode == "inverse":
            inv_scores = np.array([1.0 / max(r_hat_teachers[t], 1e-4) if usable[t] else 0.0 for t in teachers])
            tot = np.sum(inv_scores)
            shares = inv_scores / tot if tot > 0 else np.zeros_like(inv_scores)
            return {t: float(shares[i]) for i, t in enumerate(teachers)}

        # Softmax or proportional share
        scores = np.array([r_hat_teachers[t] if usable[t] else -1e9 for t in teachers])

        if self.config.share_type == "softmax":
            temp = max(self.config.temperature, 1e-4)
            exp_scores = np.exp((scores - np.max(scores)) / temp)
            for i, t in enumerate(teachers):
                if not usable[t]:
                    exp_scores[i] = 0.0
            tot = np.sum(exp_scores)
            raw_shares = exp_scores / tot if tot > 0 else np.zeros_like(exp_scores)
        else:
            # Proportional
            pos_scores = np.array([max(0.0, r_hat_teachers[t]) if usable[t] else 0.0 for t in teachers])
            tot = np.sum(pos_scores)
            raw_shares = pos_scores / tot if tot > 0 else np.zeros_like(pos_scores)

        # Apply floor s_min to preserve teacher diversity when both are usable
        s_min = self.config.s_min
        usable_indices = [i for i, t in enumerate(teachers) if usable[t]]
        if len(usable_indices) > 1 and s_min > 0.0:
            for idx in usable_indices:
                raw_shares[idx] = max(raw_shares[idx], s_min)
            tot = np.sum(raw_shares)
            if tot > 0:
                raw_shares /= tot

        return {t: float(raw_shares[i]) for i, t in enumerate(teachers)}

    def compute_instance_omega(
        self,
        status: str,
        iou: float,
        tau_low: float = 0.10,
        confidence: Optional[float] = None
    ) -> float:
        """
        Instance-level weight factor omega in [0, 1].
        GT can veto (setting omega=0), but can never boost above 1.0 (asymmetry rule).
        """
        if status in ("FP_class", "FP_loc", "FP_unmatched"):
            return 0.0

        if status == "TP":
            # Better overlap -> higher quality score in [omega_min, 1.0]
            omega_min = self.config.omega_min
            denom = max(1.0 - tau_low, 1e-4)
            raw_omega = (iou - tau_low) / denom
            return float(np.clip(raw_omega, omega_min, 1.0))

        if status == "unlabeled":
            # For unlabeled images (Phase C), use confidence
            return float(confidence) if confidence is not None else 0.5

        return 0.0

    def compute_full_weight(
        self,
        teacher_name: str,
        class_id: int,
        status: str,
        iou: float,
        r_hat_teachers: Dict[str, float],
        r_ref: float,
        tau_low: float = 0.10
    ) -> InstanceWeightResult:
        """Computes complete decomposed weight w[k, i, j] = g[c] * s[k, c] * omega[k, i, j]."""
        g = self.compute_gate(class_id, r_hat_teachers, r_ref)
        shares = self.compute_shares(class_id, r_hat_teachers, r_ref)
        s = shares.get(teacher_name, 0.0)

        omega = self.compute_instance_omega(status, iou, tau_low)
        is_vetoed = (omega == 0.0)

        w = g * s * omega

        return InstanceWeightResult(
            teacher_name=teacher_name,
            class_id=class_id,
            gate_g=g,
            share_s=s,
            instance_omega=omega,
            final_weight_w=w,
            is_vetoed=is_vetoed
        )

