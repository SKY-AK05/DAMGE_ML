"""
Experiment Comparison and Hypothesis Verification module.
Tracks and reports metrics across the complete experimental matrix (E0-E6)
and control conditions (C1-C5) according to pre-registered hypotheses.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class ExperimentResult:
    exp_id: str                            # "E0", "E1", "E2", ...
    name: str                              # "GT-only Student C", "GT + A + B Adaptive", etc.
    strategy: str                          # description of supervision
    mask_map50: float
    mask_map50_95: float
    box_map50: float
    box_map50_95: float
    delta_vs_e2: float                     # Improvement relative to GT-only baseline
    ci_95: List[float]                     # [ci_low, ci_high]
    seeds_evaluated: int
    peak_vram_gb: float
    per_class_map50: Dict[str, float]


class ExperimentComparison:
    """Aggregates experiment metrics and validates pre-registered hypotheses."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, ExperimentResult] = {}

    def add_result(self, res: ExperimentResult) -> None:
        self.results[res.exp_id] = res

    def evaluate_hypotheses(self) -> Dict[str, Dict[str, Any]]:
        """Evaluates hypotheses H1 to H5 defined in design document."""
        hyps = {}

        e2 = self.results.get("E2")
        e3 = self.results.get("E3")
        e4 = self.results.get("E4")
        e5 = self.results.get("E5")
        e6 = self.results.get("E6")
        c1 = self.results.get("C1")
        c2 = self.results.get("C2")

        # H1: Single teacher beats GT-only
        if e2 and (e3 or e4):
            best_single = max(e3.mask_map50 if e3 else 0, e4.mask_map50 if e4 else 0)
            passed = best_single > e2.mask_map50
            hyps["H1_Single_Teacher_Helps"] = {
                "supported": passed,
                "delta": best_single - e2.mask_map50,
                "conclusion": "Single teacher distillation outperforms GT-only scratch baseline" if passed else "No single teacher benefit observed"
            }

        # H2: Two teachers beat either alone
        if (e3 or e4) and e5:
            max_single = max(e3.mask_map50 if e3 else 0, e4.mask_map50 if e4 else 0)
            passed = e5.mask_map50 > max_single
            hyps["H2_Two_Teachers_Beat_Single"] = {
                "supported": passed,
                "delta": e5.mask_map50 - max_single,
                "conclusion": "Combining teachers beats the best single teacher" if passed else "Multi-teacher does not outperform best single teacher"
            }

        # H3: Adaptive weights beat fixed weights
        if e5 and e6:
            passed = e6.mask_map50 > e5.mask_map50
            hyps["H3_Adaptive_Beats_Fixed"] = {
                "supported": passed,
                "delta": e6.mask_map50 - e5.mask_map50,
                "conclusion": "Adaptive reliability weighting outperforms fixed uniform weighting" if passed else "Adaptive weighting adds no benefit over fixed weighting"
            }

        # H4: Reliability signal itself matters (beats shuffled and uniform controls)
        if e6 and c1:
            passed = e6.mask_map50 > c1.mask_map50
            hyps["H4_Reliability_Signal_Matters"] = {
                "supported": passed,
                "delta": e6.mask_map50 - c1.mask_map50,
                "conclusion": "Real reliability outperforms shuffled weights (rules out noise regularization)" if passed else "Shuffled weights achieve parity (signal not verified)"
            }

        return hyps

    def generate_report(self, output_path: Optional[str] = None) -> str:
        out_p = Path(output_path) if output_path else self.output_dir / "experiment_comparison_report.md"
        out_p.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Adaptive Multi-Teacher Distillation: Experiment Comparison Report",
            "",
            "## Summary of Experimental Matrix",
            "",
            "| ID | Strategy | Mask mAP50 | Mask mAP50-95 | Box mAP50 | Δ vs E2 [95% CI] | Peak VRAM |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]

        for exp_id in sorted(self.results.keys()):
            r = self.results[exp_id]
            ci_str = f"[{r.ci_95[0]:+.3f}, {r.ci_95[1]:+.3f}]" if r.ci_95 else "N/A"
            delta_str = f"{r.delta_vs_e2:+.4f}" if exp_id != "E2" else "0.0000 (Ref)"
            lines.append(
                f"| **{r.exp_id}** | {r.name} | {r.mask_map50:.4f} | {r.mask_map50_95:.4f} | "
                f"{r.box_map50:.4f} | {delta_str} {ci_str} | {r.peak_vram_gb:.1f} GB |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Pre-Registered Hypotheses Evaluation",
            "",
        ])

        hyps = self.evaluate_hypotheses()
        if not hyps:
            lines.append("*Awaiting completion of all matrix runs to evaluate full hypothesis set.*")
        else:
            for hid, hdata in hyps.items():
                status = "SUPPORTED" if hdata["supported"] else "REFUTED"
                lines.append(f"- **{hid}**: **{status}** (Δ = {hdata.get('delta', 0.0):+.4f}) — *{hdata['conclusion']}*")

        lines.append("")
        content = "\n".join(lines)

        with open(out_p, "w", encoding="utf-8") as f:
            f.write(content)

        # Also save JSON machine-readable
        json_p = out_p.with_suffix(".json")
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.results.items()}, f, indent=2)

        return content

