"""
Checkpointing utilities for saving and restoring training states.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class CheckpointManager:
    """Saves and restores model and optimizer states safely."""

    @staticmethod
    def save(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        path: Path | str,
        is_best: bool = False
    ) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }
        torch.save(checkpoint, p)

        if is_best:
            best_p = p.parent / "best_model.pt"
            torch.save(checkpoint, best_p)

    @staticmethod
    def load(
        path: Path | str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {p}")
        data = torch.load(p, map_location=device)
        model.load_state_dict(data["model_state_dict"])
        if optimizer and "optimizer_state_dict" in data:
            optimizer.load_state_dict(data["optimizer_state_dict"])
        return data

