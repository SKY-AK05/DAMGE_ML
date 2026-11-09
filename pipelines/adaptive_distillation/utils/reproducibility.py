"""
Reproducibility and Environment Logging module.
Fixes random seeds across libraries and captures git commit and system environment hashes.
"""

import os
import random
import subprocess
from typing import Dict, Any
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds deterministically across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def capture_environment_metadata() -> Dict[str, Any]:
    """Captures runtime environment, software versions, and git status."""
    vram_gb = 0.0
    if torch.cuda.is_available():
        try:
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        except Exception:
            pass

    meta = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_memory_gb": vram_gb,
        "git_commit": "unknown",
        "git_dirty": False
    }

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        meta["git_commit"] = commit
        meta["git_dirty"] = (len(status) > 0)
    except Exception:
        pass

    return meta

