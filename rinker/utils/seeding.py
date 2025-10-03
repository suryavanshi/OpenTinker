"""Utility helpers to seed Python and PyTorch RNGs."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - GPU not present in CI
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


__all__ = ["seed_everything"]
