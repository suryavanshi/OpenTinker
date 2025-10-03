"""Loss functions used by the training engine."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

def cross_entropy(logits: torch.Tensor, *, targets: torch.Tensor, weights: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    """Token-level cross entropy with sum reduction.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab).
        targets: Tensor of shape (batch, seq_len) with target token ids.
        weights: Optional tensor of shape (batch, seq_len) providing per-token
            weights. When omitted, every target contributes equally.
    Returns:
        Dictionary with the scalar loss (for backward) and per-token log-probs.
    """

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if weights is None:
        weights = torch.ones_like(target_log_probs, dtype=log_probs.dtype)
    weights = weights.to(log_probs.dtype)
    loss = -(target_log_probs * weights).sum()
    return {
        "loss": loss,
        "target_log_probs": target_log_probs.detach(),
        "weights": weights.detach(),
    }


def importance_sampling(*args, **kwargs):  # pragma: no cover - stub for week 1
    raise NotImplementedError("Importance sampling loss will arrive in week 2")


def ppo(*args, **kwargs):  # pragma: no cover - stub for week 1
    raise NotImplementedError("PPO loss will arrive in week 2")


__all__ = ["cross_entropy", "importance_sampling", "ppo"]
