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


def importance_sampling(
    logits: torch.Tensor,
    *,
    target_tokens: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Token-level importance sampling / REINFORCE objective.

    Args:
        logits: Tensor of shape ``(batch, seq_len, vocab)``.
        target_tokens: Tokens whose log-probabilities are evaluated. Shape
            ``(batch, seq_len)``.
        sampling_logprobs: Log-probabilities produced by the behaviour policy
            that generated ``target_tokens``. Same shape as ``target_tokens``.
        advantages: Pre-computed advantages per token. Same shape as
            ``target_tokens``.

    Returns:
        Dictionary containing the scalar loss used for ``backward`` and
        diagnostics such as the per-token log-probabilities and ratios.
    """

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
    sampling_logprobs = sampling_logprobs.to(target_log_probs.dtype)
    advantages = advantages.to(target_log_probs.dtype)

    log_ratio = target_log_probs - sampling_logprobs
    ratio = torch.exp(log_ratio)
    loss = -(ratio * advantages).sum()

    return {
        "loss": loss,
        "target_log_probs": target_log_probs.detach(),
        "sampling_logprobs": sampling_logprobs.detach(),
        "advantages": advantages.detach(),
        "ratio": ratio.detach(),
    }


def ppo(
    logits: torch.Tensor,
    *,
    target_tokens: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Token-level PPO clipped objective.

    Args:
        logits: Tensor of shape ``(batch, seq_len, vocab)``.
        target_tokens: Tokens whose log-probabilities are evaluated. Shape
            ``(batch, seq_len)``.
        sampling_logprobs: Behaviour policy log-probabilities. Same shape as
            ``target_tokens``.
        advantages: Advantages corresponding to each token. Same shape as
            ``target_tokens``.
        clip_epsilon: PPO clipping parameter (default ``0.2``).

    Returns:
        Dictionary containing the scalar loss and diagnostics including
        per-token ratios and clipping statistics.
    """

    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
    sampling_logprobs = sampling_logprobs.to(target_log_probs.dtype)
    advantages = advantages.to(target_log_probs.dtype)

    log_ratio = target_log_probs - sampling_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    objective = torch.minimum(unclipped_objective, clipped_objective)
    loss = -objective.sum()

    clip_mask = torch.logical_or(ratio > (1.0 + clip_epsilon), ratio < (1.0 - clip_epsilon))
    clip_fraction = clip_mask.float().mean()

    return {
        "loss": loss,
        "target_log_probs": target_log_probs.detach(),
        "sampling_logprobs": sampling_logprobs.detach(),
        "advantages": advantages.detach(),
        "ratio": ratio.detach(),
        "clipped_ratio": clipped_ratio.detach(),
        "clip_fraction": clip_fraction.detach(),
    }


__all__ = ["cross_entropy", "importance_sampling", "ppo"]
