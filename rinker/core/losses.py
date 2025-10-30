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


def generalized_jsd(
    logits: torch.Tensor,
    *,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "sum",
) -> Dict[str, torch.Tensor]:
    """Generalised Jensen-Shannon divergence between student and teacher logits.

    Args:
        logits: Student logits of shape ``(batch, seq_len, vocab)``.
        teacher_logits: Teacher logits of the same shape as ``logits``.
        loss_mask: Optional boolean mask selecting the token positions that
            contribute to the loss. Shape ``(batch, seq_len)``.
        beta: Interpolation coefficient between student and teacher
            distributions. Values of ``0`` or ``1`` reduce to KL divergences
            against the teacher or student respectively.
        temperature: Softmax temperature applied to both distributions.
        reduction: Specifies the reduction to apply. Supported values:
            ``"sum"`` (default), ``"mean"``, ``"batchmean"``.

    Returns:
        Dictionary containing the scalar loss along with detached
        student/teacher log-probabilities for diagnostics.
    """

    if logits.shape != teacher_logits.shape:
        raise ValueError("teacher_logits must match logits shape")

    student_logits = logits / temperature
    teacher_logits = teacher_logits.to(logits.dtype) / temperature

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta <= 0.0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta >= 1.0:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_tensor = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        log_beta = torch.log(beta_tensor)
        log_one_minus_beta = torch.log1p(-beta_tensor)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                (
                    student_log_probs + log_one_minus_beta,
                    teacher_log_probs + log_beta,
                )
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_tensor * kl_teacher + (1 - beta_tensor) * kl_student

    mask = None
    if loss_mask is not None:
        if loss_mask.shape != student_log_probs.shape[:2]:
            raise ValueError("loss_mask must have shape (batch, seq_len)")
        mask = loss_mask.to(dtype=torch.bool, device=jsd.device)
        jsd = jsd[mask]

    if jsd.numel() == 0:
        loss = torch.tensor(0.0, dtype=student_log_probs.dtype, device=student_log_probs.device)
    else:
        if reduction == "sum":
            loss = jsd.sum()
        elif reduction == "mean":
            loss = jsd.mean()
        elif reduction == "batchmean":
            if mask is not None:
                denom = mask.sum().clamp(min=1).to(jsd.dtype)
            else:
                denom = torch.tensor(jsd.numel() / jsd.shape[-1], dtype=jsd.dtype, device=jsd.device)
                denom = denom.clamp(min=1)
            loss = jsd.sum() / denom
        else:
            raise ValueError(f"Unsupported reduction '{reduction}'")

    return {
        "loss": loss,
        "student_log_probs": student_log_probs.detach(),
        "teacher_log_probs": teacher_log_probs.detach(),
        "loss_mask": loss_mask.detach() if loss_mask is not None else None,
    }


__all__ = ["cross_entropy", "importance_sampling", "ppo", "generalized_jsd"]
