"""Training utilities for the local single GPU/CPU backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import losses
from .types import AdamParams, Datum


LOSS_REGISTRY: Dict[str, Callable[..., Dict[str, torch.Tensor]]] = {
    "cross_entropy": losses.cross_entropy,
    "importance_sampling": losses.importance_sampling,
    "ppo": losses.ppo,
}


class SimpleLanguageModel(nn.Module):
    """A minimal autoregressive model suitable for quick CPU experiments."""

    def __init__(self, vocab_size: int, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(input_ids)
        outputs, _ = self.rnn(embeds)
        outputs = self.ln(outputs)
        logits = self.head(outputs)
        return logits


@dataclass
class ForwardBackwardResult:
    loss: float
    details: Dict[str, torch.Tensor]


def forward_backward(
    model: nn.Module,
    batch: Sequence[Datum],
    loss_name: str,
    device: torch.device,
) -> ForwardBackwardResult:
    """Runs a forward/backward pass for the provided batch."""

    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{loss_name}'")

    model.train()
    inputs = torch.cat([datum.model_input.to_batch(device) for datum in batch], dim=0)
    targets = torch.cat([datum.loss_fn_inputs["targets"].unsqueeze(0).to(device) for datum in batch], dim=0)
    weights = None
    if "weights" in batch[0].loss_fn_inputs:
        weights = torch.cat([datum.loss_fn_inputs["weights"].unsqueeze(0).to(device) for datum in batch], dim=0)

    logits = model(inputs)
    loss_fn = LOSS_REGISTRY[loss_name]
    result = loss_fn(logits, targets=targets, weights=weights)
    loss = result["loss"]
    loss.backward()
    return ForwardBackwardResult(loss=float(loss.detach().cpu()), details=result)


def optim_step(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
) -> Dict[str, float]:
    """Applies an optimisation step and zeros gradients."""

    optimiser.step()
    optimiser.zero_grad(set_to_none=True)
    return {"grad_norm": _total_grad_norm(model)}


def ensure_adam(
    model: nn.Module,
    optimiser: torch.optim.Optimizer | None,
    params: AdamParams,
) -> torch.optim.Optimizer:
    if optimiser is not None:
        return optimiser
    betas = tuple(params.betas)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=params.lr,
        betas=betas,  # type: ignore[arg-type]
        eps=params.eps,
        weight_decay=params.weight_decay,
    )
    return optimiser


def _total_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += param.grad.detach().data.norm(2).item() ** 2
    return total ** 0.5


__all__ = [
    "SimpleLanguageModel",
    "ForwardBackwardResult",
    "forward_backward",
    "optim_step",
    "ensure_adam",
]
