"""Core data structures for the Rinker training and sampling API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional

import torch


TensorDict = MutableMapping[str, torch.Tensor]


@dataclass
class ModelInput:
    """Represents the tokenised input to the model.

    The public API mirrors Tinker where model inputs are provided as chunks of
    token ids (to allow prefix packing in the future). For the week 1
    implementation we simply expect a single chunk, but the abstraction keeps
    the surface compatible with the longer term design.
    """

    token_chunks: List[torch.Tensor]
    metadata: Mapping[str, object] = field(default_factory=dict)
    attachments: Mapping[str, Any] | None = None

    def to_batch(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Stacks the token chunks into a single tensor for model consumption."""

        if len(self.token_chunks) != 1:
            raise ValueError("Week 1 implementation expects exactly one chunk per datum")
        tokens = self.token_chunks[0]
        if device is not None:
            tokens = tokens.to(device)
        return tokens.unsqueeze(0)


@dataclass
class Datum:
    """A training datum consisting of model input and loss specific tensors."""

    model_input: ModelInput
    loss_fn_inputs: TensorDict
    policy_version: int | None = None


@dataclass
class AdamParams:
    """Hyper-parameters for the Adam optimiser."""

    lr: float = 1e-3
    betas: Iterable[float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class SamplingParams:
    """Parameters controlling autoregressive sampling."""

    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None


__all__ = [
    "TensorDict",
    "ModelInput",
    "Datum",
    "AdamParams",
    "SamplingParams",
]
