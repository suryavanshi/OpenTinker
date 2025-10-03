"""Training client implementing the synchronous week 1 API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..core import engine
from ..core.types import AdamParams, Datum
from ..utils.futures import ImmediateFuture
from ..utils.tokenizer import SimpleTokenizer
from .sampling_client import SamplingClient


@dataclass
class ForwardBackwardResponse:
    loss: float


class TrainingClient:
    def __init__(self, *, model: torch.nn.Module, tokenizer: SimpleTokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device("cpu")
        self._model.to(self._device)
        self._optimiser: torch.optim.Optimizer | None = None

    @property
    def tokenizer(self) -> SimpleTokenizer:
        return self._tokenizer

    def forward_backward(self, batch: Sequence[Datum], loss_fn: str = "cross_entropy") -> ImmediateFuture[ForwardBackwardResponse]:
        result = engine.forward_backward(self._model, batch, loss_fn, device=self._device)
        return ImmediateFuture(ForwardBackwardResponse(loss=result.loss))

    def optim_step(self, params: AdamParams) -> ImmediateFuture[dict]:
        self._optimiser = engine.ensure_adam(self._model, self._optimiser, params)
        metrics = engine.optim_step(self._model, self._optimiser)
        return ImmediateFuture(metrics)

    def save_weights_and_get_sampling_client(self, name: str) -> SamplingClient:
        model_copy = engine.SimpleLanguageModel(self._tokenizer.vocab_size)
        model_copy.load_state_dict(self._model.state_dict())
        return SamplingClient(model=model_copy, tokenizer=self._tokenizer)


__all__ = ["TrainingClient", "ForwardBackwardResponse"]
