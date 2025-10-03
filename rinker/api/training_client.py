"""Training client backed by Ray actors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

from ..core import engine
from ..core.types import AdamParams, Datum
from ..ray_runtime import RayRuntime
from ..ray_runtime.actors import ForwardBackwardPayload
from ..utils.futures import RayFuture
from ..utils.tokenizer import SimpleTokenizer
from .sampling_client import SamplingClient


@dataclass
class ForwardBackwardResponse:
    loss: float
    metrics: dict[str, float]
    loss_fn_outputs: dict[str, torch.Tensor]


class TrainingClient:
    def __init__(self, *, runtime: RayRuntime) -> None:
        self._runtime = runtime
        self._tokenizer = runtime.tokenizer

    @property
    def tokenizer(self) -> SimpleTokenizer:
        return self._tokenizer

    def forward_backward(
        self,
        batch: Sequence[Datum],
        loss_fn: str = "cross_entropy",
    ) -> RayFuture[ForwardBackwardResponse]:
        payload_future = self._runtime.forward_backward(batch, loss_fn)
        return RayFuture(payload_future.object_ref, self._build_response)

    def forward_backward_custom(
        self,
        batch: Sequence[Datum],
        loss_fn: Callable[[Sequence[Datum], torch.Tensor], engine.CustomLossOutputs],
    ) -> RayFuture[ForwardBackwardResponse]:
        payload_future = self._runtime.forward_backward_custom(batch, loss_fn)
        return RayFuture(payload_future.object_ref, self._build_response)

    def optim_step(self, params: AdamParams) -> RayFuture[dict]:
        return self._runtime.optim_step(params)

    def save_weights_and_get_sampling_client(self, name: str) -> SamplingClient:
        version = self._runtime.refresh_sampler_weights()
        return SamplingClient(runtime=self._runtime, tokenizer=self._tokenizer, weights_version=version)

    def create_reward_actors(self, reward_fns: Iterable[Callable[..., float]]):
        return self._runtime.create_reward_actors(reward_fns)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_response(self, payload: ForwardBackwardPayload) -> ForwardBackwardResponse:
        return ForwardBackwardResponse(
            loss=payload.loss,
            metrics=dict(payload.metrics),
            loss_fn_outputs=dict(payload.loss_fn_outputs),
        )


__all__ = ["TrainingClient", "ForwardBackwardResponse"]
