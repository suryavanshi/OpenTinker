"""Training client backed by Ray actors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Sequence, TYPE_CHECKING
import warnings

import torch
import ray

from ..core import engine
from ..core.types import AdamParams, Datum
from ..ray_runtime import RayRuntime, RayRuntimeConfig
from ..ray_runtime.config import StreamMinibatchConfig
from ..ray_runtime.actors import ForwardBackwardPayload
from ..utils.futures import RayFuture
from ..utils.tokenizer import TokenizerProtocol
from .sampling_client import SamplingClient

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..rl.dataset import RLDataset

@dataclass
class ForwardBackwardResponse:
    loss: float
    metrics: dict[str, float]
    loss_fn_outputs: dict[str, torch.Tensor]


class TrainingClient:
    def __init__(self, *, runtime: RayRuntime) -> None:
        self._runtime = runtime
        self._tokenizer = runtime.tokenizer
        self._max_steps_off_policy = runtime.config.max_steps_off_policy

    @property
    def tokenizer(self) -> TokenizerProtocol:
        return self._tokenizer

    @property
    def runtime_config(self) -> RayRuntimeConfig:
        return self._runtime.config

    def forward_backward(
        self,
        batch: Sequence[Datum],
        loss_fn: str = "cross_entropy",
    ) -> RayFuture[ForwardBackwardResponse]:
        filtered = self._filter_off_policy_batch(batch)
        payload_future = self._runtime.forward_backward(filtered, loss_fn)
        return payload_future.with_transform(self._build_response)

    def forward_backward_custom(
        self,
        batch: Sequence[Datum],
        loss_fn: Callable[[Sequence[Datum], torch.Tensor], engine.CustomLossOutputs],
    ) -> RayFuture[ForwardBackwardResponse]:
        filtered = self._filter_off_policy_batch(batch)
        payload_future = self._runtime.forward_backward_custom(filtered, loss_fn)
        return payload_future.with_transform(self._build_response)

    def optim_step(self, params: AdamParams) -> RayFuture[dict]:
        return self._runtime.optim_step(params)

    def save_weights_and_get_sampling_client(self, name: str) -> SamplingClient:
        version = self._runtime.refresh_sampler_weights()
        return SamplingClient(runtime=self._runtime, tokenizer=self._tokenizer, weights_version=version)

    def create_reward_actors(self, reward_fns: Iterable[Callable[..., float]]):
        return self._runtime.create_reward_actors(reward_fns)

    def save_state(self) -> dict[str, object]:
        state = self._runtime.save_state()
        return dict(state)

    def load_state(self, state: Mapping[str, object]) -> None:
        self._runtime.load_state(state)

    def export_lora_weights(self) -> Mapping[str, Mapping[str, torch.Tensor]]:
        return self._runtime.export_for_hf()

    def stream_minibatch_train(
        self,
        dataset: "RLDataset",
        *,
        loss_fn: str,
        optimiser: AdamParams,
        config: StreamMinibatchConfig | None = None,
    ) -> List[ForwardBackwardResponse]:
        if dataset.is_empty:
            return []
        schedule = config or self._runtime.config.stream_minibatch or StreamMinibatchConfig()
        if schedule.groups_per_batch <= 0 or schedule.num_minibatches <= 0:
            raise ValueError("StreamMinibatchConfig must have positive fields")
        pending: List[RayFuture[ForwardBackwardResponse]] = []
        responses: List[ForwardBackwardResponse] = []
        max_pending = max(1, schedule.num_minibatches)
        for minibatch in dataset.iter_minibatches(schedule):
            future = self.forward_backward(minibatch, loss_fn=loss_fn)
            pending.append(future)
            if len(pending) >= max_pending:
                completed = self._pop_ready_future(pending)
                responses.append(completed.result())
                self.optim_step(optimiser).result()
        while pending:
            completed = self._pop_ready_future(pending)
            responses.append(completed.result())
            self.optim_step(optimiser).result()
        return responses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_response(self, payload: ForwardBackwardPayload) -> ForwardBackwardResponse:
        return ForwardBackwardResponse(
            loss=payload.loss,
            metrics=dict(payload.metrics),
            loss_fn_outputs=dict(payload.loss_fn_outputs),
        )

    def _filter_off_policy_batch(self, batch: Sequence[Datum]) -> List[Datum]:
        if self._max_steps_off_policy < 0:
            return list(batch)
        current_version = self._runtime.weights_version
        allowed_min = max(0, current_version - self._max_steps_off_policy)
        filtered: List[Datum] = []
        discarded = 0
        missing = 0
        for datum in batch:
            version = getattr(datum, "policy_version", None)
            if version is None:
                missing += 1
                filtered.append(datum)
                continue
            if version < allowed_min:
                discarded += 1
                continue
            filtered.append(datum)
        if not filtered:
            raise ValueError("All samples exceeded the max_steps_off_policy window")
        if discarded:
            warnings.warn(
                f"Discarded {discarded} datums older than policy version {allowed_min}",
                RuntimeWarning,
            )
        if missing:
            warnings.warn(
                "Datums missing policy_version metadata treated as current policy",
                RuntimeWarning,
            )
        return filtered

    def _pop_ready_future(
        self, futures: List[RayFuture[ForwardBackwardResponse]]
    ) -> RayFuture[ForwardBackwardResponse]:
        if len(futures) == 1:
            return futures.pop(0)
        refs = [future.object_ref for future in futures]
        ready_refs, _ = ray.wait(refs, num_returns=1)
        ready_ref = ready_refs[0]
        for idx, future in enumerate(futures):
            if future.object_ref == ready_ref:
                return futures.pop(idx)
        raise RuntimeError("Failed to locate ready future")


__all__ = ["TrainingClient", "ForwardBackwardResponse"]
