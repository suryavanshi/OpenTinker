"""Runtime orchestration utilities for Ray based training."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, List, Mapping, Sequence

import torch
import ray
from ray.exceptions import RayActorError

from ..core.types import AdamParams, Datum, ModelInput, SamplingParams
from ..utils.futures import RayFuture
from ..utils.tokenizer import SimpleTokenizer
from .actors import ForwardBackwardPayload, LearnerActor, RewardActor, SamplerActor
from .config import RayRuntimeConfig


@dataclass
class SamplingTaskResult:
    """Container returned from sampler actors."""

    text: str
    token_ids: List[int]
    logprobs: List[float]
    parsed_response: str | None
    weights_version: int


class _ResilientRayFuture(RayFuture):
    """Future wrapper that restarts the learner actor on failure."""

    def __init__(
        self,
        runtime: "RayRuntime",
        method_name: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        *,
        transform: Callable[[object], object] | None = None,
    ) -> None:
        self._runtime = runtime
        self._method_name = method_name
        self._args = tuple(args)
        self._kwargs = dict(kwargs)
        ref = runtime._invoke_learner(method_name, *args, **kwargs)
        super().__init__(ref, transform)

    def result(self):  # type: ignore[override]
        try:
            return super().result()
        except RayActorError:
            self._runtime._restart_learner()
            self._ref = self._runtime._invoke_learner(
                self._method_name, *self._args, **self._kwargs
            )
            return super().result()

    def with_transform(self, transform: Callable[[object], object]):
        self._transform = transform
        return self


class RayRuntime:
    """Manages Ray actors and mediates communication with the driver."""

    def __init__(
        self,
        tokenizer: SimpleTokenizer,
        config: RayRuntimeConfig,
    ) -> None:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)
        self._tokenizer = tokenizer
        self._config = config
        self._config = config
        self._learner = LearnerActor.options(num_gpus=config.learner_num_gpus).remote(
            tokenizer.vocab_size, **config.learner_kwargs()
        )
        self._samplers = [
            SamplerActor.options(num_gpus=config.sampler_num_gpus).remote(
                tokenizer, **config.sampler_kwargs()
            )
            for _ in range(max(1, config.num_sampler_actors))
        ]
        self._next_sampler = 0
        self._inflight_rollouts: Deque[ray.ObjectRef] = deque()
        self._weights_version = 0
        self._last_checkpoint: Mapping[str, object] | None = None

    @property
    def tokenizer(self) -> SimpleTokenizer:
        return self._tokenizer

    @property
    def weights_version(self) -> int:
        return self._weights_version

    @property
    def config(self) -> RayRuntimeConfig:
        return self._config

    # ------------------------------------------------------------------
    # Learner orchestration
    # ------------------------------------------------------------------
    def forward_backward(self, batch: Sequence[Datum], loss_fn: str) -> RayFuture[ForwardBackwardPayload]:
        return _ResilientRayFuture(self, "forward_backward", (batch, loss_fn), {})

    def forward_backward_custom(
        self,
        batch: Sequence[Datum],
        loss_fn: Callable[[Sequence[Datum], torch.Tensor], object],
    ) -> RayFuture[ForwardBackwardPayload]:
        return _ResilientRayFuture(
            self,
            "forward_backward_custom",
            (batch, loss_fn),
            {},
        )

    def optim_step(self, params: AdamParams) -> RayFuture[Mapping[str, float]]:
        return _ResilientRayFuture(self, "optim_step", (params,), {})

    def get_state(self) -> Mapping[str, torch.Tensor]:
        return self._call_learner_sync("get_state")

    def refresh_sampler_weights(self) -> int:
        state = self.get_state()
        self.set_sampler_weights(state)
        self._weights_version += 1
        return self._weights_version

    def export_for_hf(self) -> Mapping[str, Mapping[str, torch.Tensor]]:
        return self._call_learner_sync("export_for_hf")

    def save_state(self) -> Mapping[str, object]:
        checkpoint = self._call_learner_sync("save_state")
        self._last_checkpoint = checkpoint
        return checkpoint

    def load_state(self, state: Mapping[str, object]) -> None:
        self._last_checkpoint = dict(state)
        self._call_learner_sync("load_state", state)

    # ------------------------------------------------------------------
    # Sampler orchestration
    # ------------------------------------------------------------------
    def set_sampler_weights(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        state_ref = ray.put(state_dict)
        ray.get([sampler.set_weights.remote(state_ref) for sampler in self._samplers])

    def sample(
        self,
        model_input: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int,
    ) -> List[SamplingTaskResult]:
        sampler = self._acquire_sampler()
        self._ensure_backpressure()
        ref = sampler.generate.remote(model_input, sampling_params, num_samples)
        self._inflight_rollouts.append(ref)
        try:
            results: List[Mapping[str, object]] = ray.get(ref)
        except RayActorError:
            self._inflight_rollouts.remove(ref)
            sampler = self._restart_sampler(sampler)
            ref = sampler.generate.remote(model_input, sampling_params, num_samples)
            self._inflight_rollouts.append(ref)
            results = ray.get(ref)
        self._inflight_rollouts.remove(ref)
        return [
            SamplingTaskResult(
                text=result["text"],
                token_ids=list(result["token_ids"]),
                logprobs=list(result["logprobs"]),
                parsed_response=result.get("parsed_response"),
                weights_version=self._weights_version,
            )
            for result in results
        ]

    def _acquire_sampler(self) -> SamplerActor:
        handle = self._samplers[self._next_sampler]
        self._next_sampler = (self._next_sampler + 1) % len(self._samplers)
        return handle

    def _ensure_backpressure(self) -> None:
        if len(self._inflight_rollouts) < self._config.max_inflight_rollouts:
            return
        ready, remaining = ray.wait(list(self._inflight_rollouts), num_returns=1)
        self._inflight_rollouts = deque(remaining)

    def _invoke_learner(self, method: str, *args, **kwargs) -> ray.ObjectRef:
        return getattr(self._learner, method).remote(*args, **kwargs)

    def _call_learner_sync(self, method: str, *args, **kwargs):
        ref = self._invoke_learner(method, *args, **kwargs)
        try:
            return ray.get(ref)
        except RayActorError:
            self._restart_learner()
            ref = self._invoke_learner(method, *args, **kwargs)
            return ray.get(ref)

    def _restart_learner(self) -> None:
        self._learner = LearnerActor.options(num_gpus=self._config.learner_num_gpus).remote(
            self._tokenizer.vocab_size, **self._config.learner_kwargs()
        )
        if self._last_checkpoint is not None:
            self._call_learner_sync("load_state", self._last_checkpoint)

    def _restart_sampler(self, sampler: SamplerActor) -> SamplerActor:
        index = self._samplers.index(sampler)
        new_sampler = SamplerActor.options(num_gpus=self._config.sampler_num_gpus).remote(
            self._tokenizer, **self._config.sampler_kwargs()
        )
        self._samplers[index] = new_sampler
        if self._last_checkpoint is not None:
            state = self._last_checkpoint.get("model")
            if isinstance(state, Mapping):
                state_ref = ray.put(state)
                ray.get(new_sampler.set_weights.remote(state_ref))
        return new_sampler

    # ------------------------------------------------------------------
    # Reward actor helpers
    # ------------------------------------------------------------------
    def create_reward_actors(self, reward_fns: Iterable[Callable[..., float]]) -> List[ray.ActorHandle]:
        actors: List[ray.ActorHandle] = []
        for fn in reward_fns:
            actor = RewardActor.options(num_cpus=self._config.reward_num_cpus or None).remote(fn)
            actors.append(actor)
        return actors


__all__ = ["RayRuntime", "SamplingTaskResult"]
