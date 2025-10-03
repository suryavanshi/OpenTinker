"""Runtime orchestration utilities for Ray based training."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, List, Mapping, Sequence

import torch
import ray

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
        self._learner = LearnerActor.options(num_gpus=config.learner_num_gpus).remote(tokenizer.vocab_size)
        self._samplers = [
            SamplerActor.options(num_gpus=config.sampler_num_gpus).remote(tokenizer)
            for _ in range(max(1, config.num_sampler_actors))
        ]
        self._next_sampler = 0
        self._inflight_rollouts: Deque[ray.ObjectRef] = deque()
        self._weights_version = 0

    @property
    def tokenizer(self) -> SimpleTokenizer:
        return self._tokenizer

    @property
    def weights_version(self) -> int:
        return self._weights_version

    # ------------------------------------------------------------------
    # Learner orchestration
    # ------------------------------------------------------------------
    def forward_backward(self, batch: Sequence[Datum], loss_fn: str) -> RayFuture[ForwardBackwardPayload]:
        ref = self._learner.forward_backward.remote(batch, loss_fn)
        return RayFuture(ref)

    def forward_backward_custom(
        self,
        batch: Sequence[Datum],
        loss_fn: Callable[[Sequence[Datum], torch.Tensor], object],
    ) -> RayFuture[ForwardBackwardPayload]:
        ref = self._learner.forward_backward_custom.remote(batch, loss_fn)
        return RayFuture(ref)

    def optim_step(self, params: AdamParams) -> RayFuture[Mapping[str, float]]:
        ref = self._learner.optim_step.remote(params)
        return RayFuture(ref)

    def get_state(self) -> Mapping[str, torch.Tensor]:
        return ray.get(self._learner.get_state.remote())

    def refresh_sampler_weights(self) -> int:
        state = self.get_state()
        self.set_sampler_weights(state)
        self._weights_version += 1
        return self._weights_version

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
        results: List[Mapping[str, object]] = ray.get(ref)
        self._inflight_rollouts.remove(ref)
        return [
            SamplingTaskResult(
                text=result["text"],
                token_ids=list(result["token_ids"]),
                logprobs=list(result["logprobs"]),
                parsed_response=result.get("parsed_response"),
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
