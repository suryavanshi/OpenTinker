"""Runtime orchestration utilities for Ray based training."""
from __future__ import annotations

import contextlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, List, Mapping, Sequence

import torch
import ray
from ray.exceptions import RayActorError
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group

from ..core.model_zoo import ModelSpec
from ..core.types import AdamParams, Datum, ModelInput, SamplingParams
from ..utils.futures import RayFuture
from ..utils.telemetry import ThroughputDashboard
from ..utils.tokenizer import TokenizerProtocol
from .actors import ForwardBackwardPayload, LearnerActor, RewardActor, SamplerActor
from .config import RayRuntimeConfig


@dataclass
class SamplingTaskResult:
    """Container returned from sampler actors."""

    text: str
    token_ids: List[int]
    logprobs: List[float]
    token_logprobs: List[float | None]
    parsed_response: str | None
    weights_version: int
    prompt_tokens: int
    tokenizer_time_s: float
    gpu_utilization: float | None
    processor_inputs: Mapping[str, object] | None


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
        tokenizer: TokenizerProtocol,
        config: RayRuntimeConfig,
        model_spec: ModelSpec,
    ) -> None:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)
        self._tokenizer = tokenizer
        self._config = config
        self._model_spec = model_spec
        telemetry_kwargs = config.telemetry_kwargs()
        if any(telemetry_kwargs.values()):
            self._dashboard: ThroughputDashboard | None = ThroughputDashboard(**telemetry_kwargs)
        else:
            self._dashboard = None
        self._placement_group: PlacementGroup | None = None
        if config.use_placement_group:
            self._placement_group = self._create_placement_group()
        self._learner = self._create_learner_actor()
        self._samplers = [
            self._create_sampler_actor(index)
            for index in range(max(1, config.num_sampler_actors))
        ]
        self._next_sampler = 0
        self._inflight_rollouts: Deque[ray.ObjectRef] = deque()
        self._weights_version = 0
        self._last_checkpoint: Mapping[str, object] | None = None

    @property
    def tokenizer(self) -> TokenizerProtocol:
        return self._tokenizer

    @property
    def weights_version(self) -> int:
        return self._weights_version

    @property
    def config(self) -> RayRuntimeConfig:
        return self._config

    @property
    def model_spec(self) -> ModelSpec:
        return self._model_spec

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._dashboard is not None:
            self._dashboard.close()
            self._dashboard = None
        if self._placement_group is not None:
            with contextlib.suppress(Exception):
                remove_placement_group(self._placement_group)
            self._placement_group = None

    def __del__(self):  # pragma: no cover - destructor best-effort
        with contextlib.suppress(Exception):
            self.close()

    def _create_placement_group(self) -> PlacementGroup | None:
        bundles: List[Mapping[str, float]] = []
        learner_bundle: Mapping[str, float] = {}
        cpu = float(self._config.learner_num_cpus)
        gpu = float(self._config.learner_num_gpus)
        if cpu > 0:
            learner_bundle = {"CPU": cpu}
        else:
            learner_bundle = {}
        if gpu > 0:
            learner_bundle = {**learner_bundle, "GPU": gpu}
        if learner_bundle:
            bundles.append(learner_bundle)
        sampler_bundle: Mapping[str, float] = {}
        sampler_cpu = float(self._config.sampler_num_cpus)
        sampler_gpu = float(self._config.sampler_num_gpus)
        if sampler_cpu > 0:
            sampler_bundle = {"CPU": sampler_cpu}
        if sampler_gpu > 0:
            sampler_bundle = {**sampler_bundle, "GPU": sampler_gpu}
        if not sampler_bundle:
            sampler_bundle = {"CPU": 0.1}
        for _ in range(max(1, self._config.num_sampler_actors)):
            bundles.append(dict(sampler_bundle))
        if not bundles:
            return None
        pg = placement_group(bundles, strategy=self._config.placement_strategy)
        try:
            ray.get(pg.ready(), timeout=self._config.placement_timeout_s)
        except Exception:
            remove_placement_group(pg)
            raise
        return pg

    def _create_learner_actor(self):
        options: dict[str, object] = {
            "num_gpus": self._config.learner_num_gpus,
            "num_cpus": self._config.learner_num_cpus,
        }
        if self._placement_group is not None:
            options.update(
                placement_group=self._placement_group,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
        return LearnerActor.options(**options).remote(
            self._model_spec,
            tokenizer_vocab_size=self._tokenizer.vocab_size,
            **self._config.learner_kwargs(),
        )

    def _create_sampler_actor(self, index: int):
        options: dict[str, object] = {
            "num_gpus": self._config.sampler_num_gpus,
            "num_cpus": self._config.sampler_num_cpus,
        }
        if self._placement_group is not None:
            bundle_index = 1 + index
            options.update(
                placement_group=self._placement_group,
                placement_group_bundle_index=bundle_index,
                placement_group_capture_child_tasks=True,
            )
        return SamplerActor.options(**options).remote(
            self._tokenizer,
            **self._config.sampler_kwargs(),
            hidden_size=self._model_spec.hidden_size,
        )

    # ------------------------------------------------------------------
    # Learner orchestration
    # ------------------------------------------------------------------
    def forward_backward(self, batch: Sequence[Datum], loss_fn: str) -> RayFuture[ForwardBackwardPayload]:
        start_time = time.perf_counter()
        future: RayFuture[ForwardBackwardPayload] = _ResilientRayFuture(
            self, "forward_backward", (batch, loss_fn), {}
        )
        if self._dashboard is not None:
            future = future.with_transform(
                lambda payload: self._record_training(payload, start_time)
            )
        return future

    def forward_backward_custom(
        self,
        batch: Sequence[Datum],
        loss_fn: Callable[[Sequence[Datum], torch.Tensor], object],
    ) -> RayFuture[ForwardBackwardPayload]:
        start_time = time.perf_counter()
        future: RayFuture[ForwardBackwardPayload] = _ResilientRayFuture(
            self,
            "forward_backward_custom",
            (batch, loss_fn),
            {},
        )
        if self._dashboard is not None:
            future = future.with_transform(
                lambda payload: self._record_training(payload, start_time)
            )
        return future

    def _record_training(
        self,
        payload: ForwardBackwardPayload,
        start_time: float,
    ) -> ForwardBackwardPayload:
        duration = time.perf_counter() - start_time
        payload.duration_s = duration
        if self._dashboard is not None:
            self._dashboard.log_training(
                duration_s=duration,
                tokens=payload.tokens_processed,
                metrics=payload.metrics,
                gpu_util=payload.gpu_utilization,
            )
        return payload

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
        start_time = time.perf_counter()
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
        duration = time.perf_counter() - start_time
        sampling_results = [
            SamplingTaskResult(
                text=result["text"],
                token_ids=list(result["token_ids"]),
                logprobs=list(result["logprobs"]),
                token_logprobs=list(result.get("token_logprobs", [])),
                parsed_response=result.get("parsed_response"),
                weights_version=self._weights_version,
                prompt_tokens=int(result.get("prompt_tokens", 0)),
                tokenizer_time_s=float(result.get("tokenizer_time_s", 0.0)),
                gpu_utilization=result.get("gpu_utilization"),
                processor_inputs=result.get("processor_inputs"),
            )
            for result in results
        ]
        if self._dashboard is not None:
            prompt_tokens = sum(item.prompt_tokens for item in sampling_results)
            generated_tokens = sum(len(item.logprobs) for item in sampling_results)
            tokenizer_time = max((item.tokenizer_time_s for item in sampling_results), default=0.0)
            gpu_utils = [item.gpu_utilization for item in sampling_results if item.gpu_utilization is not None]
            gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
            self._dashboard.log_sampling(
                duration_s=duration,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                tokenizer_duration_s=tokenizer_time,
                gpu_util=gpu_util,
            )
        return sampling_results

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
        self._learner = self._create_learner_actor()
        if self._last_checkpoint is not None:
            self._call_learner_sync("load_state", self._last_checkpoint)

    def _restart_sampler(self, sampler: SamplerActor) -> SamplerActor:
        index = self._samplers.index(sampler)
        new_sampler = self._create_sampler_actor(index)
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
