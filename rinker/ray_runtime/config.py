"""Configuration for the Ray runtime."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StreamMinibatchConfig:
    """Configuration describing the streaming minibatch schedule."""

    groups_per_batch: int = 1
    num_minibatches: int = 1


@dataclass(slots=True)
class RayRuntimeConfig:
    """Configuration controlling Ray actor placement and behaviour."""

    num_sampler_actors: int = 1
    learner_num_gpus: float = 0.0
    sampler_num_gpus: float = 0.0
    reward_num_cpus: float = 0.0
    max_inflight_rollouts: int = 8
    sampler_backend: str = "torch"
    lora_rank: int = 8
    lora_alpha: float = 16
    lora_dropout: float = 0.05
    amp_dtype: str | None = None
    gradient_accumulation_steps: int = 1
    max_steps_off_policy: int = 0
    stream_minibatch: StreamMinibatchConfig | None = None

    def learner_kwargs(self) -> dict[str, object]:
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "amp_dtype": self.amp_dtype,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }

    def sampler_kwargs(self) -> dict[str, object]:
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "backend": self.sampler_backend,
        }


__all__ = ["RayRuntimeConfig", "StreamMinibatchConfig"]
