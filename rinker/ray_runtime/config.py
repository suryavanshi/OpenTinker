"""Configuration for the Ray runtime."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RayRuntimeConfig:
    """Configuration controlling Ray actor placement and behaviour."""

    num_sampler_actors: int = 1
    learner_num_gpus: float = 0.0
    sampler_num_gpus: float = 0.0
    reward_num_cpus: float = 0.0
    max_inflight_rollouts: int = 8


__all__ = ["RayRuntimeConfig"]
