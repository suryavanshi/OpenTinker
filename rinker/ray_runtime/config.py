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
    learner_num_cpus: float = 1.0
    learner_mode: str = "single"  # single|ddp|fsdp
    learner_backend: str = "nccl"
    sampler_num_gpus: float = 0.0
    sampler_num_cpus: float = 1.0
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
    base_model: str = "tiny-char-gpt"
    vision_processor_name: str | None = None
    vision_max_pixels: int = 1048576
    use_placement_group: bool = True
    placement_strategy: str = "PACK"
    placement_timeout_s: float = 120.0
    tensorboard_logdir: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_entity: str | None = None

    def learner_kwargs(self) -> dict[str, object]:
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "amp_dtype": self.amp_dtype,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learner_mode": self.learner_mode,
            "distributed_backend": self.learner_backend,
        }

    def sampler_kwargs(self) -> dict[str, object]:
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "backend": self.sampler_backend,
            "vision_processor": self.vision_processor_name,
            "vision_max_pixels": self.vision_max_pixels,
        }

    def telemetry_kwargs(self) -> dict[str, object]:
        return {
            "tensorboard_logdir": self.tensorboard_logdir,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "wandb_entity": self.wandb_entity,
        }


__all__ = ["RayRuntimeConfig", "StreamMinibatchConfig"]
