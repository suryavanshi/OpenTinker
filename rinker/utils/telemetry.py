"""Telemetry helpers for logging runtime throughput metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


@dataclass
class ThroughputDashboard:
    """Publishes throughput metrics to TensorBoard and W&B."""

    tensorboard_logdir: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    _writer: "SummaryWriter | None" = field(init=False, default=None)
    _wandb_run: object | None = field(init=False, default=None)
    _wandb: object | None = field(init=False, default=None)
    _training_step: int = field(init=False, default=0)
    _sampling_step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.tensorboard_logdir and SummaryWriter is not None:
            self._writer = SummaryWriter(self.tensorboard_logdir)
        if self.wandb_project:
            try:  # pragma: no cover - wandb is optional in tests
                import wandb

                self._wandb = wandb
                self._wandb_run = wandb.init(
                    project=self.wandb_project,
                    name=self.wandb_run_name,
                    entity=self.wandb_entity,
                    reinit=True,
                )
            except Exception:
                self._wandb = None
                self._wandb_run = None

    # ------------------------------------------------------------------
    # Public logging helpers
    # ------------------------------------------------------------------
    def log_training(
        self,
        *,
        duration_s: float,
        tokens: int,
        metrics: Mapping[str, float] | None = None,
        gpu_util: float | None = None,
    ) -> None:
        if duration_s <= 0:
            duration_s = 1e-9
        throughput = tokens / duration_s if tokens else 0.0
        values: MutableMapping[str, float] = {
            "learner/tokens_per_s": throughput,
            "learner/step_duration_s": duration_s,
        }
        if gpu_util is not None:
            values["learner/gpu_util"] = gpu_util
        if metrics:
            for key, value in metrics.items():
                values[f"learner/{key}"] = value
        self._record(values, step=self._training_step)
        self._training_step += 1

    def log_sampling(
        self,
        *,
        duration_s: float,
        prompt_tokens: int,
        generated_tokens: int,
        tokenizer_duration_s: float,
        gpu_util: float | None = None,
    ) -> None:
        if duration_s <= 0:
            duration_s = 1e-9
        total_tokens = prompt_tokens + generated_tokens
        values: MutableMapping[str, float] = {
            "sampler/tokens_per_s": total_tokens / duration_s if total_tokens else 0.0,
            "sampler/generated_tokens_per_s": generated_tokens / duration_s if generated_tokens else 0.0,
            "sampler/tokenizer_tps": prompt_tokens / tokenizer_duration_s if tokenizer_duration_s > 0 else 0.0,
            "sampler/step_duration_s": duration_s,
        }
        if gpu_util is not None:
            values["sampler/gpu_util"] = gpu_util
        self._record(values, step=self._sampling_step)
        self._sampling_step += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
        if self._wandb_run is not None:
            try:  # pragma: no cover - wandb optional
                self._wandb_run.finish()
            except Exception:
                pass
            self._wandb_run = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record(self, values: Mapping[str, float], *, step: int) -> None:
        if self._writer is not None:
            for key, value in values.items():
                self._writer.add_scalar(key, value, global_step=step)
        if self._wandb is not None:
            try:  # pragma: no cover - wandb optional
                self._wandb.log(dict(values), step=step)
            except Exception:
                pass


__all__ = ["ThroughputDashboard"]
