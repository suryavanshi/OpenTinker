"""Inline evaluation hooks for RL training loops."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, MutableMapping, Optional

from ..core.types import SamplingParams
from ..rl import EnvAction, EnvGroupBuilder


@dataclass(slots=True)
class InlineEvalResult:
    """Summary of an inline evaluation pass."""

    step: int
    global_step: int
    reward_mean: float
    reward_std: float
    acceptance: float
    metrics: Mapping[str, float] = field(default_factory=dict)


class InlineEvaluator:
    """Runs evaluation episodes at a fixed cadence during training."""

    def __init__(
        self,
        *,
        builder: EnvGroupBuilder,
        sampling_params: SamplingParams,
        every_steps: int,
        num_env_groups: int,
        group_size: int,
        output_dir: Optional[Path] = None,
    ) -> None:
        self._builder = builder
        self._sampling_params = sampling_params
        self._every_steps = max(int(every_steps), 0)
        self._num_env_groups = max(int(num_env_groups), 0)
        self._group_size = max(int(group_size), 1)
        self._records: List[InlineEvalResult] = []
        self._csv_path: Path | None = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = output_dir / "inline_eval.csv"
            if not self._csv_path.exists():
                with self._csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([
                        "step",
                        "global_step",
                        "reward_mean",
                        "reward_std",
                        "acceptance",
                    ])

    @property
    def records(self) -> Iterable[InlineEvalResult]:
        return tuple(self._records)

    def maybe_run(self, step: int, *, training_client) -> InlineEvalResult | None:
        if self._every_steps <= 0:
            return None
        if (step + 1) % self._every_steps != 0:
            return None
        return self.run(step=step, training_client=training_client)

    def run(self, *, step: int, training_client) -> InlineEvalResult:
        sampler = training_client.save_weights_and_get_sampling_client(f"inline-eval-{step}")
        self._builder.reset()
        if self._num_env_groups <= 0:
            raise ValueError("InlineEvaluator requires num_env_groups > 0")
        groups = self._builder.build(self._num_env_groups)

        rewards: List[float] = []
        metric_accumulator: MutableMapping[str, List[float]] = {}
        accepted = 0
        total = 0
        for group in groups:
            results = sampler.sample(
                group.observation.model_input,
                self._sampling_params,
                num_samples=self._group_size,
            )
            for sample in results:
                action = EnvAction(token_ids=sample.token_ids, logprobs=sample.logprobs, text=sample.text)
                transition = group.env.step(action)
                reward = float(transition.reward)
                rewards.append(reward)
                accepted += 1 if reward > 0 else 0
                total += 1
                for key, value in transition.metrics.items():
                    metric_accumulator.setdefault(key, []).append(float(value))

        reward_mean = mean(rewards) if rewards else 0.0
        reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
        acceptance = accepted / total if total else 0.0
        summary_metrics: Dict[str, float] = {
            key: (sum(values) / len(values) if values else 0.0)
            for key, values in metric_accumulator.items()
        }

        result = InlineEvalResult(
            step=step,
            global_step=step + 1,
            reward_mean=reward_mean,
            reward_std=reward_std,
            acceptance=acceptance,
            metrics=summary_metrics,
        )
        self._records.append(result)
        self._write_csv(result)
        return result

    def _write_csv(self, result: InlineEvalResult) -> None:
        if self._csv_path is None:
            return
        with self._csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    result.step,
                    result.global_step,
                    f"{result.reward_mean:.6f}",
                    f"{result.reward_std:.6f}",
                    f"{result.acceptance:.6f}",
                ]
            )


__all__ = ["InlineEvaluator", "InlineEvalResult"]
