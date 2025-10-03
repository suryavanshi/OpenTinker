"""Offline evaluation sweeps over saved checkpoints."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Mapping, MutableMapping, Optional

import matplotlib.pyplot as plt
import torch

from ..core.types import SamplingParams
from ..rl import EnvAction, EnvGroupBuilder


@dataclass(slots=True)
class OfflineEvalSummary:
    """Container returned from :class:`OfflineEvaluator.run`."""

    csv_path: Path
    plot_path: Path | None
    results: List[Mapping[str, float]]


class OfflineEvaluator:
    """Runs evaluation jobs for each checkpoint in a directory."""

    def __init__(
        self,
        *,
        builder: EnvGroupBuilder,
        sampling_params: SamplingParams,
        num_env_groups: int,
        group_size: int,
        output_dir: Optional[Path] = None,
    ) -> None:
        self._builder = builder
        self._sampling_params = sampling_params
        self._num_env_groups = max(int(num_env_groups), 0)
        self._group_size = max(int(group_size), 1)
        self._output_dir = output_dir
        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        training_client,
        checkpoint_root: Path,
    ) -> OfflineEvalSummary:
        if self._num_env_groups <= 0:
            raise ValueError("OfflineEvaluator requires num_env_groups > 0")
        csv_path = (self._output_dir or checkpoint_root) / "offline_eval.csv"
        plot_path = (self._output_dir or checkpoint_root) / "offline_eval_reward.png"

        checkpoints = sorted(p for p in checkpoint_root.iterdir() if p.is_dir())
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found under {checkpoint_root}")

        results: List[Mapping[str, float]] = []
        steps: List[int] = []
        rewards: List[float] = []

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["checkpoint", "global_step", "reward_mean", "reward_std", "acceptance"])
            for checkpoint in checkpoints:
                state_path = checkpoint / "trainer_state.pt"
                if not state_path.exists():
                    continue
                state = torch.load(state_path, map_location="cpu")
                training_client.load_state(state)
                global_step = int(state.get("global_step", 0))
                result = self._evaluate_checkpoint(training_client)
                writer.writerow(
                    [
                        checkpoint.name,
                        global_step,
                        f"{result['reward_mean']:.6f}",
                        f"{result['reward_std']:.6f}",
                        f"{result['acceptance']:.6f}",
                    ]
                )
                enriched = dict(result)
                enriched["checkpoint"] = checkpoint.name
                enriched["global_step"] = float(global_step)
                results.append(enriched)
                steps.append(global_step)
                rewards.append(result["reward_mean"])

        plot_path = self._write_plot(plot_path, steps, rewards)
        return OfflineEvalSummary(csv_path=csv_path, plot_path=plot_path, results=results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _evaluate_checkpoint(self, training_client) -> Mapping[str, float]:
        sampler = training_client.save_weights_and_get_sampling_client("offline-eval")
        self._builder.reset()
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
        summary_metrics: Mapping[str, float] = {
            key: (sum(values) / len(values) if values else 0.0)
            for key, values in metric_accumulator.items()
        }
        payload = dict(summary_metrics)
        payload["reward_mean"] = reward_mean
        payload["reward_std"] = reward_std
        payload["acceptance"] = acceptance
        return payload

    def _write_plot(self, path: Path, steps: List[int], rewards: List[float]) -> Path | None:
        if not steps:
            return None
        plt.figure(figsize=(6, 4))
        plt.plot(steps, rewards, marker="o")
        plt.title("Offline evaluation reward vs. step")
        plt.xlabel("Global step")
        plt.ylabel("Mean reward")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path


__all__ = ["OfflineEvaluator", "OfflineEvalSummary"]
