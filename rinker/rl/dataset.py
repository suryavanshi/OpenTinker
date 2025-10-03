"""Dataset utilities for token-level reinforcement learning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch

from ..core.types import Datum


@dataclass
class RLSample:
    """Container storing data for a single sampled trajectory."""

    group_id: int
    datum: Datum
    reward: float
    raw_reward: float
    token_count: int
    advantage_mask: torch.Tensor | None = None
    kl_value: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)


class RLDataset:
    """Mutable dataset that accumulates samples across environment groups."""

    def __init__(self) -> None:
        self._samples: List[RLSample] = []

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add_sample(
        self,
        *,
        group_id: int,
        datum: Datum,
        reward: float,
        token_count: int,
        advantage_mask: torch.Tensor | None = None,
        kl_value: float | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        if advantage_mask is not None and not isinstance(advantage_mask, torch.Tensor):
            raise TypeError("advantage_mask must be a torch.Tensor or None")
        if advantage_mask is not None:
            if advantage_mask.dtype != torch.bool:
                raise TypeError("advantage_mask must be a boolean tensor")
        sample = RLSample(
            group_id=group_id,
            datum=datum,
            reward=reward,
            raw_reward=reward,
            token_count=int(token_count),
            advantage_mask=advantage_mask,
            kl_value=kl_value,
            metrics=dict(metrics or {}),
        )
        self._samples.append(sample)

    def extend(self, samples: Iterable[RLSample]) -> None:
        for sample in samples:
            self._samples.append(sample)

    def clear(self) -> None:
        self._samples.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def samples(self) -> Sequence[RLSample]:
        return tuple(self._samples)

    @property
    def is_empty(self) -> bool:
        return not self._samples

    @property
    def num_groups(self) -> int:
        return len({sample.group_id for sample in self._samples})

    @property
    def num_samples(self) -> int:
        return len(self._samples)

    @property
    def total_tokens(self) -> int:
        return sum(sample.token_count for sample in self._samples)

    def mean_reward(self, *, shaped: bool = True) -> float:
        rewards = [sample.reward if shaped else sample.raw_reward for sample in self._samples]
        if not rewards:
            return 0.0
        return float(torch.tensor(rewards, dtype=torch.float32).mean().item())

    def reward_std(self, *, shaped: bool = True) -> float:
        rewards = [sample.reward if shaped else sample.raw_reward for sample in self._samples]
        if not rewards:
            return 0.0
        return float(torch.tensor(rewards, dtype=torch.float32).std(unbiased=False).item())

    def acceptance_rate(self, *, threshold: float = 0.0, shaped: bool = True) -> float:
        if self.is_empty:
            return 0.0
        accepted = 0
        for sample in self._samples:
            reward = sample.reward if shaped else sample.raw_reward
            if reward > threshold:
                accepted += 1
        return accepted / max(len(self._samples), 1)

    def average_kl(self) -> float:
        kl_values = [sample.kl_value for sample in self._samples if sample.kl_value is not None]
        if not kl_values:
            return 0.0
        return float(torch.tensor(kl_values, dtype=torch.float32).mean().item())

    # ------------------------------------------------------------------
    # Advantage utilities
    # ------------------------------------------------------------------
    def apply_group_centered_advantages(self) -> None:
        rewards_by_group: Dict[int, List[float]] = {}
        for sample in self._samples:
            rewards_by_group.setdefault(sample.group_id, []).append(sample.reward)

        for group_id, rewards in rewards_by_group.items():
            group_rewards = torch.tensor(rewards, dtype=torch.float32)
            centred = group_rewards - group_rewards.mean()
            centred_list = centred.tolist()
            idx = 0
            for sample in self._samples:
                if sample.group_id != group_id:
                    continue
                advantage_value = centred_list[idx]
                idx += 1
                advantages_tensor = sample.datum.loss_fn_inputs.get("advantages")
                if not isinstance(advantages_tensor, torch.Tensor):
                    raise KeyError("Datum is missing an 'advantages' tensor")
                if sample.advantage_mask is not None and sample.advantage_mask.shape != advantages_tensor.shape:
                    raise ValueError("advantage_mask must match the 'advantages' tensor shape")
                if sample.advantage_mask is None:
                    advantages_tensor.fill_(advantage_value)
                else:
                    advantages_tensor[sample.advantage_mask] = advantage_value

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------
    def apply_kl_shaping(self, beta: float) -> None:
        if beta <= 0:
            return
        for sample in self._samples:
            if sample.kl_value is None:
                sample.reward = sample.raw_reward
            else:
                sample.reward = sample.raw_reward - beta * sample.kl_value

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def build_batch(self) -> List[Datum]:
        return [sample.datum for sample in self._samples]

    def group_reward_summary(self) -> Dict[int, float]:
        summary: Dict[int, float] = {}
        for sample in self._samples:
            summary.setdefault(sample.group_id, 0.0)
            summary[sample.group_id] += sample.reward
        return summary

    def metrics_summary(self) -> Dict[str, float]:
        aggregate: MutableMapping[str, List[float]] = {}
        for sample in self._samples:
            for key, value in sample.metrics.items():
                aggregate.setdefault(key, []).append(float(value))
        summary: Dict[str, float] = {}
        for key, values in aggregate.items():
            summary[key] = float(torch.tensor(values, dtype=torch.float32).mean().item())
        return summary


__all__ = ["RLDataset", "RLSample"]
