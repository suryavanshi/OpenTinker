"""Helper utilities for reinforcement learning loops."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .dataset import RLDataset


def center_group_advantages(dataset: RLDataset) -> None:
    """Applies group-centred advantages in-place.

    This helper simply forwards to :meth:`RLDataset.apply_group_centered_advantages`
    and exists for API parity with the Tinker cookbook where a standalone helper
    is provided.
    """

    dataset.apply_group_centered_advantages()


def apply_kl_shaping(rewards: Sequence[float], kl_values: Sequence[float], beta: float) -> torch.Tensor:
    """Returns a shaped reward tensor ``r - beta * KL``.

    The helper mirrors the reward-shaping guidance from the cookbook: users are
    expected to subtract the KL penalty before advantages are computed rather
    than mixing it into the PPO loss directly.
    """

    if beta <= 0 or not rewards:
        return torch.tensor(rewards, dtype=torch.float32)
    if len(rewards) != len(kl_values):
        raise ValueError("rewards and kl_values must have matching lengths")
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    kl_tensor = torch.tensor(kl_values, dtype=torch.float32)
    return rewards_tensor - beta * kl_tensor


__all__ = ["apply_kl_shaping", "center_group_advantages"]
