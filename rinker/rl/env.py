"""Environment abstractions for token-level reinforcement learning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, MutableSequence, Protocol, Sequence

from ..core.types import ModelInput


@dataclass
class EnvObservation:
    """Observation returned by an :class:`Env` prior to sampling.

    The observation contains the token-level :class:`~rinker.core.types.ModelInput`
    that should be passed to the sampler as well as optional metadata describing
    the task. The metadata dictionary intentionally accepts arbitrary values so
    that multi-modal environments (for example Qwen2.5-VL that requires image
    tensors) can attach additional artefacts such as pixel buffers or bounding
    boxes. Downstream code can forward these attachments to renderers or
    sampler-side adaptors when vision models are used.
    """

    model_input: ModelInput
    metadata: Mapping[str, Any] = field(default_factory=dict)
    attachments: Mapping[str, Any] | None = None


@dataclass
class EnvAction:
    """Action produced by a policy for a given observation.

    The token ids, text, and per-token log-probabilities are optional to support
    both textual and multi-modal completions. Vision models such as
    Qwen2.5-VL emit text tokens conditioned on image features; the optional
    ``metadata`` field allows callers to include the raw modality payload when
    required by a downstream reward model.
    """

    token_ids: Sequence[int] | None = None
    logprobs: Sequence[float] | None = None
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EnvStepResult:
    """Outcome of :meth:`Env.step`."""

    reward: float
    done: bool = True
    next_observation: EnvObservation | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)


class Env(Protocol):
    """Protocol implemented by reinforcement learning environments."""

    def initial_observation(self) -> EnvObservation:
        """Returns the first observation before any actions are taken."""

    def step(self, action: EnvAction) -> EnvStepResult:
        """Consumes an action and returns the resulting transition."""


@dataclass
class EnvGroup:
    """Container representing a single environment within a batch."""

    env: Env
    observation: EnvObservation
    group_index: int


class EnvGroupBuilder:
    """Utility to assemble batched environment groups for sampling.

    The builder mirrors the grouping semantics from the Tinker cookbook: each
    environment contributes one prompt to the batch and ``group_size`` rollouts
    will later be sampled for that prompt. The builder rotates through the
    provided environments when invoked repeatedly so that long-running training
    loops can operate on datasets larger than ``batch_size`` without manual
    book-keeping.
    """

    def __init__(self, envs: Iterable[Env]):
        env_list = list(envs)
        if not env_list:
            raise ValueError("EnvGroupBuilder requires at least one environment")
        self._envs: List[Env] = env_list
        self._cursor: int = 0

    def build(self, batch_size: int) -> List[EnvGroup]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._envs):
            raise ValueError(
                "batch_size cannot exceed the number of registered environments"
            )
        groups: List[EnvGroup] = []
        for slot in range(batch_size):
            env = self._envs[(self._cursor + slot) % len(self._envs)]
            observation = env.initial_observation()
            groups.append(EnvGroup(env=env, observation=observation, group_index=slot))
        self._cursor = (self._cursor + batch_size) % len(self._envs)
        return groups

    def extend(self, envs: Iterable[Env]) -> None:
        """Registers additional environments for future batches."""

        additional: MutableSequence[Env] = list(envs)
        if not additional:
            return
        self._envs.extend(additional)

    def reset(self) -> None:
        """Resets the rotation cursor, starting the next batch from the first env."""

        self._cursor = 0


__all__ = [
    "Env",
    "EnvAction",
    "EnvGroup",
    "EnvGroupBuilder",
    "EnvObservation",
    "EnvStepResult",
]
