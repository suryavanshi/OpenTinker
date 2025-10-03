"""Configuration utilities for the ``rinker`` CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import yaml

from ..core.types import AdamParams, SamplingParams
from ..ray_runtime import RayRuntimeConfig
from ..ray_runtime.config import StreamMinibatchConfig


@dataclass(slots=True)
class SamplingConfig:
    """Serializable sampling configuration used by the CLI."""

    max_new_tokens: int = 32
    temperature: float = 0.7
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: List[str] = field(default_factory=list)

    def to_params(self) -> SamplingParams:
        return SamplingParams(
            max_new_tokens=int(self.max_new_tokens),
            temperature=float(self.temperature),
            top_k=int(self.top_k) if self.top_k is not None else None,
            top_p=float(self.top_p) if self.top_p is not None else None,
            stop_sequences=list(self.stop_sequences) if self.stop_sequences else None,
        )


@dataclass(slots=True)
class CheckpointConfig:
    """Configuration describing checkpoint persistence."""

    dir: str
    every_steps: int = 0
    keep_last: Optional[int] = None
    save_optimizer: bool = True
    save_tokenizer: bool = True
    save_config: bool = True


@dataclass(slots=True)
class InlineEvalConfig:
    """Configuration describing inline evaluation hooks."""

    every_steps: int = 0
    num_env_groups: int = 0
    group_size: int = 1
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    output_dir: Optional[str] = None


@dataclass(slots=True)
class OfflineEvalConfig:
    """Configuration used for offline evaluation sweeps."""

    num_env_groups: int = 0
    group_size: int = 1
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    output_dir: Optional[str] = None


@dataclass(slots=True)
class MathTaskConfig:
    """Toy arithmetic task mirroring the "Your first RL run" walkthrough."""

    prompt: str
    answer: str


@dataclass(slots=True)
class RLLoopConfig:
    """Hyper-parameters controlling the reinforcement learning loop."""

    iterations: int
    batch_size: int
    group_size: int
    num_substeps: int
    beta_kl: float
    loss: str
    learning_rate: float


@dataclass(slots=True)
class RLConfig:
    """Top level RL configuration exposed through the CLI."""

    loop: RLLoopConfig
    tasks: List[MathTaskConfig]
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    eval: Optional[InlineEvalConfig] = None
    offline_eval: Optional[OfflineEvalConfig] = None


@dataclass(slots=True)
class SLDatasetConfig:
    """Simple supervised dataset consisting of prompt/completion pairs."""

    prompts: List[str] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)


@dataclass(slots=True)
class SLConfig:
    """Supervised training configuration for ``rinker train sl``."""

    dataset: SLDatasetConfig
    loss: str = "cross_entropy"
    optimiser: AdamParams = field(default_factory=AdamParams)
    epochs: int = 1


@dataclass(slots=True)
class TrainConfig:
    """Common configuration shared between SL and RL entrypoints."""

    seed: int = 1234
    base_model: Optional[str] = None
    lora_rank: int = 4
    runtime: Mapping[str, Any] | None = None
    checkpoint: Optional[CheckpointConfig] = None
    eval: Optional[InlineEvalConfig] = None
    rl: Optional[RLConfig] = None
    sl: Optional[SLConfig] = None


@dataclass(slots=True)
class EvalConfig:
    """Configuration for ``rinker eval``."""

    seed: int = 1234
    base_model: Optional[str] = None
    lora_rank: int = 4
    runtime: Mapping[str, Any] | None = None
    rl: RLConfig | None = None


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_toml(path: Path) -> Mapping[str, Any]:
    import tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config(path: Path) -> Mapping[str, Any]:
    """Loads a YAML or TOML configuration into a dictionary."""

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(path)
    if suffix == ".toml":
        return _load_toml(path)
    raise ValueError(f"Unsupported configuration format: {path.suffix}")


def build_train_config(data: Mapping[str, Any]) -> TrainConfig:
    payload = dict(data)
    rl_data = payload.get("rl")
    if isinstance(rl_data, Mapping):
        rl_payload = dict(rl_data)
        tasks = [MathTaskConfig(**task) for task in rl_payload.pop("tasks", [])]
        loop = RLLoopConfig(**rl_payload.pop("loop"))
        sampling = SamplingConfig(**rl_payload.pop("sampling", {}))
        eval_cfg = None
        if rl_payload.get("eval"):
            eval_cfg = InlineEvalConfig(
                sampling=SamplingConfig(**rl_payload["eval"].get("sampling", {})),
                **{
                    key: value
                    for key, value in rl_payload["eval"].items()
                    if key != "sampling"
                },
            )
        offline_cfg = None
        if rl_payload.get("offline_eval"):
            offline_cfg = OfflineEvalConfig(
                sampling=SamplingConfig(**rl_payload["offline_eval"].get("sampling", {})),
                **{
                    key: value
                    for key, value in rl_payload["offline_eval"].items()
                    if key != "sampling"
                },
            )
        payload["rl"] = RLConfig(
            loop=loop,
            tasks=tasks,
            sampling=sampling,
            eval=eval_cfg,
            offline_eval=offline_cfg,
        )
    sl_data = payload.get("sl")
    if isinstance(sl_data, Mapping):
        dataset = SLDatasetConfig(**sl_data.get("dataset", {}))
        optimiser = AdamParams(**sl_data.get("optimiser", {}))
        payload["sl"] = SLConfig(
            dataset=dataset,
            loss=sl_data.get("loss", "cross_entropy"),
            optimiser=optimiser,
            epochs=int(sl_data.get("epochs", 1)),
        )
    checkpoint_data = payload.get("checkpoint")
    if isinstance(checkpoint_data, Mapping):
        payload["checkpoint"] = CheckpointConfig(**checkpoint_data)
    eval_data = payload.get("eval")
    if isinstance(eval_data, Mapping):
        payload["eval"] = InlineEvalConfig(
            sampling=SamplingConfig(**eval_data.get("sampling", {})),
            **{key: value for key, value in eval_data.items() if key != "sampling"},
        )
    return TrainConfig(**payload)


def build_eval_config(data: Mapping[str, Any]) -> EvalConfig:
    payload = dict(data)
    rl_data = payload.get("rl")
    if isinstance(rl_data, Mapping):
        rl_payload = dict(rl_data)
        tasks = [MathTaskConfig(**task) for task in rl_payload.pop("tasks", [])]
        loop_payload = rl_payload.get("loop")
        loop = RLLoopConfig(**loop_payload) if isinstance(loop_payload, Mapping) else None
        sampling = SamplingConfig(**rl_payload.get("sampling", {}))
        eval_cfg = rl_payload.get("eval")
        inline_cfg = None
        if isinstance(eval_cfg, Mapping):
            inline_cfg = InlineEvalConfig(
                sampling=SamplingConfig(**eval_cfg.get("sampling", {})),
                **{key: value for key, value in eval_cfg.items() if key != "sampling"},
            )
        offline_cfg = rl_payload.get("offline_eval")
        offline_eval = None
        if isinstance(offline_cfg, Mapping):
            offline_eval = OfflineEvalConfig(
                sampling=SamplingConfig(**offline_cfg.get("sampling", {})),
                **{key: value for key, value in offline_cfg.items() if key != "sampling"},
            )
        payload["rl"] = RLConfig(
            loop=loop if loop is not None else RLLoopConfig(
                iterations=1,
                batch_size=1,
                group_size=1,
                num_substeps=1,
                beta_kl=0.0,
                loss="ppo",
                learning_rate=1e-3,
            ),
            tasks=tasks,
            sampling=sampling,
            eval=inline_cfg,
            offline_eval=offline_eval,
        )
    return EvalConfig(**payload)


def build_runtime_config(overrides: Mapping[str, Any] | None) -> RayRuntimeConfig | None:
    if not overrides:
        return None
    params = dict(overrides)
    stream = params.get("stream_minibatch")
    if isinstance(stream, Mapping):
        params["stream_minibatch"] = StreamMinibatchConfig(**stream)
    return RayRuntimeConfig(**params)


__all__ = [
    "SamplingConfig",
    "CheckpointConfig",
    "InlineEvalConfig",
    "OfflineEvalConfig",
    "MathTaskConfig",
    "RLLoopConfig",
    "RLConfig",
    "SLConfig",
    "TrainConfig",
    "EvalConfig",
    "build_eval_config",
    "build_runtime_config",
    "build_train_config",
    "load_config",
]
