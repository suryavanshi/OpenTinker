"""Entry point for users interacting with the Ray training runtime."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

from ..core.model_zoo import ModelRegistry
from ..ray_runtime import RayRuntime, RayRuntimeConfig
from ..ray_runtime.config import StreamMinibatchConfig
from .training_client import TrainingClient


@dataclass
class ServiceCapabilities:
    base_models: List[str]


class ServiceClient:
    """Factory for training and sampling clients backed by Ray."""

    def __init__(self, *, runtime_config: RayRuntimeConfig | None = None) -> None:
        self._registry = ModelRegistry()
        self._runtime_config = runtime_config or RayRuntimeConfig()

    def get_server_capabilities(self) -> ServiceCapabilities:
        return ServiceCapabilities(base_models=self._registry.list_models())

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int,
        **kwargs,
    ) -> TrainingClient:
        try:
            spec = self._registry.get(base_model)
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported base model: {base_model}") from exc
        tokenizer = self._registry.create_tokenizer(base_model)
        runtime_config = kwargs.pop("runtime_config", None) or self._runtime_config
        updates: dict[str, object] = {"lora_rank": rank}
        allowed = {
            "lora_alpha",
            "lora_dropout",
            "amp_dtype",
            "gradient_accumulation_steps",
            "sampler_backend",
            "max_steps_off_policy",
            "stream_minibatch",
        }
        for key in list(kwargs.keys()):
            if key not in allowed:
                raise TypeError(f"Unexpected keyword argument '{key}'")
            value = kwargs.pop(key)
            if key == "stream_minibatch" and isinstance(value, dict):
                value = StreamMinibatchConfig(**value)
            updates[key] = value
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
        updates.setdefault("base_model", spec.alias)
        runtime_config = replace(runtime_config, **updates)
        runtime = RayRuntime(
            tokenizer=tokenizer,
            config=runtime_config,
            model_spec=spec,
        )
        return TrainingClient(runtime=runtime)


__all__ = ["ServiceClient", "ServiceCapabilities"]
