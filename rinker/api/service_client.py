"""Entry point for users interacting with the Ray training runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..utils.tokenizer import SimpleTokenizer
from ..ray_runtime import RayRuntime, RayRuntimeConfig
from .training_client import TrainingClient


@dataclass
class ServiceCapabilities:
    base_models: List[str]


class ServiceClient:
    """Factory for training and sampling clients backed by Ray."""

    def __init__(self, *, runtime_config: RayRuntimeConfig | None = None) -> None:
        self._tokenizer = SimpleTokenizer()
        self._base_models = ["tiny-char-gpt"]
        self._runtime_config = runtime_config or RayRuntimeConfig()

    def get_server_capabilities(self) -> ServiceCapabilities:
        return ServiceCapabilities(base_models=list(self._base_models))

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int,
        **kwargs,
    ) -> TrainingClient:
        if base_model not in self._base_models:
            raise ValueError(f"Unsupported base model: {base_model}")
        runtime_config = kwargs.pop("runtime_config", None) or self._runtime_config
        runtime = RayRuntime(tokenizer=self._tokenizer, config=runtime_config)
        return TrainingClient(runtime=runtime)


__all__ = ["ServiceClient", "ServiceCapabilities"]
