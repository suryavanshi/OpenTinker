"""Entry point for users interacting with the local training runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..core.engine import SimpleLanguageModel
from ..utils.tokenizer import SimpleTokenizer
from .training_client import TrainingClient


@dataclass
class ServiceCapabilities:
    base_models: List[str]


class ServiceClient:
    """Factory for training and sampling clients."""

    def __init__(self) -> None:
        self._tokenizer = SimpleTokenizer()
        self._base_models = ["tiny-char-gpt"]

    def get_server_capabilities(self) -> ServiceCapabilities:
        return ServiceCapabilities(base_models=list(self._base_models))

    def create_lora_training_client(self, base_model: str, rank: int, **_) -> TrainingClient:
        if base_model not in self._base_models:
            raise ValueError(f"Unsupported base model: {base_model}")
        model = SimpleLanguageModel(self._tokenizer.vocab_size)
        return TrainingClient(model=model, tokenizer=self._tokenizer)


__all__ = ["ServiceClient", "ServiceCapabilities"]
