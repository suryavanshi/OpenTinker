"""Model registry and factory utilities for learner and sampler actors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional
import importlib.util
import warnings

import torch

from .engine import SimpleLanguageModel
from .lora import LoRAConfig
from ..utils.tokenizer import HFTokenizerWrapper, SimpleTokenizer, TokenizerProtocol


@dataclass(frozen=True)
class ModelSpec:
    """Describes a base model available to the runtime."""

    alias: str
    architecture: str
    model_id: str | None = None
    tokenizer_id: str | None = None
    hidden_size: int | None = None
    is_moe: bool = False
    default_dtype: str | None = None
    init_kwargs: Mapping[str, object] = field(default_factory=dict)
    trust_remote_code: bool = False


_DEFAULT_SPECS: Dict[str, ModelSpec] = {
    "tiny-char-gpt": ModelSpec(
        alias="tiny-char-gpt",
        architecture="simple",
        hidden_size=256,
    ),
    "qwen3-0.5b": ModelSpec(
        alias="qwen3-0.5b",
        architecture="hf",
        model_id="Qwen/Qwen2.5-0.5B",
        tokenizer_id="Qwen/Qwen2.5-0.5B",
        default_dtype="bfloat16",
        trust_remote_code=True,
    ),
    "qwen3-moe-a14b": ModelSpec(
        alias="qwen3-moe-a14b",
        architecture="hf",
        model_id="Qwen/Qwen2.5-MoE-A14B",
        tokenizer_id="Qwen/Qwen2.5-MoE-A14B",
        default_dtype="bfloat16",
        trust_remote_code=True,
        is_moe=True,
    ),
    "llama3-8b": ModelSpec(
        alias="llama3-8b",
        architecture="hf",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-8B-Instruct",
        default_dtype="bfloat16",
    ),
}


class ModelRegistry:
    """Keeps track of available models and helper factories."""

    def __init__(self, *, overrides: Mapping[str, ModelSpec] | None = None) -> None:
        self._specs: Dict[str, ModelSpec] = dict(_DEFAULT_SPECS)
        if overrides:
            self._specs.update(overrides)

    def list_models(self) -> List[str]:
        def _sort_key(alias: str) -> tuple[int, str]:
            spec = self._specs[alias]
            priority = 0 if spec.architecture == "simple" else 1
            return (priority, alias)

        return sorted(self._specs, key=_sort_key)

    def get(self, alias: str) -> ModelSpec:
        if alias not in self._specs:
            raise KeyError(f"Unknown base model '{alias}'")
        return self._specs[alias]

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------
    def create_tokenizer(self, alias: str) -> TokenizerProtocol:
        spec = self.get(alias)
        return build_tokenizer(spec)

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def create_model(
        self,
        alias: str,
        *,
        vocab_size: int,
        lora_config: LoRAConfig | None,
        amp_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.nn.Module:
        spec = self.get(alias)
        return build_model(
            spec,
            vocab_size=vocab_size,
            lora_config=lora_config,
            amp_dtype=amp_dtype,
            device=device,
        )


def build_tokenizer(spec: ModelSpec) -> TokenizerProtocol:
    if spec.architecture == "simple":
        return SimpleTokenizer()
    if spec.architecture != "hf":
        raise ValueError(f"Unsupported tokenizer architecture '{spec.architecture}'")
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError(
            "transformers is required for Hugging Face model tokenizers. Install with `pip install transformers`."
        )
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        spec.tokenizer_id or spec.model_id,
        use_fast=True,
        trust_remote_code=spec.trust_remote_code,
    )
    wrapper = HFTokenizerWrapper(
        tokenizer=tokenizer,
        decode_skip_special_tokens=bool(getattr(tokenizer, "chat_template", None)),
    )
    return wrapper


def _resolve_dtype(spec: ModelSpec, amp_dtype: torch.dtype | None) -> torch.dtype | None:
    if amp_dtype is not None:
        return amp_dtype
    if spec.default_dtype is None:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = spec.default_dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{spec.default_dtype}' for model '{spec.alias}'")
    return mapping[key]


def build_model(
    spec: ModelSpec,
    *,
    vocab_size: int,
    lora_config: LoRAConfig | None,
    amp_dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.nn.Module:
    target_device = device or torch.device("cpu")
    if spec.architecture == "simple":
        model = SimpleLanguageModel(
            vocab_size,
            hidden_size=spec.hidden_size or 128,
            lora_config=lora_config if lora_config and lora_config.rank > 0 else None,
        )
        model.to(target_device)
        return model
    if spec.architecture != "hf":
        raise ValueError(f"Unsupported architecture '{spec.architecture}'")
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError(
            "transformers is required for Hugging Face models. Install with `pip install transformers`."
        )
    from transformers import AutoModelForCausalLM  # type: ignore

    dtype = _resolve_dtype(spec, amp_dtype)
    torch_dtype = dtype
    init_kwargs: MutableMapping[str, object] = dict(spec.init_kwargs)
    if torch_dtype is not None:
        init_kwargs.setdefault("torch_dtype", torch_dtype)
    init_kwargs.setdefault("trust_remote_code", spec.trust_remote_code)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            **init_kwargs,
        )
    if lora_config and lora_config.rank > 0:
        warnings.warn(
            "LoRA application for Hugging Face models is not implemented in the toy runtime.",
            RuntimeWarning,
        )
    model.to(target_device)
    model.eval()
    return model


__all__ = ["ModelSpec", "ModelRegistry", "build_model", "build_tokenizer"]
