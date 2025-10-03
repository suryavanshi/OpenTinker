"""Lightweight LoRA utilities used by the simple reference model."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class LoRAConfig:
    """Configuration controlling low rank adapters."""

    rank: int
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = ("head",)


class LoRALinear(nn.Module):
    """Wraps an ``nn.Linear`` layer with a trainable low-rank adapter."""

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        if config.rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.rank = config.rank
        self.scaling = config.alpha / float(config.rank)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.zeros(config.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple delegation
        base_out = self.base(inputs)
        dropped = self.dropout(inputs)
        lora_out = F.linear(dropped, self.lora_a)
        lora_out = F.linear(lora_out, self.lora_b)
        return base_out + self.scaling * lora_out


def apply_lora(model: nn.Module, config: LoRAConfig) -> None:
    """Injects LoRA adapters into the requested sub-modules of ``model``."""

    if config.rank <= 0:
        return
    for name, module in model.named_modules():
        if name.split(".")[-1] in config.target_modules and isinstance(module, nn.Linear):
            parent, attr = _resolve_parent(model, name)
            setattr(parent, attr, LoRALinear(module, config))


def merge_lora_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Creates a ``state_dict`` where LoRA weights are merged into their bases."""

    merged_state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        merged_state[key] = value.detach().cpu()

    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        weight_key = f"{name}.base.weight"
        bias_key = f"{name}.base.bias"
        update = (module.lora_b @ module.lora_a) * module.scaling
        merged_state[f"{name}.weight"] = module.base.weight.detach().cpu() + update.detach().cpu()
        if bias_key in merged_state:
            merged_state[f"{name}.bias"] = merged_state.pop(bias_key)
        merged_state.pop(weight_key, None)
        merged_state.pop(f"{name}.lora_a", None)
        merged_state.pop(f"{name}.lora_b", None)
    return merged_state


def extract_lora_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Returns a state dict containing only the LoRA adapter parameters."""

    adapters = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            adapters[f"{name}.lora_a"] = module.lora_a.detach().cpu()
            adapters[f"{name}.lora_b"] = module.lora_b.detach().cpu()
    return adapters


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]

__all__ = ["LoRAConfig", "LoRALinear", "apply_lora", "merge_lora_weights", "extract_lora_parameters"]
