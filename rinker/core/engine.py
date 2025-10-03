"""Training utilities for the local single GPU/CPU backend."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, MutableMapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from . import losses
from .lora import LoRAConfig, apply_lora
from .types import AdamParams, Datum


LOSS_REGISTRY: Dict[str, Callable[..., Dict[str, torch.Tensor]]] = {
    "cross_entropy": losses.cross_entropy,
    "importance_sampling": losses.importance_sampling,
    "ppo": losses.ppo,
}


class SimpleLanguageModel(nn.Module):
    """A minimal autoregressive model suitable for quick CPU experiments."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        *,
        lora_config: LoRAConfig | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        if lora_config and lora_config.rank > 0:
            apply_lora(self, lora_config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(input_ids)
        outputs, _ = self.rnn(embeds)
        outputs = self.ln(outputs)
        logits = self.head(outputs)
        return logits


@dataclass
class ForwardBackwardOutput:
    loss: float
    metrics: Dict[str, float]
    loss_fn_outputs: Dict[str, torch.Tensor]


@dataclass
class CustomLossOutputs:
    loss: torch.Tensor
    log_prob_grads: torch.Tensor
    metrics: Mapping[str, float | torch.Tensor] | None = None
    loss_fn_outputs: Mapping[str, torch.Tensor] | None = None


def forward_backward(
    model: nn.Module,
    batch: Sequence[Datum],
    loss_name: str,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None = None,
    grad_scaler: GradScaler | None = None,
    accumulation_steps: int = 1,
) -> ForwardBackwardOutput:
    """Runs a forward/backward pass for the provided batch."""

    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{loss_name}'")

    model.train()
    inputs = torch.cat([datum.model_input.to_batch(device) for datum in batch], dim=0)
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )
    with autocast:
        logits = model(inputs)
    loss_fn = LOSS_REGISTRY[loss_name]

    tensor_inputs = _gather_tensor_inputs(batch, device)
    kwargs: MutableMapping[str, torch.Tensor | float] = dict(tensor_inputs)
    if "targets" in kwargs:
        kwargs["targets"] = kwargs["targets"].long()
    if "target_tokens" in kwargs:
        kwargs["target_tokens"] = kwargs["target_tokens"].long()

    if loss_name == "ppo":
        clip_value = batch[0].loss_fn_inputs.get("clip_epsilon", 0.2)
        if isinstance(clip_value, torch.Tensor):
            clip_value = float(clip_value.item())
        kwargs["clip_epsilon"] = float(clip_value)

    result = loss_fn(logits, **kwargs)
    loss = result["loss"]
    if accumulation_steps > 1:
        loss = loss / float(accumulation_steps)
    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()

    metrics = _build_metrics(result)
    loss_fn_outputs = _detach_loss_outputs(result)

    return ForwardBackwardOutput(
        loss=float(loss.detach().cpu()),
        metrics=metrics,
        loss_fn_outputs=loss_fn_outputs,
    )


def forward_backward_custom(
    model: nn.Module,
    batch: Sequence[Datum],
    custom_fn: Callable[[Sequence[Datum], torch.Tensor], CustomLossOutputs],
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None = None,
    grad_scaler: GradScaler | None = None,
    accumulation_steps: int = 1,
) -> ForwardBackwardOutput:
    """Runs a forward/backward pass using a custom loss defined on log-probs."""

    model.train()
    inputs = torch.cat([datum.model_input.to_batch(device) for datum in batch], dim=0)
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )
    with autocast:
        logits = model(inputs)

    tensor_inputs = _gather_tensor_inputs(batch, device)
    if "target_tokens" not in tensor_inputs:
        raise ValueError("Custom loss requires 'target_tokens' in loss_fn_inputs")

    target_tokens = tensor_inputs["target_tokens"].long()
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)

    custom_outputs = custom_fn(batch, target_log_probs.detach())
    if not isinstance(custom_outputs, CustomLossOutputs):
        raise TypeError("Custom loss function must return CustomLossOutputs")

    log_prob_grads = custom_outputs.log_prob_grads.to(target_log_probs.device)
    surrogate = (log_prob_grads * target_log_probs).sum()
    if accumulation_steps > 1:
        surrogate = surrogate / float(accumulation_steps)
    if grad_scaler is not None:
        grad_scaler.scale(surrogate).backward()
    else:
        surrogate.backward()

    loss_value = custom_outputs.loss
    metric_inputs: Dict[str, torch.Tensor] = {
        "target_log_probs": target_log_probs.detach(),
    }
    if "sampling_logprobs" in tensor_inputs:
        metric_inputs["sampling_logprobs"] = tensor_inputs["sampling_logprobs"].detach()
    metrics = _build_metrics(metric_inputs)

    if custom_outputs.metrics:
        for key, value in custom_outputs.metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = float(value.detach().cpu())
            else:
                metrics[key] = float(value)

    loss_fn_outputs = {
        "target_log_probs": target_log_probs.detach().cpu(),
    }
    if "sampling_logprobs" in tensor_inputs:
        loss_fn_outputs["sampling_logprobs"] = tensor_inputs["sampling_logprobs"].detach().cpu()
    if "advantages" in tensor_inputs:
        loss_fn_outputs["advantages"] = tensor_inputs["advantages"].detach().cpu()
    if custom_outputs.loss_fn_outputs:
        for key, value in custom_outputs.loss_fn_outputs.items():
            if isinstance(value, torch.Tensor):
                loss_fn_outputs[key] = value.detach().cpu()

    return ForwardBackwardOutput(
        loss=float(loss_value.detach().cpu()),
        metrics=metrics,
        loss_fn_outputs=loss_fn_outputs,
    )


def _gather_tensor_inputs(batch: Sequence[Datum], device: torch.device) -> Dict[str, torch.Tensor]:
    tensor_keys = set()
    for datum in batch:
        for key, value in datum.loss_fn_inputs.items():
            if isinstance(value, torch.Tensor):
                tensor_keys.add(key)

    stacked: Dict[str, torch.Tensor] = {}
    for key in tensor_keys:
        tensors = []
        for datum in batch:
            if key not in datum.loss_fn_inputs:
                raise KeyError(f"Missing required tensor '{key}' in datum")
            value = datum.loss_fn_inputs[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Expected tensor for key '{key}' but received {type(value)!r}")
            tensor = value.to(device)
            tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        stacked[key] = torch.cat(tensors, dim=0)
    return stacked


def _build_metrics(loss_outputs: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if "target_log_probs" in loss_outputs and "sampling_logprobs" in loss_outputs:
        target = loss_outputs["target_log_probs"].detach()
        sampling = loss_outputs["sampling_logprobs"].detach()
        log_ratio = target - sampling
        ratio = torch.exp(log_ratio)
        kl_qp = (sampling - target).mean()
        kl_pq = (ratio * log_ratio).mean()
        metrics["kl_q||p"] = float(kl_qp.detach().cpu())
        metrics["kl_p||q"] = float(kl_pq.detach().cpu())
    if "clip_fraction" in loss_outputs:
        clip_fraction = loss_outputs["clip_fraction"].detach()
        metrics["clip_fraction"] = float(clip_fraction.mean().cpu())
    return metrics


def _detach_loss_outputs(result: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    outputs: Dict[str, torch.Tensor] = {}
    for key, value in result.items():
        if key == "loss":
            continue
        if isinstance(value, torch.Tensor):
            outputs[key] = value.detach().cpu()
    return outputs


def optim_step(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    *,
    grad_scaler: GradScaler | None = None,
    should_step: bool = True,
) -> Dict[str, float]:
    """Applies an optimisation step and zeros gradients."""

    if grad_scaler is not None:
        grad_scaler.unscale_(optimiser)
    grad_norm = _total_grad_norm(model)
    if not should_step:
        return {"grad_norm": grad_norm, "stepped": False}
    if grad_scaler is not None:
        grad_scaler.step(optimiser)
        grad_scaler.update()
    else:
        optimiser.step()
    optimiser.zero_grad(set_to_none=True)
    return {"grad_norm": grad_norm, "stepped": True}


def ensure_adam(
    model: nn.Module,
    optimiser: torch.optim.Optimizer | None,
    params: AdamParams,
) -> torch.optim.Optimizer:
    if optimiser is not None:
        return optimiser
    betas = tuple(params.betas)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=params.lr,
        betas=betas,  # type: ignore[arg-type]
        eps=params.eps,
        weight_decay=params.weight_decay,
    )
    return optimiser


def _total_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += param.grad.detach().data.norm(2).item() ** 2
    return total ** 0.5


__all__ = [
    "SimpleLanguageModel",
    "ForwardBackwardOutput",
    "forward_backward",
    "forward_backward_custom",
    "CustomLossOutputs",
    "optim_step",
    "ensure_adam",
]
