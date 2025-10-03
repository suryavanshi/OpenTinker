"""Ray actors for the learner, sampler, and reward components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Mapping, Sequence
import importlib.util
import warnings

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from ..core import engine
from ..core.lora import LoRAConfig, extract_lora_parameters, merge_lora_weights
from ..core.types import AdamParams, Datum, ModelInput, SamplingParams
from ..utils.tokenizer import SimpleTokenizer

try:  # pragma: no cover - ray import is optional in unit tests
    import ray
except ImportError as exc:  # pragma: no cover - handled by caller
    raise RuntimeError(
        "Ray is required for the ray_runtime components. "
        "Install with `pip install ray`."
    ) from exc


@dataclass
class ForwardBackwardPayload:
    loss: float
    metrics: Mapping[str, float]
    loss_fn_outputs: Mapping[str, torch.Tensor]


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            return torch.device("cuda")
    return torch.device("cpu")


@ray.remote
class LearnerActor:
    """Learner actor that owns the trainable policy."""

    def __init__(
        self,
        vocab_size: int,
        *,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        amp_dtype: str | None = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self._tokenizer_vocab_size = vocab_size
        self._device = _resolve_device()
        lora_config = LoRAConfig(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        self._model = engine.SimpleLanguageModel(vocab_size, lora_config=lora_config if lora_rank > 0 else None)
        self._model.to(self._device)
        self._optimiser: torch.optim.Optimizer | None = None
        self._amp_dtype = self._resolve_amp_dtype(amp_dtype)
        self._grad_scaler = GradScaler(enabled=self._amp_dtype == torch.float16 and self._device.type == "cuda")
        self._accum_steps = max(int(gradient_accumulation_steps), 1)
        self._micro_step_progress = 0
        self._global_step = 0
        self._pending_optim_state: Mapping[str, object] | None = None

    def forward_backward(self, batch: Sequence[Datum], loss_fn: str) -> ForwardBackwardPayload:
        result = engine.forward_backward(
            self._model,
            batch,
            loss_fn,
            device=self._device,
            amp_dtype=self._amp_dtype,
            grad_scaler=self._grad_scaler if self._grad_scaler.is_enabled() else None,
            accumulation_steps=self._accum_steps,
        )
        self._micro_step_progress += 1
        return ForwardBackwardPayload(
            loss=result.loss,
            metrics=result.metrics,
            loss_fn_outputs=result.loss_fn_outputs,
        )

    def forward_backward_custom(
        self,
        batch: Sequence[Datum],
        loss_fn: Callable[[Sequence[Datum], torch.Tensor], engine.CustomLossOutputs],
    ) -> ForwardBackwardPayload:
        result = engine.forward_backward_custom(
            self._model,
            batch,
            loss_fn,
            device=self._device,
            amp_dtype=self._amp_dtype,
            grad_scaler=self._grad_scaler if self._grad_scaler.is_enabled() else None,
            accumulation_steps=self._accum_steps,
        )
        self._micro_step_progress += 1
        return ForwardBackwardPayload(
            loss=result.loss,
            metrics=result.metrics,
            loss_fn_outputs=result.loss_fn_outputs,
        )

    def optim_step(self, params: AdamParams) -> Mapping[str, float]:
        self._optimiser = engine.ensure_adam(self._model, self._optimiser, params)
        if self._pending_optim_state is not None:
            self._optimiser.load_state_dict(self._pending_optim_state)  # type: ignore[arg-type]
            self._pending_optim_state = None
        should_step = self._micro_step_progress >= self._accum_steps
        metrics = engine.optim_step(
            self._model,
            self._optimiser,
            grad_scaler=self._grad_scaler if self._grad_scaler.is_enabled() else None,
            should_step=should_step,
        )
        if metrics.get("stepped"):
            self._global_step += 1
            self._micro_step_progress = 0
        return metrics

    def get_state(self) -> Mapping[str, torch.Tensor]:
        state_dict = self._model.state_dict()
        return {key: value.detach().cpu() for key, value in state_dict.items()}

    def save_state(self) -> Mapping[str, object]:
        optimiser_state = self._optimiser.state_dict() if self._optimiser is not None else None
        grad_scaler_state = self._grad_scaler.state_dict() if self._grad_scaler.is_enabled() else None
        model_state = {key: value.detach().cpu() for key, value in self._model.state_dict().items()}
        return {
            "model": model_state,
            "optimiser": optimiser_state,
            "grad_scaler": grad_scaler_state,
            "global_step": self._global_step,
            "accumulation_progress": self._micro_step_progress,
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        model_state = state.get("model")
        if model_state:
            self._model.load_state_dict(model_state)
        if self._optimiser is not None and state.get("optimiser") is not None:
            self._optimiser.load_state_dict(state["optimiser"])  # type: ignore[arg-type]
        elif state.get("optimiser") is not None:
            self._pending_optim_state = state["optimiser"]  # type: ignore[assignment]
            warnings.warn(
                "Optimiser state deferred until optimiser initialisation",
                RuntimeWarning,
            )
        scaler_state = state.get("grad_scaler")
        if scaler_state and self._grad_scaler.is_enabled():
            self._grad_scaler.load_state_dict(scaler_state)  # type: ignore[arg-type]
        self._global_step = int(state.get("global_step", self._global_step))
        self._micro_step_progress = int(state.get("accumulation_progress", 0))

    def export_for_hf(self) -> Mapping[str, Mapping[str, torch.Tensor]]:
        return {
            "merged": merge_lora_weights(self._model),
            "adapters": extract_lora_parameters(self._model),
        }

    @staticmethod
    def _resolve_amp_dtype(value: str | None) -> torch.dtype | None:
        if value is None:
            return None
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if value.lower() not in mapping:
            raise ValueError(f"Unsupported AMP dtype '{value}'")
        return mapping[value.lower()]


@ray.remote
class SamplerActor:
    """Sampler actor responsible for autoregressive generation."""

    def __init__(
        self,
        tokenizer: SimpleTokenizer,
        *,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        backend: str = "torch",
    ):
        self._tokenizer = tokenizer
        self._device = _resolve_device()
        lora_config = LoRAConfig(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        self._model = engine.SimpleLanguageModel(
            tokenizer.vocab_size,
            lora_config=lora_config if lora_rank > 0 else None,
        )
        self._model.to(self._device)
        self._model.eval()
        self._stop_sequences: List[str] = []
        self._backend = backend
        if backend == "vllm":
            if importlib.util.find_spec("vllm") is None:
                warnings.warn(
                    "vLLM backend requested but vllm is not installed; falling back to torch sampler",
                    RuntimeWarning,
                )
                self._backend = "torch"
            else:  # pragma: no cover - optional path
                import vllm  # type: ignore

                self._vllm = vllm

    def set_weights(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

    def generate(
        self,
        model_input: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int,
    ) -> List[Mapping[str, object]]:
        prompt_tokens = self._prepare_prompt(model_input)
        prompt_tokens = prompt_tokens.to(self._device)
        outputs: List[Mapping[str, object]] = []
        stop_sequences = sampling_params.stop_sequences or self._stop_sequences

        for _ in range(num_samples):
            token_ids = prompt_tokens.clone().tolist()
            generated_logprobs: List[float] = []
            for _ in range(sampling_params.max_new_tokens):
                input_tensor = torch.tensor(token_ids, dtype=torch.long, device=self._device).unsqueeze(0)
                logits = self._model(input_tensor)[0, -1]
                token_id, logprob = self._sample_token(logits, sampling_params)
                token_ids.append(int(token_id))
                generated_logprobs.append(float(logprob))

                decoded = self._tokenizer.decode(token_ids[len(prompt_tokens) :])
                if stop_sequences and any(decoded.endswith(stop) for stop in stop_sequences):
                    break

            text = self._tokenizer.decode(token_ids)
            parsed = self._parse_response(model_input, text)
            outputs.append(
                {
                    "text": text,
                    "token_ids": token_ids,
                    "logprobs": generated_logprobs,
                    "parsed_response": parsed,
                }
            )
        return outputs

    def _prepare_prompt(self, model_input: ModelInput) -> torch.Tensor:
        renderer = model_input.metadata.get("renderer") if model_input.metadata else None
        messages = model_input.metadata.get("messages") if model_input.metadata else None
        if renderer and messages:
            prompt_text = renderer.build_generation_prompt(messages)
            self._stop_sequences = renderer.get_stop_sequences()
            encoded = self._tokenizer.encode(prompt_text)
            return torch.tensor(encoded, dtype=torch.long)
        if not model_input.token_chunks:
            raise ValueError("ModelInput must contain at least one token chunk")
        self._stop_sequences = []
        return model_input.token_chunks[0]

    def _parse_response(self, model_input: ModelInput, text: str) -> str | None:
        renderer = model_input.metadata.get("renderer") if model_input.metadata else None
        if renderer:
            return renderer.parse_response(text)
        return None

    def _sample_token(self, logits: torch.Tensor, params: SamplingParams) -> tuple[int, float]:
        if params.temperature <= 0:
            raise ValueError("Temperature must be > 0")
        logits = logits / params.temperature
        probs = F.softmax(logits, dim=-1)
        if params.top_k is not None:
            topk = min(params.top_k, probs.numel())
            values, indices = torch.topk(probs, topk)
            probs = values / values.sum()
            token_candidates = indices
        else:
            token_candidates = torch.arange(probs.numel(), device=probs.device)
        if params.top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative <= params.top_p
            if sorted_probs.numel() > 0:
                mask[..., 0] = True
            filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            if filtered.sum() == 0:
                filtered = sorted_probs
            filtered = filtered / filtered.sum()
            choice = torch.multinomial(filtered, 1).item()
            token_id = token_candidates[sorted_idx[choice]].item()
            logprob = torch.log(filtered[choice] + 1e-12).item()
            return token_id, logprob
        sampled_index = torch.multinomial(probs, 1).item()
        token_id = token_candidates[sampled_index].item()
        logprob = torch.log(probs[sampled_index] + 1e-12).item()
        return token_id, logprob


@ray.remote
class RewardActor:
    """Reward actor wrapping a user supplied callable."""

    def __init__(self, reward_fn: Callable[..., float]):
        self._reward_fn = reward_fn

    def compute(self, *args, **kwargs) -> float:
        return float(self._reward_fn(*args, **kwargs))


__all__ = ["LearnerActor", "SamplerActor", "RewardActor", "ForwardBackwardPayload"]
