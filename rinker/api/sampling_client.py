"""Sampling client supporting both local and Ray-backed inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn.functional as F

from ..core.types import ModelInput, SamplingParams
from ..ray_runtime import RayRuntime
from ..utils.tokenizer import SimpleTokenizer


@dataclass
class SamplingResult:
    text: str
    token_ids: List[int]
    logprobs: List[float]
    parsed_response: str | None = None
    weights_version: int | None = None


class SamplingClient:
    """Sampling client that can run locally or via the Ray sampler pool."""

    def __init__(
        self,
        *,
        tokenizer: SimpleTokenizer,
        model: torch.nn.Module | None = None,
        runtime: RayRuntime | None = None,
        weights_version: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        if model is not None and runtime is not None:
            raise ValueError("Provide either a local model or a Ray runtime, not both")
        if model is not None:
            self._mode: Literal["local", "ray"] = "local"
            self._model = model
            self._device = torch.device("cpu")
            self._model.to(self._device)
            self._model.eval()
            self._runtime = None
            self._weights_version = None
        elif runtime is not None:
            self._mode = "ray"
            self._model = None
            self._device = None
            self._runtime = runtime
            self._weights_version = weights_version
        else:
            raise ValueError("SamplingClient requires either a model or a Ray runtime")

    def sample(
        self,
        model_input: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int = 1,
    ) -> List[SamplingResult]:
        if self._mode == "local":
            return self._sample_local(model_input, sampling_params, num_samples)
        assert self._runtime is not None, "Ray runtime not initialised"
        results = self._runtime.sample(model_input, sampling_params, num_samples)
        return [
            SamplingResult(
                text=result.text,
                token_ids=result.token_ids,
                logprobs=result.logprobs,
                parsed_response=result.parsed_response,
                weights_version=result.weights_version,
            )
            for result in results
        ]

    # ------------------------------------------------------------------
    # Local sampling implementation (used by legacy unit tests)
    # ------------------------------------------------------------------
    def _sample_local(
        self,
        model_input: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int,
    ) -> List[SamplingResult]:
        assert self._model is not None
        assert self._device is not None
        prompt_tokens = model_input.token_chunks[0].to(self._device)
        results: List[SamplingResult] = []
        for _ in range(num_samples):
            token_ids = prompt_tokens.clone().tolist()
            generated_logprobs: List[float] = []
            for _ in range(sampling_params.max_new_tokens):
                input_tensor = torch.tensor(token_ids, dtype=torch.long, device=self._device).unsqueeze(0)
                logits = self._model(input_tensor)[0, -1]
                next_token, logprob = self._select_token(logits, sampling_params)
                token_ids.append(int(next_token))
                generated_logprobs.append(float(logprob))
                decoded = self._tokenizer.decode(token_ids[len(prompt_tokens) :])
                if self._should_stop(decoded, sampling_params):
                    break
            text = self._tokenizer.decode(token_ids)
            results.append(
                SamplingResult(
                    text=text,
                    token_ids=token_ids,
                    logprobs=generated_logprobs,
                    weights_version=self._weights_version,
                )
            )
        return results

    def _select_token(self, logits: torch.Tensor, params: SamplingParams) -> tuple[int, float]:
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
            if sorted_probs.numel() > 0:
                mask = cumulative <= params.top_p
                mask[..., 0] = True
            else:
                mask = torch.zeros_like(sorted_probs, dtype=torch.bool)
            filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            if filtered_probs.sum() == 0:
                filtered_probs = sorted_probs
            filtered_probs = filtered_probs / filtered_probs.sum()
            choice = torch.multinomial(filtered_probs, 1).item()
            token_id = token_candidates[sorted_idx[choice]].item()
            logprob = torch.log(filtered_probs[choice] + 1e-12).item()
            return token_id, logprob
        sampled_index = torch.multinomial(probs, 1).item()
        token_id = token_candidates[sampled_index].item()
        logprob = torch.log(probs[sampled_index] + 1e-12).item()
        return token_id, logprob

    def _should_stop(self, decoded: str, params: SamplingParams) -> bool:
        if not params.stop_sequences:
            return False
        return any(decoded.endswith(stop) for stop in params.stop_sequences)

    @property
    def weights_version(self) -> int | None:
        return self._weights_version


__all__ = ["SamplingClient", "SamplingResult"]
