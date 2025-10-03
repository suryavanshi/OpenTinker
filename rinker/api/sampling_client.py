"""Sampling client for synchronous inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from ..core.types import ModelInput, SamplingParams
from ..utils.tokenizer import SimpleTokenizer


@dataclass
class SamplingResult:
    text: str
    token_ids: List[int]
    logprobs: List[float]


class SamplingClient:
    def __init__(self, *, model: torch.nn.Module, tokenizer: SimpleTokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()

    def sample(
        self,
        model_input: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int = 1,
    ) -> List[SamplingResult]:
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
            results.append(SamplingResult(text=text, token_ids=token_ids, logprobs=generated_logprobs))
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
            mask = cumulative <= params.top_p
            mask[..., 0] = True
            filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            if filtered_probs.sum() == 0:
                filtered_probs = sorted_probs
            filtered_probs = filtered_probs / filtered_probs.sum()
            choice = torch.multinomial(filtered_probs, 1).item()
            token_id = token_candidates[sorted_idx[choice]].item()
            logprob = torch.log(filtered_probs[choice] + 1e-12).item()
            return token_id, logprob
        else:
            sampled_index = torch.multinomial(probs, 1).item()
            token_id = token_candidates[sampled_index].item()
            logprob = torch.log(probs[sampled_index] + 1e-12).item()
            return token_id, logprob

    def _should_stop(self, decoded: str, params: SamplingParams) -> bool:
        if not params.stop_sequences:
            return False
        return any(decoded.endswith(stop) for stop in params.stop_sequences)


__all__ = ["SamplingClient", "SamplingResult"]
