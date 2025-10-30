from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Sequence

import torch

from ..api.sampling_client import SamplingClient, SamplingResult
from ..api.training_client import ForwardBackwardResponse, TrainingClient
from ..core.model_zoo import ModelRegistry, build_model
from ..core.types import AdamParams, Datum, ModelInput, SamplingParams
from ..utils.tokenizer import TokenizerProtocol


@dataclass(slots=True)
class DistillationExample:
    """Single prompt used for on-policy distillation."""

    prompt: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class DistillationConfig:
    """Hyper-parameters controlling the distillation loop."""

    iterations: int = 1
    batch_size: int = 1
    samples_per_prompt: int = 1
    beta: float = 0.5
    temperature: float = 1.0
    learning_rate: float = 1e-4
    on_policy_probability: float = 1.0
    loss_name: str = "generalized_jsd"
    reduction: str = "sum"


@dataclass(slots=True)
class _PreparedExample:
    prompt: str
    model_input: ModelInput
    prompt_tokens: torch.Tensor
    metadata: Mapping[str, object]


@dataclass(slots=True)
class _RawDistillationDatum:
    inputs: torch.Tensor
    teacher_logits: torch.Tensor
    mask: torch.Tensor
    policy_version: int | None


class DistillationTrainer:
    """Implements the on-policy distillation loop inspired by Tinker."""

    def __init__(
        self,
        *,
        training_client: TrainingClient,
        teacher_model: torch.nn.Module | str,
        prompts: Sequence[str | DistillationExample],
        sampling_params: SamplingParams,
        config: DistillationConfig,
        teacher_device: torch.device | None = None,
    ) -> None:
        if not prompts:
            raise ValueError("DistillationTrainer requires at least one prompt")
        self._training = training_client
        self._tokenizer = training_client.tokenizer
        self._sampling_params = sampling_params
        self._config = config
        self._prepared = self._prepare_examples(prompts)
        self._teacher = _TeacherWrapper(
            teacher_model,
            tokenizer=self._tokenizer,
            device=teacher_device,
        )
        self._optim_params = AdamParams(lr=config.learning_rate)
        self._cached_samples: dict[int, List[SamplingResult]] = {}
        self._cursor = 0
        self._rng = random.Random()

    def run(self) -> List[ForwardBackwardResponse]:
        """Executes the distillation loop and returns per-iteration metrics."""

        history: List[ForwardBackwardResponse] = []
        for iteration in range(self._config.iterations):
            batch = self._next_batch()
            sampler = self._training.save_weights_and_get_sampling_client(f"distill-{iteration}")
            datums: List[Datum] = []
            for index, example in batch:
                raw_samples = self._collect_raw_samples(index, example, sampler)
                datums.extend(self._build_datums(raw_samples))
            if not datums:
                continue
            response = self._training.forward_backward(datums, loss_fn=self._config.loss_name).result()
            self._training.optim_step(self._optim_params).result()
            history.append(response)
        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_examples(self, prompts: Sequence[str | DistillationExample]) -> List[_PreparedExample]:
        prepared: List[_PreparedExample] = []
        for entry in prompts:
            if isinstance(entry, DistillationExample):
                prompt = entry.prompt
                metadata = dict(entry.metadata)
            else:
                prompt = str(entry)
                metadata = {}
            token_ids = torch.tensor(self._tokenizer.encode(prompt), dtype=torch.long)
            if token_ids.numel() == 0:
                raise ValueError("Prompts must yield at least one token")
            model_input = ModelInput(token_chunks=[token_ids])
            prepared.append(
                _PreparedExample(
                    prompt=prompt,
                    model_input=model_input,
                    prompt_tokens=token_ids,
                    metadata=metadata,
                )
            )
        return prepared

    def _next_batch(self) -> List[tuple[int, _PreparedExample]]:
        total = len(self._prepared)
        if total == 0:
            raise RuntimeError("No prompts configured for distillation")
        batch: List[tuple[int, _PreparedExample]] = []
        for _ in range(max(1, self._config.batch_size)):
            index = self._cursor % total
            batch.append((index, self._prepared[index]))
            self._cursor = (self._cursor + 1) % total
        return batch

    def _collect_raw_samples(
        self,
        index: int,
        example: _PreparedExample,
        sampler: SamplingClient,
    ) -> List[_RawDistillationDatum]:
        samples = self._obtain_samples(index, example, sampler)
        raw: List[_RawDistillationDatum] = []
        for sample in samples:
            datum = self._prepare_raw_datum(example, sample)
            if datum is not None:
                raw.append(datum)
        return raw

    def _obtain_samples(
        self,
        index: int,
        example: _PreparedExample,
        sampler: SamplingClient,
    ) -> List[SamplingResult]:
        should_refresh = (
            self._rng.random() <= self._config.on_policy_probability
            or index not in self._cached_samples
        )
        if should_refresh:
            results = sampler.sample(
                example.model_input,
                self._sampling_params,
                num_samples=max(1, self._config.samples_per_prompt),
            )
            self._cached_samples[index] = list(results)
        return list(self._cached_samples[index])

    def _prepare_raw_datum(
        self,
        example: _PreparedExample,
        sample: SamplingResult,
    ) -> _RawDistillationDatum | None:
        token_ids = torch.tensor(sample.token_ids, dtype=torch.long)
        if token_ids.numel() <= 1:
            return None
        inputs = token_ids[:-1]
        teacher_logits = self._teacher.compute_logits(inputs)
        seq_len = teacher_logits.shape[0]
        mask = torch.zeros(seq_len, dtype=torch.bool)
        completion_start = max(int(example.prompt_tokens.numel()) - 1, 0)
        completion_start = min(completion_start, mask.shape[0])
        mask[completion_start:] = True
        return _RawDistillationDatum(
            inputs=inputs,
            teacher_logits=teacher_logits,
            mask=mask,
            policy_version=getattr(sample, "weights_version", None),
        )

    def _build_datums(self, samples: Iterable[_RawDistillationDatum]) -> List[Datum]:
        samples = list(samples)
        if not samples:
            return []
        max_len = max(d.teacher_logits.shape[0] for d in samples)
        vocab = samples[0].teacher_logits.shape[1] if max_len > 0 else self._teacher.vocab_size
        datums: List[Datum] = []
        for sample in samples:
            seq_len = sample.teacher_logits.shape[0]
            if seq_len < max_len:
                pad_rows = max_len - seq_len
                padding = torch.zeros((pad_rows, vocab), dtype=sample.teacher_logits.dtype)
                teacher_logits = torch.cat([sample.teacher_logits, padding], dim=0)
            else:
                teacher_logits = sample.teacher_logits
            mask = sample.mask
            if mask.shape[0] < max_len:
                padded_mask = torch.zeros(max_len, dtype=mask.dtype)
                padded_mask[: mask.shape[0]] = mask
                mask = padded_mask
            datum = Datum(
                model_input=ModelInput(token_chunks=[sample.inputs]),
                loss_fn_inputs={
                    "teacher_logits": teacher_logits,
                    "loss_mask": mask,
                    "beta": self._config.beta,
                    "temperature": self._config.temperature,
                    "reduction": self._config.reduction,
                },
                policy_version=sample.policy_version,
            )
            datums.append(datum)
        return datums


class _TeacherWrapper:
    """Utility that standardises access to the teacher model."""

    def __init__(
        self,
        teacher_model: torch.nn.Module | str,
        *,
        tokenizer: TokenizerProtocol,
        device: torch.device | None = None,
    ) -> None:
        registry = ModelRegistry()
        self._device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if isinstance(teacher_model, str):
            spec = registry.get(teacher_model)
            model = build_model(
                spec,
                vocab_size=tokenizer.vocab_size,
                lora_config=None,
                amp_dtype=None,
                device=self._device,
            )
        else:
            model = teacher_model.to(self._device)
        self._model = model.eval()
        self._vocab_size = tokenizer.vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def compute_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.zeros((0, self._vocab_size), dtype=torch.float32)
        batch = input_ids.unsqueeze(0).to(self._device)
        attention_mask = torch.ones_like(batch)
        with torch.no_grad():
            try:
                outputs = self._model(input_ids=batch, attention_mask=attention_mask)  # type: ignore[arg-type]
            except TypeError:
                outputs = self._model(batch)  # type: ignore[call-arg]
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        logits = logits.squeeze(0).to(torch.float32).detach().cpu()
        return logits
