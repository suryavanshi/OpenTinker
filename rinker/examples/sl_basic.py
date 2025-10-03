"""Minimal supervised fine-tuning loop using the synchronous API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

if __package__ is None or __package__ == "":
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from rinker.api.service_client import ServiceClient
from rinker.core.types import AdamParams, Datum, ModelInput, SamplingParams
from rinker.utils.seeding import seed_everything


@dataclass
class ToyDataset:
    prompts: List[str]
    completions: List[str]

    def __iter__(self):
        yield from zip(self.prompts, self.completions)


if __name__ == "__main__":
    seed_everything(42)

    dataset = ToyDataset(
        prompts=["hello" for _ in range(16)],
        completions=[" world" for _ in range(16)],
    )

    service = ServiceClient()
    capabilities = service.get_server_capabilities()
    print("Available models:", capabilities.base_models)
    training = service.create_lora_training_client(capabilities.base_models[0], rank=4)

    tokenizer = training.tokenizer

    batch: List[Datum] = []
    for prompt, completion in dataset:
        full_text = prompt + completion
        token_ids = torch.tensor(tokenizer.encode(full_text), dtype=torch.long)
        inputs = token_ids[:-1]
        targets = token_ids[1:]
        model_input = ModelInput(token_chunks=[inputs])
        loss_inputs = {
            "targets": targets,
            "weights": torch.ones_like(targets, dtype=torch.float32),
        }
        batch.append(Datum(model_input=model_input, loss_fn_inputs=loss_inputs))

    future = training.forward_backward(batch, loss_fn="cross_entropy")
    print("Loss:", future.result().loss)
    training.optim_step(AdamParams(lr=5e-3)).result()

    sampler = training.save_weights_and_get_sampling_client("toy")
    sampling_params = SamplingParams(max_new_tokens=12, temperature=0.8, stop_sequences=["\n"])
    sample = sampler.sample(batch[0].model_input, sampling_params=sampling_params, num_samples=1)[0]
    print("Sample:", sample.text)
