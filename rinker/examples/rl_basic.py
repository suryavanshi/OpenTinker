"""Toy PPO/IS reinforcement learning loop demonstrating rising rewards."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

if __package__ is None or __package__ == "":
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from rinker.api.service_client import ServiceClient
from rinker.api.training_client import ForwardBackwardResponse
from rinker.core.types import AdamParams, Datum, ModelInput, SamplingParams
from rinker.utils.seeding import seed_everything


@dataclass
class PromptSpec:
    prompt: str
    answer: str


def build_batch(
    tokenizer,
    sampler,
    specs: Sequence[PromptSpec],
    *,
    group_size: int,
    clip_epsilon: float,
) -> Tuple[List[Datum], float]:
    batch: List[Datum] = []
    total_reward = 0.0
    total_samples = 0

    sampling_params = SamplingParams(max_new_tokens=1, temperature=0.8)

    for spec in specs:
        prompt_tokens = torch.tensor(tokenizer.encode(spec.prompt), dtype=torch.long)
        model_input = ModelInput(token_chunks=[prompt_tokens])
        samples = sampler.sample(model_input, sampling_params=sampling_params, num_samples=group_size)

        group: List[Tuple[float, Datum, int]] = []
        for sample in samples:
            completion = sample.text[len(spec.prompt) :].strip()
            reward = 1.0 if completion.startswith(spec.answer) else 0.0
            total_reward += reward
            total_samples += 1

            token_ids = torch.tensor(sample.token_ids, dtype=torch.long)
            inputs = token_ids[:-1]
            targets = token_ids[1:]
            sampling_logprobs = torch.zeros_like(targets, dtype=torch.float32)
            completion_start = len(prompt_tokens) - 1
            if sample.logprobs:
                completion_logprobs = torch.tensor(sample.logprobs, dtype=torch.float32)
                sampling_logprobs[completion_start : completion_start + completion_logprobs.numel()] = completion_logprobs

            advantages = torch.zeros_like(targets, dtype=torch.float32)
            loss_inputs: Dict[str, object] = {
                "target_tokens": targets,
                "sampling_logprobs": sampling_logprobs,
                "advantages": advantages,
                "clip_epsilon": clip_epsilon,
            }
            datum = Datum(model_input=ModelInput(token_chunks=[inputs]), loss_fn_inputs=loss_inputs)
            group.append((reward, datum, completion_start))

        if not group:
            continue

        rewards = torch.tensor([entry[0] for entry in group], dtype=torch.float32)
        centred = rewards - rewards.mean()
        for centred_advantage, (_, datum, completion_start) in zip(centred, group):
            advantages_tensor = datum.loss_fn_inputs["advantages"]
            assert isinstance(advantages_tensor, torch.Tensor)
            advantages_tensor[completion_start:] = centred_advantage
            batch.append(datum)

    avg_reward = total_reward / max(total_samples, 1)
    return batch, avg_reward


def run_loop(
    loss_name: str,
    *,
    specs: Sequence[PromptSpec],
    iterations: int,
    group_size: int = 4,
) -> List[ForwardBackwardResponse]:
    service = ServiceClient()
    base_model = service.get_server_capabilities().base_models[0]
    training = service.create_lora_training_client(base_model, rank=4)
    tokenizer = training.tokenizer

    history: List[ForwardBackwardResponse] = []
    for step in range(iterations):
        sampler = training.save_weights_and_get_sampling_client(f"{loss_name}-step-{step}")
        batch, avg_reward = build_batch(
            tokenizer,
            sampler,
            specs,
            group_size=group_size,
            clip_epsilon=0.2,
        )
        if not batch:
            break

        fb = training.forward_backward(batch, loss_fn=loss_name).result()
        training.optim_step(AdamParams(lr=5e-3)).result()
        history.append(fb)

        kl_old_new = fb.metrics.get("kl_q||p", 0.0)
        kl_new_old = fb.metrics.get("kl_p||q", 0.0)
        print(
            f"[{loss_name}] step={step:02d} reward={avg_reward:.3f} "
            f"loss={fb.loss:.3f} kl_q||p={kl_old_new:.4f} kl_p||q={kl_new_old:.4f}"
        )

    return history


if __name__ == "__main__":
    seed_everything(1234)

    prompts = [
        PromptSpec(prompt="1+1=", answer="2"),
        PromptSpec(prompt="2+2=", answer="4"),
        PromptSpec(prompt="3+3=", answer="6"),
        PromptSpec(prompt="1+2=", answer="3"),
    ]

    print("Running PPO training loop")
    ppo_history = run_loop("ppo", specs=prompts, iterations=12)

    print("\nRunning importance sampling training loop")
    is_history = run_loop("importance_sampling", specs=prompts, iterations=12)

    print("\nFinal PPO reward metrics:")
    for idx, fb in enumerate(ppo_history[-3:]):
        step = len(ppo_history) - len(ppo_history[-3:]) + idx
        print(f"  step {step}: loss={fb.loss:.3f} metrics={fb.metrics}")

    print("\nFinal IS reward metrics:")
    for idx, fb in enumerate(is_history[-3:]):
        step = len(is_history) - len(is_history[-3:]) + idx
        print(f"  step {step}: loss={fb.loss:.3f} metrics={fb.metrics}")
