"""End-to-end PPO loop running on Ray actors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import ray
import torch

if __package__ is None or __package__ == "":
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from rinker.api.service_client import ServiceClient
from rinker.api.training_client import ForwardBackwardResponse
from rinker.core.types import AdamParams, Datum, ModelInput, SamplingParams
from rinker.ray_runtime import RayRuntimeConfig
from rinker.utils.seeding import seed_everything


def addition_reward(prompt: str, completion: str, target: str) -> float:
    return 1.0 if completion.strip().startswith(target) else 0.0


@dataclass
class PromptSpec:
    prompt: str
    answer: str


def build_datums(
    tokenizer,
    prompt: str,
    sample_text: str,
    logprobs: Sequence[float],
    clip_epsilon: float,
) -> Datum:
    token_ids = torch.tensor(tokenizer.encode(sample_text), dtype=torch.long)
    inputs = token_ids[:-1]
    targets = token_ids[1:]
    sampling_logprobs = torch.zeros_like(targets, dtype=torch.float32)
    prompt_len = len(tokenizer.encode(prompt)) - 1
    if logprobs:
        lp_tensor = torch.tensor(logprobs, dtype=torch.float32)
        sampling_logprobs[prompt_len : prompt_len + lp_tensor.numel()] = lp_tensor
    datum = Datum(
        model_input=ModelInput(token_chunks=[inputs]),
        loss_fn_inputs={
            "target_tokens": targets,
            "sampling_logprobs": sampling_logprobs,
            "advantages": torch.zeros_like(targets, dtype=torch.float32),
            "clip_epsilon": clip_epsilon,
        },
    )
    return datum


def ray_reward_evaluation(
    actor: ray.ActorHandle,
    prompt: str,
    completion: str,
    target: str,
) -> float:
    ref = actor.compute.remote(prompt, completion, target)
    return float(ray.get(ref))


def run_iteration(
    tokenizer,
    sampler,
    reward_actor,
    specs: Iterable[PromptSpec],
    *,
    group_size: int,
    clip_epsilon: float,
    sampling_params: SamplingParams,
) -> tuple[List[Datum], float]:
    batch: List[Datum] = []
    rewards: List[float] = []

    for spec in specs:
        prompt_tokens = torch.tensor(tokenizer.encode(spec.prompt), dtype=torch.long)
        model_input = ModelInput(token_chunks=[prompt_tokens])
        samples = sampler.sample(model_input, sampling_params=sampling_params, num_samples=group_size)

        group_rewards: List[float] = []
        group_data: List[Datum] = []
        for sample in samples:
            completion = sample.text[len(spec.prompt) :]
            reward = ray_reward_evaluation(reward_actor, spec.prompt, completion, spec.answer)
            group_rewards.append(reward)
            datum = build_datums(tokenizer, spec.prompt, sample.text, sample.logprobs, clip_epsilon)
            group_data.append(datum)

        if not group_rewards:
            continue

        group_tensor = torch.tensor(group_rewards, dtype=torch.float32)
        centred = group_tensor - group_tensor.mean()
        for datum, advantage in zip(group_data, centred):
            advantages_tensor = datum.loss_fn_inputs["advantages"]
            assert isinstance(advantages_tensor, torch.Tensor)
            prompt_len = len(tokenizer.encode(spec.prompt)) - 1
            advantages_tensor[prompt_len:] = advantage
            batch.append(datum)
            rewards.append(float(advantage + group_tensor.mean()))

    avg_reward = float(torch.tensor(rewards, dtype=torch.float32).mean()) if rewards else 0.0
    return batch, avg_reward


def run_loop(
    specs: Sequence[PromptSpec],
    *,
    iterations: int,
    group_size: int,
) -> List[ForwardBackwardResponse]:
    config = RayRuntimeConfig(num_sampler_actors=1, max_inflight_rollouts=4)
    service = ServiceClient(runtime_config=config)
    base_model = service.get_server_capabilities().base_models[0]
    training = service.create_lora_training_client(base_model, rank=4)
    tokenizer = training.tokenizer

    [reward_actor] = training.create_reward_actors([addition_reward])

    sampling_params = SamplingParams(max_new_tokens=4, temperature=0.8)
    history: List[ForwardBackwardResponse] = []

    for step in range(iterations):
        sampler = training.save_weights_and_get_sampling_client(f"ppo-step-{step}")
        batch, avg_reward = run_iteration(
            tokenizer,
            sampler,
            reward_actor,
            specs,
            group_size=group_size,
            clip_epsilon=0.2,
            sampling_params=sampling_params,
        )
        if not batch:
            break

        fb = training.forward_backward(batch, loss_fn="ppo").result()
        training.optim_step(AdamParams(lr=5e-3)).result()
        history.append(fb)

        print(
            f"[ray] step={step:02d} reward={avg_reward:.3f} loss={fb.loss:.3f} "
            f"kl_q||p={fb.metrics.get('kl_q||p', 0.0):.4f} "
            f"kl_p||q={fb.metrics.get('kl_p||q', 0.0):.4f}"
        )

    ray.shutdown()
    return history


if __name__ == "__main__":
    seed_everything(2024)

    prompts = [
        PromptSpec(prompt="1+1=", answer="2"),
        PromptSpec(prompt="2+2=", answer="4"),
        PromptSpec(prompt="3+3=", answer="6"),
        PromptSpec(prompt="4+4=", answer="8"),
    ]

    run_loop(prompts, iterations=6, group_size=2)
