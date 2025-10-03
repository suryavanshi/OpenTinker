"""Toy RLHF example using a fixed reward model and PPO updates."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

from ..api.service_client import ServiceClient
from ..core.types import ModelInput, SamplingParams
from ..recipes.rl.train import LoopConfig, RLTrainer, build_reference_model
from ..rl import EnvGroupBuilder
from ..rl.env import Env, EnvAction, EnvObservation, EnvStepResult
from ..utils.seeding import seed_everything


__all__ = [
    "ToyRewardModel",
    "ToyPreferenceEnv",
    "build_preference_envs",
    "run",
    "main",
]


@dataclass
class ToyRewardModel:
    """Deterministic reward model that scores responses with simple features."""

    positive_weight: float = 2.5
    refusal_penalty: float = -3.0
    apology_penalty: float = -1.5
    length_penalty: float = -0.02
    bias: float = -0.2

    def score(self, prompt: str, completion: str) -> float:
        text = completion.lower()
        positive = sum(text.count(token) for token in ["glad", "sure", "help", "happy"])
        refusal = sum(text.count(token) for token in ["cannot", "can't", "refuse", "decline"])
        apology = text.count("sorry")
        tokens = max(len(completion.split()), 1)
        value = (
            self.bias
            + self.positive_weight * positive
            + self.refusal_penalty * refusal
            + self.apology_penalty * apology
            + self.length_penalty * tokens
        )
        return float(1.0 / (1.0 + math.exp(-value)))


class ToyPreferenceEnv(Env):
    """Environment that returns the reward model score as the RL reward."""

    def __init__(self, prompt: str, tokenizer, reward_model: ToyRewardModel) -> None:
        prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
        self._observation = EnvObservation(
            model_input=ModelInput(token_chunks=[prompt_tokens]),
            metadata={"prompt": prompt},
        )
        self._prompt = prompt
        self._reward_model = reward_model

    def initial_observation(self) -> EnvObservation:
        return self._observation

    def step(self, action: EnvAction) -> EnvStepResult:
        text = action.text or ""
        completion = text[len(self._prompt) :]
        score = self._reward_model.score(self._prompt, completion)
        metrics = {"rm_score": score, "length": float(len(completion.split()))}
        return EnvStepResult(reward=score, metrics=metrics, done=True)


def build_preference_envs(tokenizer, prompts: Iterable[str], reward_model: ToyRewardModel):
    return [ToyPreferenceEnv(prompt, tokenizer, reward_model) for prompt in prompts]


def run(
    *,
    iterations: int,
    batch_size: int,
    group_size: int,
    num_substeps: int,
    beta_kl: float,
    learning_rate: float,
    seed: int,
) -> None:
    seed_everything(seed)
    service = ServiceClient()
    base_model = service.get_server_capabilities().base_models[0]
    training = service.create_lora_training_client(base_model, rank=8)
    reference = build_reference_model(training)

    tokenizer = training.tokenizer
    reward_model = ToyRewardModel()
    prompts = [
        "User: Please give me an encouraging update about our project. Assistant:",
        "User: Can you summarise the latest sprint retro with positive framing? Assistant:",
        "User: Draft a helpful follow-up email for the customer demo. Assistant:",
    ]
    envs = build_preference_envs(tokenizer, prompts, reward_model)
    builder = EnvGroupBuilder(envs)
    config = LoopConfig(
        iterations=iterations,
        batch_size=batch_size,
        group_size=group_size,
        num_substeps=num_substeps,
        beta_kl=beta_kl,
        loss_name="ppo",
        learning_rate=learning_rate,
    )
    trainer = RLTrainer(
        training_client=training,
        reference_model=reference,
        config=config,
    )
    sampling_params = SamplingParams(max_new_tokens=64, temperature=0.8, top_p=0.9)
    trainer.run(builder, sampling_params)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Toy RLHF recipe with PPO over RM scores")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--num-substeps", type=int, default=1)
    parser.add_argument("--beta-kl", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args(argv)
    run(
        iterations=args.iterations,
        batch_size=args.batch_size,
        group_size=args.group_size,
        num_substeps=args.num_substeps,
        beta_kl=args.beta_kl,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
