"""Simple synchronous RL loop mirroring the Tinker cookbook example."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import torch

from ..api.service_client import ServiceClient
from ..core.engine import SimpleLanguageModel
from ..core.types import AdamParams, Datum, ModelInput, SamplingParams
from ..rl import Env, EnvAction, EnvGroupBuilder, EnvObservation, RLDataset
from ..utils.seeding import seed_everything


@dataclass
class MathExample:
    prompt: str
    answer: str


class MathQAEnv(Env):
    """One-step environment that scores arithmetic answers."""

    def __init__(self, example: MathExample, tokenizer) -> None:
        prompt_tokens = torch.tensor(tokenizer.encode(example.prompt), dtype=torch.long)
        self._observation = EnvObservation(model_input=ModelInput(token_chunks=[prompt_tokens]))
        self._prompt = example.prompt
        self._answer = example.answer

    def initial_observation(self) -> EnvObservation:
        return self._observation

    def step(self, action: EnvAction):
        text = action.text or ""
        completion = text[len(self._prompt) :].strip()
        reward = 1.0 if completion.startswith(self._answer) else 0.0
        return self._terminal(reward)

    def _terminal(self, reward: float):
        from ..rl.env import EnvStepResult

        return EnvStepResult(reward=reward, done=True, metrics={"is_correct": reward})


def _compute_reference_model(training_client) -> SimpleLanguageModel:
    runtime = training_client._runtime  # type: ignore[attr-defined]
    state_dict = runtime.get_state()
    model = SimpleLanguageModel(runtime.tokenizer.vocab_size)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _logprob_under(model: SimpleLanguageModel, token_ids: List[int]) -> torch.Tensor:
    if len(token_ids) <= 1:
        return torch.empty(0, dtype=torch.float32)
    with torch.no_grad():
        inputs = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0)
        targets = torch.tensor(token_ids[1:], dtype=torch.long)
        logits = model(inputs)
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, targets.unsqueeze(0).unsqueeze(-1)).squeeze(0).squeeze(-1)
        return gathered


def run_loop(
    *,
    batch_size: int = 4,
    group_size: int = 4,
    iterations: int = 12,
    loss_name: str = "ppo",
    beta_kl: float = 0.0,
) -> None:
    service = ServiceClient()
    base_model = service.get_server_capabilities().base_models[0]
    training = service.create_lora_training_client(base_model, rank=4)
    tokenizer = training.tokenizer

    reference_model = _compute_reference_model(training)

    examples = [
        MathExample("1+1=", "2"),
        MathExample("2+2=", "4"),
        MathExample("3+3=", "6"),
        MathExample("1+2=", "3"),
        MathExample("3+4=", "7"),
        MathExample("4+5=", "9"),
    ]
    envs = [MathQAEnv(example, tokenizer) for example in examples]
    builder = EnvGroupBuilder(envs)

    sampling_params = SamplingParams(max_new_tokens=4, temperature=0.7)
    optimiser = AdamParams(lr=5e-3)

    for step in range(iterations):
        dataset = RLDataset()
        sampler = training.save_weights_and_get_sampling_client(f"simple-loop-{step}")
        groups = builder.build(batch_size)

        start = time.time()
        for group in groups:
            prompt_tokens = group.observation.model_input.token_chunks[0]
            results = sampler.sample(group.observation.model_input, sampling_params, num_samples=group_size)
            for sample in results:
                token_ids = torch.tensor(sample.token_ids, dtype=torch.long)
                inputs = token_ids[:-1]
                targets = token_ids[1:]
                sampling_logprobs = torch.zeros_like(targets, dtype=torch.float32)
                completion_start = prompt_tokens.numel() - 1
                generated_logprobs = torch.tensor(sample.logprobs, dtype=torch.float32)
                sampling_logprobs[
                    completion_start : completion_start + generated_logprobs.numel()
                ] = generated_logprobs

                datum = Datum(
                    model_input=ModelInput(token_chunks=[inputs]),
                    loss_fn_inputs={
                        "target_tokens": targets,
                        "sampling_logprobs": sampling_logprobs,
                        "advantages": torch.zeros_like(targets, dtype=torch.float32),
                        "clip_epsilon": 0.2,
                    },
                    policy_version=getattr(sample, "weights_version", None),
                )

                mask = torch.zeros_like(targets, dtype=torch.bool)
                mask[completion_start:] = True

                env_action = EnvAction(
                    token_ids=sample.token_ids,
                    logprobs=sample.logprobs,
                    text=sample.text,
                )
                transition = group.env.step(env_action)

                ref_logprobs = _logprob_under(reference_model, sample.token_ids)
                ref_slice = ref_logprobs[
                    completion_start : completion_start + generated_logprobs.numel()
                ]
                kl_value = float((generated_logprobs - ref_slice).sum().item())

                dataset.add_sample(
                    group_id=group.group_index,
                    datum=datum,
                    reward=float(transition.reward),
                    token_count=int(mask.sum().item()),
                    advantage_mask=mask,
                    kl_value=kl_value,
                    metrics=transition.metrics,
                    policy_version=getattr(sample, "weights_version", None),
                )
        elapsed = time.time() - start

        dataset.apply_kl_shaping(beta_kl)
        dataset.apply_group_centered_advantages()

        for substep in range(1):
            fb = training.forward_backward(dataset.build_batch(), loss_fn=loss_name).result()
            training.optim_step(optimiser).result()

        reward_mean = dataset.mean_reward()
        reward_raw = dataset.mean_reward(shaped=False)
        tokens_per_second = dataset.total_tokens / max(elapsed, 1e-6)
        acceptance = dataset.acceptance_rate()
        kl_qp = fb.metrics.get("kl_q||p", 0.0)
        kl_pq = fb.metrics.get("kl_p||q", 0.0)
        clip_fraction = fb.metrics.get("clip_fraction", 0.0)
        print(
            f"step={step:02d} reward_mean={reward_mean:.3f} raw={reward_raw:.3f} "
            f"kl_q||p={kl_qp:.4f} kl_p||q={kl_pq:.4f} "
            f"accept={acceptance:.2f} clip={clip_fraction:.2f} tps={tokens_per_second:.1f}"
        )


if __name__ == "__main__":
    seed_everything(1234)
    run_loop()
