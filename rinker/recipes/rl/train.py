"""Performant RL training loop with group-centred advantages and KL shaping."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import torch

from ...api.service_client import ServiceClient
from ...core.engine import SimpleLanguageModel
from ...core.types import AdamParams, Datum, ModelInput, SamplingParams
from ...eval.inline import InlineEvaluator
from ...rl import Env, EnvAction, EnvGroupBuilder, EnvObservation, RLDataset
from ...utils.checkpointing import CheckpointManager
from ...utils.seeding import seed_everything


__all__ = [
    "MathTask",
    "MathEnv",
    "build_envs",
    "build_reference_model",
    "LoopConfig",
    "RLTrainer",
    "main",
]


@dataclass
class MathTask:
    prompt: str
    answer: str


class MathEnv(Env):
    """Deterministic environment rewarding exact arithmetic answers."""

    def __init__(self, task: MathTask, tokenizer) -> None:
        prompt_tokens = torch.tensor(tokenizer.encode(task.prompt), dtype=torch.long)
        self._observation = EnvObservation(
            model_input=ModelInput(token_chunks=[prompt_tokens]),
            metadata={"prompt": task.prompt},
        )
        self._prompt = task.prompt
        self._answer = task.answer

    def initial_observation(self) -> EnvObservation:
        return self._observation

    def step(self, action: EnvAction):
        text = action.text or ""
        completion = text[len(self._prompt) :].strip()
        reward = 1.0 if completion.startswith(self._answer) else 0.0
        from ...rl.env import EnvStepResult

        return EnvStepResult(
            reward=reward,
            done=True,
            metrics={"is_correct": reward},
        )


def build_reference_model(training_client) -> SimpleLanguageModel:
    runtime = training_client._runtime  # type: ignore[attr-defined]
    state_dict = runtime.get_state()
    model = SimpleLanguageModel(runtime.tokenizer.vocab_size)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _sequence_logprobs(model: SimpleLanguageModel, token_ids: Sequence[int]) -> torch.Tensor:
    if len(token_ids) <= 1:
        return torch.empty(0, dtype=torch.float32)
    with torch.no_grad():
        inputs = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0)
        targets = torch.tensor(token_ids[1:], dtype=torch.long)
        logits = model(inputs)
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, targets.unsqueeze(0).unsqueeze(-1)).squeeze(0).squeeze(-1)
        return gathered


def build_envs(tokenizer, tasks: Iterable[MathTask]) -> List[Env]:
    """Constructs RL environments for the provided arithmetic tasks."""

    return [MathEnv(task, tokenizer) for task in tasks]


def _prepare_datum(
    *,
    sample,
    prompt_tokens: torch.Tensor,
) -> tuple[Datum, torch.Tensor, torch.Tensor]:
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
    return datum, mask, generated_logprobs


def _kl_value(
    *,
    reference_logprobs: torch.Tensor,
    generated_logprobs: torch.Tensor,
    completion_start: int,
) -> float:
    if generated_logprobs.numel() == 0:
        return 0.0
    ref_slice = reference_logprobs[completion_start : completion_start + generated_logprobs.numel()]
    if ref_slice.numel() != generated_logprobs.numel():
        ref_slice = ref_slice[: generated_logprobs.numel()]
    return float((generated_logprobs - ref_slice).sum().item())


@dataclass
class LoopConfig:
    iterations: int
    batch_size: int
    group_size: int
    num_substeps: int
    beta_kl: float
    loss_name: str
    learning_rate: float


class RLTrainer:
    def __init__(
        self,
        *,
        training_client,
        reference_model: SimpleLanguageModel,
        config: LoopConfig,
        checkpoint_manager: CheckpointManager | None = None,
        inline_evaluator: InlineEvaluator | None = None,
        training_config: Mapping[str, object] | None = None,
    ) -> None:
        self._training = training_client
        self._reference_model = reference_model
        self._config = config
        self._optim = AdamParams(lr=config.learning_rate)
        self._checkpoint_manager = checkpoint_manager
        self._inline_evaluator = inline_evaluator
        self._training_config = training_config

    def run(self, builder: EnvGroupBuilder, sampling_params: SamplingParams) -> None:
        for step in range(self._config.iterations):
            dataset = RLDataset()
            sampler = self._training.save_weights_and_get_sampling_client(f"rl-train-{step}")
            groups = builder.build(self._config.batch_size)

            start = time.time()
            for group in groups:
                prompt_tokens = group.observation.model_input.token_chunks[0]
                results = sampler.sample(
                    group.observation.model_input,
                    sampling_params,
                    num_samples=self._config.group_size,
                )
                for sample in results:
                    datum, mask, generated_logprobs = _prepare_datum(
                        sample=sample,
                        prompt_tokens=prompt_tokens,
                    )
                    env_action = EnvAction(
                        token_ids=sample.token_ids,
                        logprobs=sample.logprobs,
                        text=sample.text,
                    )
                    transition = group.env.step(env_action)
                    ref_logprobs = _sequence_logprobs(self._reference_model, sample.token_ids)
                    kl_val = _kl_value(
                        reference_logprobs=ref_logprobs,
                        generated_logprobs=generated_logprobs,
                        completion_start=prompt_tokens.numel() - 1,
                    )
                    dataset.add_sample(
                        group_id=group.group_index,
                        datum=datum,
                        reward=float(transition.reward),
                        token_count=int(mask.sum().item()),
                        advantage_mask=mask,
                        kl_value=kl_val,
                        metrics=transition.metrics,
                        policy_version=getattr(sample, "weights_version", None),
                    )
            elapsed = time.time() - start

            dataset.apply_kl_shaping(self._config.beta_kl)
            dataset.apply_group_centered_advantages()

            metrics = self._run_updates(dataset)

            reward_mean = dataset.mean_reward()
            reward_raw = dataset.mean_reward(shaped=False)
            acceptance = dataset.acceptance_rate()
            tokens_per_second = dataset.total_tokens / max(elapsed, 1e-6)
            summary_metrics = dataset.metrics_summary()
            kl_avg = dataset.average_kl()

            print(
                "step={step:03d} reward={reward:.3f} raw={raw:.3f} kl={kl:.4f} "
                "accept={accept:.2f} tokens/s={tps:.1f} kl_q||p={kl_qp:.4f} "
                "kl_p||q={kl_pq:.4f} clip={clip:.2f}".format(
                    step=step,
                    reward=reward_mean,
                    raw=reward_raw,
                    kl=kl_avg,
                    accept=acceptance,
                    tps=tokens_per_second,
                    kl_qp=metrics.get("kl_q||p", 0.0),
                    kl_pq=metrics.get("kl_p||q", 0.0),
                    clip=metrics.get("clip_fraction", 0.0),
                )
            )
            if summary_metrics:
                extras = " ".join(f"{k}={v:.3f}" for k, v in sorted(summary_metrics.items()))
                print(f"    metrics: {extras}")

            if self._checkpoint_manager is not None:
                self._checkpoint_manager.maybe_save(
                    step,
                    training_config=self._training_config,
                )

            if self._inline_evaluator is not None:
                inline = self._inline_evaluator.maybe_run(step, training_client=self._training)
                if inline is not None:
                    metric_summary = " ".join(
                        f"{k}={v:.3f}" for k, v in sorted(inline.metrics.items())
                    )
                    if metric_summary:
                        metric_summary = f" metrics[{metric_summary}]"
                    print(
                        "    inline_eval: reward={reward:.3f} std={std:.3f} accept={accept:.2f}{metrics}".format(
                            reward=inline.reward_mean,
                            std=inline.reward_std,
                            accept=inline.acceptance,
                            metrics=metric_summary,
                        )
                    )

    def _run_updates(self, dataset: RLDataset) -> Mapping[str, float]:
        schedule = self._training.runtime_config.stream_minibatch
        if schedule is not None:
            responses = self._training.stream_minibatch_train(
                dataset,
                loss_fn=self._config.loss_name,
                optimiser=self._optim,
                config=schedule,
            )
            return responses[-1].metrics if responses else {}
        batch = dataset.build_batch()
        aggregated: Mapping[str, float] = {}
        for _ in range(self._config.num_substeps):
            fb = self._training.forward_backward(batch, loss_fn=self._config.loss_name).result()
            self._training.optim_step(self._optim).result()
            aggregated = fb.metrics
        return aggregated


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RL training loop with group advantages")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--num-substeps", type=int, default=1)
    parser.add_argument("--beta-kl", type=float, default=0.0)
    parser.add_argument("--loss", type=str, default="ppo", choices=["ppo", "importance_sampling"])
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    service = ServiceClient()
    base_model = service.get_server_capabilities().base_models[0]
    training = service.create_lora_training_client(base_model, rank=4)
    tokenizer = training.tokenizer

    reference = build_reference_model(training)

    tasks = [
        MathTask("1+1=", "2"),
        MathTask("2+2=", "4"),
        MathTask("3+3=", "6"),
        MathTask("4+4=", "8"),
        MathTask("5+5=", "10"),
        MathTask("6+3=", "9"),
        MathTask("7+2=", "9"),
        MathTask("8+1=", "9"),
    ]
    envs = build_envs(tokenizer, tasks)
    builder = EnvGroupBuilder(envs)

    config = LoopConfig(
        iterations=args.iterations,
        batch_size=args.batch_size,
        group_size=args.group_size,
        num_substeps=args.num_substeps,
        beta_kl=args.beta_kl,
        loss_name=args.loss,
        learning_rate=args.lr,
    )
    trainer = RLTrainer(training_client=training, reference_model=reference, config=config)

    sampling_params = SamplingParams(max_new_tokens=4, temperature=0.7)
    trainer.run(builder, sampling_params)


if __name__ == "__main__":
    main()
