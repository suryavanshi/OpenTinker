"""Reproduce the Tinker RLVR math recipe with unit-test style rewards."""
from __future__ import annotations

import argparse
import ast
import operator
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch

from ..api.service_client import ServiceClient
from ..core.types import SamplingParams
from ..recipes.rl.train import (
    LoopConfig,
    MathEnv,
    MathTask,
    RLTrainer,
    build_reference_model,
)
from ..rl.env import EnvAction, EnvObservation, EnvStepResult
from ..rl import EnvGroupBuilder
from ..utils.seeding import seed_everything


__all__ = [
    "UnitTestReward",
    "UnitTestMathEnv",
    "build_unit_test_envs",
    "run",
    "main",
]


_ALLOWED_OPS: Mapping[type[ast.AST], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval(expr: str) -> int:
    node = ast.parse(expr, mode="eval").body

    def _eval(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return int(node.value)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            return int(_ALLOWED_OPS[op_type](left, right))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    return _eval(node)


@dataclass
class UnitTestReward:
    """Callable implementing unit-test style verification of math answers."""

    expected_expression: str

    def __call__(self, completion: str) -> tuple[float, Mapping[str, float]]:
        expected = _safe_eval(self.expected_expression)
        token = completion.strip().split()
        candidate = token[0] if token else ""
        try:
            predicted = int(candidate)
        except ValueError:
            return 0.0, {"unit_test_passed": 0.0}
        reward = 1.0 if predicted == expected else 0.0
        return reward, {"unit_test_passed": reward}


class UnitTestMathEnv(MathEnv):
    """Math environment whose reward is derived from unit test checks."""

    def __init__(self, task: MathTask, tokenizer) -> None:
        super().__init__(task, tokenizer)
        self._tester = UnitTestReward(task.prompt.split("=")[0])

    def initial_observation(self) -> EnvObservation:  # pragma: no cover - reuse parent
        return super().initial_observation()

    def step(self, action: EnvAction) -> EnvStepResult:
        text = action.text or ""
        completion = text[len(self._prompt) :]
        reward, metrics = self._tester(completion)
        return EnvStepResult(reward=reward, metrics=dict(metrics), done=True)


def build_unit_test_envs(tokenizer, tasks: Iterable[MathTask]):
    return [UnitTestMathEnv(task, tokenizer) for task in tasks]


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
    training = service.create_lora_training_client(base_model, rank=4)
    reference = build_reference_model(training)

    tokenizer = training.tokenizer
    tasks = [
        MathTask("1+4=", "5"),
        MathTask("12-7=", "5"),
        MathTask("3*3=", "9"),
        MathTask("18//3=", "6"),
        MathTask("10%3=", "1"),
        MathTask("5+7=", "12"),
    ]
    envs = build_unit_test_envs(tokenizer, tasks)
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
    sampling_params = SamplingParams(max_new_tokens=4, temperature=0.7)
    trainer.run(builder, sampling_params)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RLVR math recipe with unit tests")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--num-substeps", type=int, default=1)
    parser.add_argument("--beta-kl", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=1234)
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
