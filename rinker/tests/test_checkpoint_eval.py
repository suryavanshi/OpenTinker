from __future__ import annotations

import json
from pathlib import Path

import torch

from rinker.core.types import ModelInput, SamplingParams
from rinker.eval import InlineEvaluator, OfflineEvaluator
from rinker.rl import Env, EnvAction, EnvGroupBuilder, EnvObservation
from rinker.utils.checkpointing import CheckpointManager
from rinker.utils.tokenizer import SimpleTokenizer


class DummyTrainingClient:
    def __init__(self) -> None:
        self.state = {"model": {}}
        self.saved = []
        self.current_reward = 1.0

    def save_state(self):
        payload = {
            "model": {"weights": torch.tensor([1.0])},
            "optimiser": {"state": 1},
            "grad_scaler": None,
            "global_step": len(self.saved) + 1,
            "accumulation_progress": 0,
        }
        self.saved.append(payload)
        return payload

    def export_lora_weights(self):
        return {"adapters": {"layer": torch.tensor([1.0])}, "merged": {"layer": torch.tensor([2.0])}}

    def save_weights_and_get_sampling_client(self, name):
        return DummySampler(self)

    def load_state(self, state):
        self.current_reward = state.get("reward", 0.0)


class DummySampler:
    def __init__(self, training_client: DummyTrainingClient) -> None:
        self._training = training_client

    def sample(self, model_input, sampling_params, num_samples):
        reward = self._training.current_reward
        text = f"reward={reward}"
        return [DummySample(text) for _ in range(num_samples)]


class DummySample:
    def __init__(self, text: str) -> None:
        self.text = text
        self.token_ids = [0, 1]
        self.logprobs = [0.0]


class ConstantEnv(Env):
    def __init__(self) -> None:
        tokens = torch.tensor([0, 1], dtype=torch.long)
        self._observation = EnvObservation(model_input=ModelInput(token_chunks=[tokens]))

    def initial_observation(self):
        return self._observation

    def step(self, action: EnvAction):
        from rinker.rl.env import EnvStepResult

        text = action.text or "reward=0.0"
        try:
            reward = float(text.split("=", 1)[1])
        except (IndexError, ValueError):  # pragma: no cover - defensive fallback
            reward = 0.0
        return EnvStepResult(reward=reward, done=True, metrics={"accuracy": float(reward > 0.0)})


def test_checkpoint_manager_writes_expected_files(tmp_path: Path):
    client = DummyTrainingClient()
    tokenizer = SimpleTokenizer()
    manager = CheckpointManager(
        training_client=client,
        tokenizer=tokenizer,
        output_dir=tmp_path,
        every_steps=1,
        keep_last=2,
    )
    config = {"train": {"epochs": 1}}

    for step in range(3):
        state = manager.maybe_save(step, training_config=config)
        assert state is not None

    checkpoints = sorted(tmp_path.glob("step_*"))
    assert len(checkpoints) == 2
    latest = checkpoints[-1]
    assert (latest / "adapter.safetensors").exists()
    assert (latest / "trainer_state.pt").exists()
    assert json.loads((latest / "metadata.json").read_text())
    assert json.loads((latest / "tokenizer.json").read_text())
    assert (latest / "config.yaml").exists()


def test_inline_evaluator_records_metrics(tmp_path: Path):
    envs = [ConstantEnv() for _ in range(2)]
    builder = EnvGroupBuilder(envs)
    evaluator = InlineEvaluator(
        builder=builder,
        sampling_params=SamplingParams(max_new_tokens=2, temperature=0.5),
        every_steps=1,
        num_env_groups=2,
        group_size=2,
        output_dir=tmp_path,
    )
    client = DummyTrainingClient()
    result = evaluator.maybe_run(0, training_client=client)
    assert result is not None
    assert result.reward_mean == 1.0
    assert result.acceptance == 1.0
    assert "accuracy" in result.metrics
    csv_contents = (tmp_path / "inline_eval.csv").read_text()
    assert "reward_mean" in csv_contents


def test_offline_evaluator_reads_checkpoints(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for step, reward in enumerate([0.1, 0.5, 0.9], start=1):
        directory = checkpoint_dir / f"step_{step:06d}"
        directory.mkdir()
        torch.save({"global_step": step, "reward": reward}, directory / "trainer_state.pt")
    envs = [ConstantEnv()]
    builder = EnvGroupBuilder(envs)
    evaluator = OfflineEvaluator(
        builder=builder,
        sampling_params=SamplingParams(max_new_tokens=2, temperature=0.5),
        num_env_groups=1,
        group_size=1,
        output_dir=tmp_path / "offline",
    )
    client = DummyTrainingClient()
    summary = evaluator.run(training_client=client, checkpoint_root=checkpoint_dir)
    assert summary.results
    assert summary.csv_path.exists()
    if summary.plot_path is not None:
        assert summary.plot_path.exists()
