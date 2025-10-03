"""Implementation of the ``rinker`` command line interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

from ..api.service_client import ServiceClient
from ..core.types import Datum, ModelInput, SamplingParams
from ..recipes.rl.train import (
    LoopConfig,
    MathTask,
    RLTrainer,
    build_envs,
    build_reference_model,
)
from ..rl import EnvGroupBuilder
from ..utils.checkpointing import CheckpointManager
from ..utils.seeding import seed_everything
from ..utils.tokenizer import SimpleTokenizer
from .config import (
    CheckpointConfig,
    InlineEvalConfig,
    RLConfig,
    build_eval_config,
    build_runtime_config,
    build_train_config,
    load_config,
)
from ..eval import InlineEvaluator, OfflineEvaluator


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rinker", description="Training and evaluation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("mode", choices=["sl", "rl"], help="Training mode")
    train_parser.add_argument("-c", "--config", required=True, help="Path to the training config")
    train_parser.set_defaults(func=_cmd_train)

    eval_parser = subparsers.add_parser("eval", help="Run offline evaluation sweeps")
    eval_parser.add_argument("-c", "--config", required=True, help="Config describing the evaluation setup")
    eval_parser.add_argument("--checkpoints", required=True, help="Directory containing saved checkpoints")
    eval_parser.set_defaults(func=_cmd_eval)

    export_parser = subparsers.add_parser("export", help="Export checkpoints to Hugging Face format")
    export_parser.add_argument("--checkpoint", required=True, help="Checkpoint directory to export")
    export_parser.add_argument("--output", required=True, help="Output directory for Hugging Face artefacts")
    export_parser.add_argument("--base-model", help="Optional override for the base model")
    export_parser.add_argument("--lora-rank", type=int, help="Optional override for the LoRA rank")
    export_parser.set_defaults(func=_cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


# ---------------------------------------------------------------------------
# Train command helpers
# ---------------------------------------------------------------------------

def _cmd_train(args) -> int:
    config_path = Path(args.config).resolve()
    raw_config = load_config(config_path)
    train_cfg = build_train_config(raw_config)

    runtime_config = build_runtime_config(train_cfg.runtime)
    service = ServiceClient(runtime_config=runtime_config)
    capabilities = service.get_server_capabilities()
    base_model = train_cfg.base_model or capabilities.base_models[0]
    seed_everything(train_cfg.seed)

    training = service.create_lora_training_client(
        base_model,
        train_cfg.lora_rank,
    )

    if args.mode == "sl":
        if train_cfg.sl is None:
            raise ValueError("Training config is missing the 'sl' section")
        _run_sl_training(
            train_cfg.sl,
            training,
            raw_config,
            config_path,
            checkpoint_cfg=train_cfg.checkpoint,
        )
    else:
        if train_cfg.rl is None:
            raise ValueError("Training config is missing the 'rl' section")
        _run_rl_training(
            train_cfg.rl,
            training,
            raw_config,
            config_path,
            checkpoint_cfg=train_cfg.checkpoint,
            eval_cfg=train_cfg.rl.eval or train_cfg.eval,
        )
    return 0


def _resolve_path(config_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return (config_path.parent / path).resolve()
    return path


def _run_sl_training(sl_cfg, training, raw_config: Mapping[str, Any], config_path: Path, *, checkpoint_cfg: CheckpointConfig | None) -> None:
    dataset_cfg = sl_cfg.dataset
    if len(dataset_cfg.prompts) != len(dataset_cfg.completions):
        raise ValueError("Dataset prompts and completions must have the same length")

    tokenizer = training.tokenizer
    datums: list[Datum] = []
    for prompt, completion in zip(dataset_cfg.prompts, dataset_cfg.completions):
        full_text = prompt + completion
        token_ids = torch.tensor(tokenizer.encode(full_text), dtype=torch.long)
        inputs = token_ids[:-1]
        targets = token_ids[1:]
        datums.append(
            Datum(
                model_input=ModelInput(token_chunks=[inputs]),
                loss_fn_inputs={
                    "targets": targets,
                    "weights": torch.ones_like(targets, dtype=torch.float32),
                },
            )
        )

    optimiser = sl_cfg.optimiser
    checkpoint_manager = _build_checkpoint_manager(
        checkpoint_cfg,
        training,
        tokenizer,
        config_path,
        raw_config,
    )

    for epoch in range(sl_cfg.epochs):
        fb = training.forward_backward(datums, loss_fn=sl_cfg.loss).result()
        metrics = training.optim_step(optimiser).result()
        print(f"epoch={epoch:03d} loss={fb.loss:.4f} grad_norm={metrics.get('grad_norm', 0.0):.4f}")
        if checkpoint_manager is not None:
            checkpoint_manager.maybe_save(epoch, training_config=raw_config)

    sampler = training.save_weights_and_get_sampling_client("sl-train")
    if datums:
        sampling_params = SamplingParams(max_new_tokens=32, temperature=0.8)
        sample = sampler.sample(datums[0].model_input, sampling_params, num_samples=1)
        if sample:
            print("Sample:", sample[0].text)


def _build_checkpoint_manager(
    checkpoint_cfg: CheckpointConfig | None,
    training,
    tokenizer: SimpleTokenizer,
    config_path: Path,
    raw_config: Mapping[str, Any],
):
    if checkpoint_cfg is None or checkpoint_cfg.every_steps <= 0:
        return None
    output_dir = _resolve_path(config_path, checkpoint_cfg.dir)
    if output_dir is None:
        return None
    return CheckpointManager(
        training_client=training,
        tokenizer=tokenizer,
        output_dir=output_dir,
        every_steps=checkpoint_cfg.every_steps,
        keep_last=checkpoint_cfg.keep_last,
        save_optimizer=checkpoint_cfg.save_optimizer,
        save_tokenizer=checkpoint_cfg.save_tokenizer,
        save_config=checkpoint_cfg.save_config,
    )


def _run_rl_training(
    rl_cfg: RLConfig,
    training,
    raw_config: Mapping[str, Any],
    config_path: Path,
    *,
    checkpoint_cfg: CheckpointConfig | None,
    eval_cfg: InlineEvalConfig | None,
) -> None:
    tokenizer = training.tokenizer
    tasks = [MathTask(task.prompt, task.answer) for task in rl_cfg.tasks]
    training_envs = build_envs(tokenizer, tasks)
    eval_envs = build_envs(tokenizer, tasks)
    builder = EnvGroupBuilder(training_envs)

    loop = rl_cfg.loop
    loop_config = LoopConfig(
        iterations=loop.iterations,
        batch_size=loop.batch_size,
        group_size=loop.group_size,
        num_substeps=loop.num_substeps,
        beta_kl=loop.beta_kl,
        loss_name=loop.loss,
        learning_rate=loop.learning_rate,
    )

    checkpoint_manager = _build_checkpoint_manager(
        checkpoint_cfg,
        training,
        tokenizer,
        config_path,
        raw_config,
    )

    inline_evaluator = _build_inline_evaluator(
        eval_cfg,
        eval_envs,
        config_path,
    )

    reference_model = build_reference_model(training)
    sampling_params = rl_cfg.sampling.to_params()
    trainer = RLTrainer(
        training_client=training,
        reference_model=reference_model,
        config=loop_config,
        checkpoint_manager=checkpoint_manager,
        inline_evaluator=inline_evaluator,
        training_config=raw_config,
    )
    trainer.run(builder, sampling_params)


def _build_inline_evaluator(
    eval_cfg: InlineEvalConfig | None,
    envs,
    config_path: Path,
):
    if eval_cfg is None or eval_cfg.every_steps <= 0:
        return None
    builder = EnvGroupBuilder(envs)
    sampling_params = eval_cfg.sampling.to_params()
    output_dir = _resolve_path(config_path, eval_cfg.output_dir)
    return InlineEvaluator(
        builder=builder,
        sampling_params=sampling_params,
        every_steps=eval_cfg.every_steps,
        num_env_groups=eval_cfg.num_env_groups or len(envs),
        group_size=eval_cfg.group_size or 1,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Eval command helpers
# ---------------------------------------------------------------------------

def _cmd_eval(args) -> int:
    config_path = Path(args.config).resolve()
    raw_config = load_config(config_path)
    eval_cfg = build_eval_config(raw_config)
    if eval_cfg.rl is None:
        raise ValueError("Evaluation config requires an 'rl' section")

    runtime_config = build_runtime_config(eval_cfg.runtime)
    service = ServiceClient(runtime_config=runtime_config)
    capabilities = service.get_server_capabilities()
    base_model = eval_cfg.base_model or capabilities.base_models[0]
    training = service.create_lora_training_client(base_model, eval_cfg.lora_rank)

    rl_cfg = eval_cfg.rl
    tasks = [MathTask(task.prompt, task.answer) for task in rl_cfg.tasks]
    envs = build_envs(training.tokenizer, tasks)
    builder = EnvGroupBuilder(envs)

    offline_cfg = rl_cfg.offline_eval or rl_cfg.eval
    if offline_cfg is None:
        raise ValueError("Evaluation config must define offline_eval or eval settings")
    sampling_params = offline_cfg.sampling.to_params() if offline_cfg.sampling else rl_cfg.sampling.to_params()
    output_dir = _resolve_path(config_path, offline_cfg.output_dir)
    evaluator = OfflineEvaluator(
        builder=builder,
        sampling_params=sampling_params,
        num_env_groups=offline_cfg.num_env_groups or len(envs),
        group_size=offline_cfg.group_size or 1,
        output_dir=output_dir,
    )

    checkpoint_root = Path(args.checkpoints).resolve()
    summary = evaluator.run(training_client=training, checkpoint_root=checkpoint_root)
    print(f"Offline evaluation summary written to {summary.csv_path}")
    if summary.plot_path is not None:
        print(f"Reward plot saved to {summary.plot_path}")
    return 0


# ---------------------------------------------------------------------------
# Export command helpers
# ---------------------------------------------------------------------------

def _cmd_export(args) -> int:
    checkpoint_dir = Path(args.checkpoint).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint does not contain trainer_state.pt: {state_path}")
    state = torch.load(state_path, map_location="cpu")

    service = ServiceClient()
    capabilities = service.get_server_capabilities()
    base_model = args.base_model or capabilities.base_models[0]
    lora_rank = args.lora_rank or 4
    training = service.create_lora_training_client(base_model, lora_rank)
    training.load_state(state)

    export = training.export_lora_weights()
    merged_path = output_dir / "pytorch_model.bin"
    adapter_path = output_dir / "adapter_model.safetensors"
    torch.save(export["merged"], merged_path)
    save_file(export["adapters"], str(adapter_path))

    metadata = {
        "base_model": base_model,
        "lora_rank": lora_rank,
        "source_checkpoint": str(checkpoint_dir),
    }
    with (output_dir / "export_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if tokenizer_path.exists():
        target_path = output_dir / "tokenizer.json"
        target_path.write_text(tokenizer_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Exported merged weights to {merged_path}")
    print(f"Exported adapters to {adapter_path}")
    return 0


__all__ = ["main"]
