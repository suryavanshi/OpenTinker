# Training & Sampling

This section mirrors the "Training + Sampling" recipes from Tinker.

## Training loops

* Use :class:`rinker.recipes.rl.train.RLTrainer` for PPO/IS style updates with group-centred advantages.
* Configure checkpoints via :class:`rinker.utils.checkpointing.CheckpointManager` and pass it to the trainer.
* Attach inline evaluation by instantiating :class:`rinker.eval.InlineEvaluator` with a dedicated `EnvGroupBuilder`.

Example snippet from the CLI implementation:
```python
loop_config = LoopConfig(...)
checkpoint_manager = CheckpointManager(...)
inline_eval = InlineEvaluator(...)
trainer = RLTrainer(
    training_client=training,
    reference_model=build_reference_model(training),
    config=loop_config,
    checkpoint_manager=checkpoint_manager,
    inline_evaluator=inline_eval,
    training_config=raw_config,
)
trainer.run(builder, sampling_params)
```

## Sampling clients

Call `training_client.save_weights_and_get_sampling_client(name)` whenever you need a refreshed sampler.
The CLI uses this during:

* Training iterations to collect PPO targets.
* Inline evaluation hooks.
* Offline evaluation sweeps.

Sampling parameters come from :class:`rinker.core.types.SamplingParams` or the CLI-friendly
:class:`rinker.cli.config.SamplingConfig` wrapper.
