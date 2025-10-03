# API Reference

Rinker's public API mirrors Tinker's Service/Training/Sampling trio. The following table summarises the surface:

| Component | Method | Description |
|-----------|--------|-------------|
| `ServiceClient` | `get_server_capabilities()` | Returns supported base models. |
| | `create_lora_training_client(base_model, rank, **kwargs)` | Spawns a `TrainingClient` backed by Ray actors. |
| `TrainingClient` | `forward_backward(batch, loss_fn="cross_entropy")` | Starts a forward/backward pass and returns a future-like object. |
| | `forward_backward_custom(batch, callable)` | Custom log-prob losses. |
| | `optim_step(AdamParams)` | Applies an optimiser step. |
| | `save_weights_and_get_sampling_client(name)` | Broadcasts the latest weights and returns a `SamplingClient`. |
| | `save_state()` / `load_state(state)` | Serialises or restores the learner. |
| | `export_lora_weights()` | Returns merged and adapter state dicts for Hugging Face export. |
| | `stream_minibatch_train(dataset, ...)` | Utility for streaming PPO/IS updates. |
| `SamplingClient` | `sample(model_input, sampling_params, num_samples)` | Generates completions with per-token log-probabilities. |

### Checkpointing helpers

The CLI wraps :class:`rinker.utils.checkpointing.CheckpointManager`, which saves the following artefacts into each step directory:

* `adapter.safetensors` – LoRA adapter weights.
* `trainer_state.pt` – model, optimiser, and scaler state for restoration.
* `optimizer.pt` – optional optimiser snapshot (configurable).
* `tokenizer.json` – serialised `SimpleTokenizer` vocabulary.
* `config.yaml` – the training configuration used for the run.
* `metadata.json` – global step metadata for offline evaluation.

Use the CLI or instantiate :class:`CheckpointManager` directly in custom loops.

### Evaluation helpers

* :class:`rinker.eval.InlineEvaluator` – attach `every_steps` hooks to a running `RLTrainer`.
* :class:`rinker.eval.OfflineEvaluator` – sweep checkpoints and emit CSV + reward plots.

Both utilities operate on the high-level :class:`EnvGroupBuilder` abstraction to stay compatible with future multi-modal
renderers.
