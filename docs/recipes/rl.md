# Reinforcement Learning

This recipe mirrors the RL section of the Tinker cookbook.

## Config-driven loop

The CLI translates `configs/rl/first_run.yaml` into:

* :class:`LoopConfig` controlling PPO hyper-parameters.
* :class:`EnvGroupBuilder` assembled from `MathTask` definitions.
* :class:`CheckpointManager` for periodic saves.
* :class:`InlineEvaluator` for on-policy validation.

## Offline sweeps

Saved checkpoints contain `metadata.json` with the global step. :class:`rinker.eval.OfflineEvaluator` iterates over the directory,
loads `trainer_state.pt` into a fresh `TrainingClient`, and records reward statistics. The resulting CSV/PNG pair makes it easy to
compare multiple runs or overlay InspectAI metrics in the future.

## KL shaping and group advantages

The trainer applies group-centred advantages and optional KL shaping before each update. Provide
`beta_kl` in the loop config to subtract a KL penalty from rewards before centering.

## InspectAI integration (future)

The evaluation infrastructure keeps the interface open for later InspectAI hooks. Add new reward actors and extend
`OfflineEvaluator` to emit additional columns once the integration is ready.
