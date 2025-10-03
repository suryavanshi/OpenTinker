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

## RLVR math with unit tests

`python -m rinker.examples.rlvr_math` mirrors Tinker's RLVR arithmetic walkthrough. The environment computes rewards by running
Python unit-test style assertions over the sampled completion and reports the pass/fail status as a metric. Use
`--beta-kl 0.1` to stabilise the policy against the reference model when increasing `group_size`.

## Toy RLHF (reward model → PPO)

`python -m rinker.examples.rlhf_toy` wires a fixed reward model into the RL loop. The reward model emits logits based on
helpfulness/refusal keywords and length penalties, simulating a pretrained RM. The PPO loss uses the per-token log-probs and
the RM score as rewards. Try experimenting with `--beta-kl` and `--group-size` to match the cookbook's behaviour.

## Multi-agent "Twenty Questions"

`python -m rinker.examples.twenty_questions` builds a token-level environment where the model must produce both the
question/answer transcript and the final guess. Rewards are derived from a rule-based oracle that validates each answer,
rewarding accurate conversations and penalising excessive questioning. This reproduces the multi-agent walkthrough while
remaining within the single-step environment abstraction.

## InspectAI integration (future)

The evaluation infrastructure keeps the interface open for later InspectAI hooks. Add new reward actors and extend
`OfflineEvaluator` to emit additional columns once the integration is ready.
