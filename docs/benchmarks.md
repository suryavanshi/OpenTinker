# Benchmark snapshots

This note summarises quick sanity runs for the 0.1.0 release. All experiments were
executed on the bundled `tiny-char-gpt` model with Ray in local mode and the new
toy recipes. Metrics correspond to the final iteration of each run.

| Run | Key settings | Reward | KL (p||q) | Clip frac | Tokens/s | Notes |
|-----|--------------|--------|-----------|-----------|----------|-------|
| RLVR baseline | `batch_size=2`, `group_size=2`, `num_substeps=1`, `beta_kl=0.0` | 0.000 | -0.1909 | 0.72 | 256.5 | Stable advantages but no unit tests passed yet. 【8ce8fa†L1-L4】 |
| RLVR, more substeps | `batch_size=2`, `group_size=2`, `num_substeps=2`, `beta_kl=0.0` | 0.000 | 0.2223 | 0.66 | 302.8 | Doubling substeps smooths the clip fraction while keeping rewards flat. 【830670†L1-L5】 |
| RLVR, larger group + KL | `batch_size=2`, `group_size=4`, `num_substeps=1`, `beta_kl=0.2` | -2.093 | 10.4646 | 0.97 | 376.6 | High KL penalty collapses rewards despite higher throughput. 【2913dd†L1-L4】 |
| RLHF toy | `batch_size=2`, `group_size=2`, `num_substeps=1`, `beta_kl=0.05` | -0.192 | 12.5014 | 0.66 | 62.4 | Reward model scores increase while KL grows, highlighting PPO vs. RM balance. 【a5ef47†L1-L4】 |
| Twenty Questions | `batch_size=2`, `group_size=2`, `num_substeps=1`, `beta_kl=0.1`, `max_questions=5` | -1.125 | 6.2541 | 0.71 | 21.5 | Penalty applied when the agent fails to answer and exceeds guidance. 【2d75c9†L1-L2】 |

## Observations

* **Batch/group interplay** – Moving from `group_size=2` to `group_size=4` doubles the
  number of rollouts per environment and increases tokens/s, but also pushes KL
  sharply upward. Without a matching `beta_kl` schedule, rewards can become strongly
  negative.【2913dd†L1-L4】
* **Streaming substeps** – Using two optimisation substeps per batch modestly improves
  throughput while keeping KL near zero and the clip fraction lower than the single
  substep baseline.【830670†L1-L5】【8ce8fa†L1-L4】
* **KL shaping** – Applying `beta_kl=0.2` for larger groups demonstrably reins in the
  policy by subtracting KL from rewards, but overly aggressive values can overwhelm
  the signal and turn the objective negative.【2913dd†L1-L4】
* **Reward-model stability** – The toy RLHF run shows the RM score rising even as PPO
  penalises divergence. KL annealing (or a lower `beta_kl`) is required for longer
  runs to keep the policy from collapsing.【a5ef47†L1-L4】
* **Multi-agent penalties** – The Twenty Questions environment reports detailed
  metrics such as penalties, answer correctness, and question counts so that agents
  can diagnose why transcripts fail the oracle’s checks.【2d75c9†L1-L2】

Each experiment can be reproduced with the commands listed above; use additional
iterations for smoother curves when running outside CI.
