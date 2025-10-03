# Quickstart

This quickstart walks through supervised learning (SL) and reinforcement learning (RL) fine-tuning with the `rinker` CLI.
All commands run locally using the Ray runtime that ships with the framework.

## Requirements

1. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure that Ray can initialise in your environment (the default configuration uses CPU-only actors).

## Supervised learning

1. Create a configuration file `configs/sl/toy.yaml` with the following contents:
   ```yaml
   seed: 42
   sl:
     epochs: 5
     dataset:
       prompts: ["hello"]
       completions: [" world"]
   checkpoint:
     dir: checkpoints/sl-toy
     every_steps: 1
   ```
2. Run the training command:
   ```bash
   rinker train sl -c configs/sl/toy.yaml
   ```
   The CLI prints the per-epoch loss and stores checkpoints under `checkpoints/sl-toy/step_000001/`.

## Reinforcement learning

1. Use the bundled config that mirrors Tinker's "Your first RL run":
   ```bash
   rinker train rl -c configs/rl/first_run.yaml
   ```
   The trainer logs PPO metrics for each iteration, saves checkpoints every five steps, and writes inline evaluation results to
   `checkpoints/first_run/eval/inline_eval.csv`.
2. Sweep the saved checkpoints offline to generate a reward plot:
   ```bash
   rinker eval -c configs/rl/first_run.yaml --checkpoints checkpoints/first_run
   ```
   This command produces `offline_eval.csv` and `offline_eval_reward.png` summarising the reward vs. global step curve.
3. Export the final checkpoint to a Hugging Face style directory:
   ```bash
   rinker export --checkpoint checkpoints/first_run/step_000020 --output exports/first_run
   ```

The offline evaluation step satisfies the definition of done by writing a reward vs. step plot that you can include in reports or
notebooks.
