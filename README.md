# OpenTinker

OpenTinker (codename **Rinker**) is an open-source RL fine-tuning framework that mirrors the ergonomics of Tinker's
Service/Training/Sampling API. It now ships with a batteries-included CLI for running supervised and reinforcement learning
experiments, scheduling evaluations, and exporting LoRA adapters.

## Getting started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the "first RL run" analogue:
   ```bash
   rinker train rl -c configs/rl/first_run.yaml
   ```
3. Sweep checkpoints offline and generate a reward plot:
   ```bash
   rinker eval -c configs/rl/first_run.yaml --checkpoints checkpoints/first_run
   ```
4. Export adapters to Hugging Face format:
   ```bash
   rinker export --checkpoint checkpoints/first_run/step_000020 --output exports/first_run
   ```

See the [docs site](docs/index.md) for a quickstart, API reference, and recipes that mirror the Tinker cookbook.
