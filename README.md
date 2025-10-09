# OpenTinker

OpenTinker (codename **Rinker**) is an open-source RL fine-tuning framework that mirrors the ergonomics of Tinker's
Service/Training/Sampling API. It now ships with a batteries-included CLI for running supervised and reinforcement learning
experiments, scheduling evaluations, and exporting LoRA adapters.

## Getting started

1. Install the package from PyPI:
   ```bash
   pip install rinker
   ```
   The source tree can still be installed in editable mode with `pip install -e .` for development workflows.
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

See the [docs site](docs/index.md) for a quickstart, API reference, and recipes that mirror the Tinker cookbook, including
new RLVR, RLHF, and multi-agent walkthroughs.

## Vision-language models

Rinker now exposes an experimental vision stack for multimodal models such as
Qwen3-VL-30B-A3B. The Ray sampler ships with an image-aware configuration block
(`vision_processor_name`, `vision_max_pixels`) and integrates with the new
`QwenVLMRenderer`, enabling multi-modal prompts, pixel down-scaling, and
per-token log probabilities for captioning-style rewards. See the forthcoming
DocVQA/ChartQA examples for a complete pipeline.

## Scaling beyond a single GPU

The Ray runtime now supports placement groups, distributed learners (DDP/FSDP),
and telemetry dashboards. See [runtime_scaling.md](docs/runtime_scaling.md) for
guidance on provisioning 8-GPU nodes (2-GPU learners + 6 sampler GPUs),
multi-node head/worker deployments, and enabling TensorBoard/W&B throughput
dashboards.
