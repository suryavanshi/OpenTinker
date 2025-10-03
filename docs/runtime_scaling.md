# Distributed Ray Runtime

This release adds first-class support for multi-GPU and multi-node deployments.

## Learner orchestration

* **Modes:** Set `RayRuntimeConfig.learner_mode` to `"single"`, `"ddp"`, or
  `"fsdp"`. The Ray learner actor now spawns per-rank subprocesses within the
  actor when DDP/FSDP is requested, using the GPUs assigned to that actor by Ray.
* **AMP/FSDP:** AMP dtype resolution happens inside the actor and is forwarded to
  every rank. The FSDP path uses PyTorch's native `FullyShardedDataParallel`.
* **State management:** `get_state`, `save_state`, and `export_for_hf` now
  transparently marshal rank-0 checkpoints to the driver. Optimiser and Grad
  scaler states are synchronised across ranks.

## Placement groups

`RayRuntime` now creates placement groups by default. The first bundle reserves
resources for the learner, and subsequent bundles map one-to-one with sampler
actors. This guarantees 2-GPU learner + 6-GPU sampler layouts on an 8 GPU node,
or larger cluster placements when spanning nodes.

Configure the layout via `RayRuntimeConfig.learner_num_gpus`,
`sampler_num_gpus`, `num_sampler_actors`, and `placement_strategy`.

## Telemetry dashboards

`RayRuntimeConfig` exposes `tensorboard_logdir` and `wandb_project` (plus
`wandb_run_name` / `wandb_entity`). When set, the runtime emits learner tokens/s,
sampler throughput, tokenizer TPS, and GPU utilisation to TensorBoard and W&B.

## Multi-node tips

1. Start Ray with a head node (`ray start --head`) and join workers with
   `ray start --address='ray://head:10001'` or similar.
2. Use placement groups to keep learner ranks colocated (e.g. STRICT_PACK) and
   samplers spread via `STRICT_SPREAD` for cross-node sampling.
3. The `configs/ray/learner2_sampler6.yaml` preset demonstrates an 8-GPU
   single-node layout (2 learner GPUs + 6 sampler GPUs).

## Model zoo

`ServiceClient` now proxies a lightweight model registry. Request base models via
string aliases such as `"qwen3-0.5b"`, `"qwen3-moe-a14b"`, or
`"llama3-8b"`. Tokenisers are loaded lazily on the driver and shipped to Ray
actors. Hugging Face dependencies remain optional and will raise a friendly
error if missing.
