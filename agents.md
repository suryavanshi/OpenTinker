# Agents.md — Implementation Guide for GPT‑5‑Codex Agents

> Goal: Build an open‑source RL fine‑tuning framework (**codename: Rinker**) with an API **intentionally close to Tinker**. Support token‑level **PPO** and **Importance‑Sampling (IS/REINFORCE)**, Ray‑managed multi‑GPU orchestration, LoRA, streaming/async updates, group‑centered advantages (GRPO‑style), optional KL‑as‑reward shaping, and reproducible evaluations.

---

## 0) Non‑negotiable Constraints (mirror these exactly)

- **Public API surface** (names & call order):
  - `ServiceClient.get_server_capabilities()`
  - `ServiceClient.create_lora_training_client(base_model, rank, **kwargs)` → `TrainingClient`
  - `TrainingClient.forward_backward(data, loss_fn|custom_fn, **kwargs)` → Future‑like
  - `TrainingClient.optim_step(AdamParams)` → Future‑like
  - `TrainingClient.save_weights_and_get_sampling_client(name)` → `SamplingClient`
  - `SamplingClient.sample(model_input, sampling_params, num_samples)` → generations (+ per‑token logprobs)
- **Built‑in losses** (token‑level, **sum reduction**): `cross_entropy`, `importance_sampling`, `ppo`.
- **User‑provided advantages**: losses don’t compute baselines; the training loop/backend must allow group‑centered advantages; KL (if any) is **reward shaping** pre‑advantage.
- **Rendering** abstraction to convert `messages[]` ↔ tokens (Qwen/Llama chat templating compatible). Separate token vs message completers.
- **Futures semantics**: `forward_backward` and `optim_step` return immediately; user may synchronize via `.result()` (our equivalent).
- **RL knobs**: `batch_size` (envs/problems), `group_size` (rollouts per env), `num_substeps` (updates per sampling iter; single epoch), streaming minibatches, optional async off‑policy ("off‑by‑K").

---

## 1) High‑Level Architecture

### Components

- **Controller (driver)**: Orchestrates sampling → reward → advantage centering → learner updates. Implements streaming/async and off‑by‑K windows.
- **LearnerActor**: Holds trainable policy (LoRA‑wrapped HF model). Methods: `forward_backward`, `optim_step`, `save_state`, `load_state`. Supports AMP, grad‑accum, DDP/FSDP (Week 7).
- **SamplerActor[N]**: Batched generation + per‑token logprobs for each sample; yields token ids, logprobs, and decoded text. Pluggable backends: HF `generate` (Week 3), vLLM (Week 5 optional).
- **RewardActor[K]**: Deterministic programmatic rewards (RLVR) and/or RM‑based scores (RLHF). Pure CPU by default.
- **Renderer**: `build_generation_prompt`, `get_stop_sequences`, `parse_response`. Provide Qwen/Llama templates.

### Resource layout (single node → multi‑node)

- Default: 1 GPU learner + M GPU samplers. Placement via Ray resource tags. Later: DDP learner spanning L GPUs; sampler pool on the remainder.

---

## 2) Week‑by‑Week Execution Plan

### Week 1 — Repo bootstrap & minimal synchronous API
**Objective**: Deliver a local (non‑Ray) skeleton matching the public API; CE loss; basic sampling.

**Tasks**
1. **Scaffold repo**
   ```
   rinker/
     api/{service_client.py,training_client.py,sampling_client.py}
     core/{types.py,losses.py,engine.py,rendering/}
     utils/{logging.py,metrics.py,seeding.py}
     examples/{sl_basic.py}
     tests/
   ```
2. **Types**: `ModelInput`, `Datum`, `TensorDict` (for `loss_fn_inputs` & `loss_fn_outputs`), `AdamParams`, `SamplingParams`.
3. **Losses**: Implement `cross_entropy` with per‑token weights; **sum over tokens**.
4. **Sampling**: Greedy/temperature top‑k/p with logits→logprobs; return per‑token logprobs.
5. **Renderer**: Qwen/Llama chat renderers (`apply_chat_template` parity), stop sequences, response parsing.
6. **API glue**: Wire `ServiceClient→TrainingClient→SamplingClient`; futures are simple Python placeholders for now.

**Acceptance**
- `examples/sl_basic.py` fine‑tunes a tiny model on CPU, then samples text; CE loss decreases.
- Unit tests validate CE against manual computation; rendering adds generation prompt when expected.

---

### Week 2 — Token‑level RL losses + forward_backward_custom
**Objective**: Exact semantics for IS and PPO (token‑level). Custom loss hook.

**Tasks**
1. **IS loss**: `ratio = exp(target_logprobs - sampling_logprobs)`; loss `= -(ratio * advantages).sum()`.
2. **PPO loss**: same `ratio`; clip to `[1-ε, 1+ε]` (default `ε=0.2`); objective `-min(r*A, r_clipped*A).sum()`.
3. **Custom loss**: `forward_backward_custom(callable)` that receives logits/logprobs and `Datum[]`, returns scalar loss + aux metrics; implement extra fwd pass path.
4. **Diagnostics**: add two numerical KL estimators between sampler and learner; log entropy, clip fraction.

**Acceptance**
- Unit tests hit PPO edge cases (A≥0, A<0) and verify gradient signs.
- `examples/rl_basic.py` (toy reward) shows monotone reward improvement.

---

### Week 3 — Ray integration (single node), async futures, sampler pool
**Objective**: Replace placeholders with Ray actors & object refs.

**Tasks**
1. **Actors**: Implement `LearnerActor` and `SamplerActor` with `@ray.remote(num_gpus=...)`.
2. **Futures**: `forward_backward` / `optim_step` return Ray ObjectRefs; add driver helper `wait_all`.
3. **GPU pinning**: Set `CUDA_VISIBLE_DEVICES` correctly inside actors; configurable mapping.
4. **Sampler pool**: Batch prompts, return tokens + per‑token logprobs for each sample; handle stop sequences.
5. **Backpressure**: Bounded in‑flight rollouts; controller avoids deadlocks.

**Acceptance**
- `examples/rl_loop_ray.py` runs with 1 learner GPU + 1 sampler GPU; shows throughput & stable loss.

---

### Week 4 — RL cookbook parity & GRPO‑style centering
**Objective**: Provide env/loop abstractions and hyperparameters that mirror Tinker semantics.

**Tasks**
1. **Env API**: `Env.initial_observation()`, `Env.step(action)` with token I/O. Add `EnvGroupBuilder` and `RLDataset` for grouping.
2. **Group centering**: Utility to compute per‑group advantages: `A_i = r_i - mean(r_group)` (no std/length normalization).
3. **Loops**: `recipes/rl_loop.py` (simple) and `recipes/rl/train.py` (performant streaming). Single epoch per batch.
4. **Hyperparams**: Support `batch_size`, `group_size`, `num_substeps`; document semantics.
5. **KL shaping**: Optional `β·KL(pi || ref)` subtraction **into rewards** prior to advantage computation.
6. **Metrics**: Log KL estimators, entropy, clip rate, tokens/s, reward stats.

**Acceptance**
- Port a GSM8K‑style RL example; with `group_size>1`, reward and correctness increase over ~10–20 iters.

---

### Week 5 — Performance: LoRA, AMP, streaming/async, vLLM option
**Objective**: Reach practical throughput & overlap sampling and learning.

**Tasks**
1. **LoRA**: Integrate PEFT; export adapters as `*.safetensors`.
2. **AMP**: Enable bf16/fp16; grad accumulation.
3. **Streaming minibatches**: Overlap sampler production and learner consumption with `groups_per_batch` and `num_minibatches`. On‑policy pipeline (single‑epoch guarantee).
4. **Async off‑policy**: Allow off‑by‑K updates with guard rails and warnings.
5. **vLLM backend** (optional): Sampler variant with higher TPS; still return per‑token logprobs. Provide fallbacks if backend cannot expose per‑token logprobs directly.

**Acceptance**
- On 4 GPUs: 1 learner + 3 samplers, near‑linear sampler scaling; stable training with `max_steps_off_policy ≤ 3`.

---

### Week 6 — Evaluations, checkpoints, CLI
**Objective**: One‑command runs; routine evals; HF export.

**Tasks**
1. **Eval hooks**: Inline eval every `eval_every` steps; offline eval on saved checkpoints.
2. **Checkpoints**: Save/load model, optimizer, tokenizer, renderer, and config.
3. **CLI/YAML**: `rinker train sl|rl -c config.yaml` with Hydra/TOML.
4. **Docs**: Quickstart, API reference, RL recipes.

**Acceptance**
- End‑to‑end “first RL run” parity: visible metrics (reward, correctness, KL est., entropy), single command.

---

### Week 7 — Multi‑node scaling: DDP/FSDP learner, Ray placement groups
**Objective**: Cluster scale.

**Tasks**
1. **DDP/FSDP** inside `LearnerActor` (rank launcher inside the actor, elastic rendezvous). Ability to pin L GPUs to learner.
2. **Placement groups**: Reserve bundles: `{"CPU": x, "GPU": L}` for learner, separate bundles for samplers; PACK/STRICT_PACK strategy for locality.
3. **Tokenizer/model zoo**: String‑selectable models (Qwen/Llama variants) with renderer bindings.

**Acceptance**
- 8‑GPU node: `learner=2 GPUs`, `samplers=6 GPUs`; demonstrates stable scaling; multi‑node smoke test.

---

### Week 8 — Hardening, examples, release
**Objective**: Stabilize and publish.

**Tasks**
1. **Examples**: RLVR math, RLHF with RM, multi‑agent “Twenty Questions”.
2. **API lock**: Freeze signatures; tag v0.1.0.
3. **Release**: Apache‑2.0; PyPI; README with benchmarks/plots.

**Acceptance**
- Reproduce cookbook‑style curves; docs polished; wheels published.

---

## 3) Detailed Specs for Key Modules

### 3.1 Losses (`core/losses.py`)

```python
@dataclass
class LossOutputs:
    loss: torch.Tensor           # scalar
    extras: dict[str, torch.Tensor]  # e.g., kl_v1, kl_v2, entropy, clip_frac

class Losses:
    def cross_entropy(logits, target_tokens, weights) -> LossOutputs: ...
    def importance_sampling(target_logprobs, sampling_logprobs, advantages) -> LossOutputs: ...
    def ppo(target_logprobs, sampling_logprobs, advantages, eps: float = 0.2) -> LossOutputs: ...
```

**Rules**
- All losses are **token‑level** and **sum** reduced. Batch items can have different lengths; use masks.
- Inputs are provided via `Datum.loss_fn_inputs: dict[str, Tensor]`.
- For PPO/IS, **advantages** are user‑provided. Support broadcast of sequence‑wise constant advantages.

### 3.2 Engine (`core/engine.py`)

- `forward_backward(data: list[Datum], loss_fn: str, **cfg) -> FwdBwdResult`
  - Computes logits/logprobs; extracts `loss_fn_inputs`; applies selected loss; `loss.backward()`; collects metrics.
- `forward_backward_custom(data, fn: Callable) -> FwdBwdResult`
  - Makes outputs (logits/logprobs, masks) available to `fn`; performs an **extra forward** if needed.
- `optim_step(AdamParams)`
  - Applies optimizer step; zero grads; returns current step, lr, grad‑norm.

### 3.3 Renderer (`core/rendering/`)

- `Renderer.build_generation_prompt(messages) -> TokenBatch`
- `Renderer.get_stop_sequences() -> list[str]`
- `Renderer.parse_response(text) -> Parsed`
- Ship `QwenRenderer` and `LlamaRenderer` with template hooks and `add_generation_prompt` behavior.

### 3.4 Ray runtime (`ray_runtime/`)

- `LearnerActor`: manages model state, gradients, optimizer, AMP, LoRA; exposes API.
- `SamplerActor`: accepts `ModelInput`, `SamplingParams`, `num_samples`; returns ids, text, and **per‑token logprobs**.
- `RewardActor`: configurable Python function or RM call; returns scalar reward per sample and aux fields.
- Placement: default `num_gpus={learner:1, sampler:1}`; configurable via YAML.

### 3.5 RL Orchestrator

- Batches N environments; for each, draws `group_size = G` samples.
- Computes rewards per sample; **centers advantages per group**: `A_i = r_i - mean(r)`.
- (Optional) subtracts `β·KL(pi || ref)` from rewards before centering.
- Packages `Datum[]` with `{target_tokens, sampling_logprobs, advantages}`; calls `forward_backward(..., loss_fn='ppo'|'importance_sampling')`.
- Streaming mode: produces minibatches while samplers continue generating.
- Async off‑policy: allows using data from prior policies up to K steps old; warns and discards older.

---

## 4) Coding Conventions & Quality Gates

- **Determinism toggles**: `torch.use_deterministic_algorithms(True)` option; seed everything; document non‑determinism in generation.
- **Numerics**: clamp logprobs; stable ratio computation; monitor clip fraction.
- **Telemetry**: log KL (two estimators), entropy, reward stats, tokens/s, latency per stage.
- **Checkpoints**: include model/optimizer/tokenizer/renderer/config.
- **Export**: convert LoRA + base to HF format for easy loading.

---

## 5) Test Plan (must pass before merge)

1. **Unit: Loss math** — CE equals manual; IS reduces to REINFORCE when `target_logprobs==sampling_logprobs`; PPO clipping behaves at boundaries.
2. **Unit: Rendering** — Qwen/Llama templates add generation prompt appropriately; stops respected.
3. **Integration: Ray** — single‑GPU end‑to‑end RL loop produces increasing reward.
4. **Scale** — 1→4 GPUs: near‑linear sampler TPS; learner stable.
5. **Fault tolerance** — kill a sampler; controller recovers; step ids monotonic.

---

## 6) Deliverables per Week (Definition of Done)

- **W1**: Local API + CE; `sl_basic.py` runs.
- **W2**: IS & PPO; custom loss; `rl_basic.py` improves reward.
- **W3**: Ray actors; `rl_loop_ray.py` runs on 2 GPUs.
- **W4**: Env/loop parity; group centering; KL shaping util; GSM8K‑style example.
- **W5**: LoRA, AMP, streaming, async off‑policy; optional vLLM sampler.
- **W6**: Evals, checkpoints, CLI.
- **W7**: Multi‑node scaling; placement groups; DDP/FSDP learner.
- **W8**: Examples, polish, v0.1.0 release.

---

## 7) Implementation Hints & Gotchas (for agents)

- **Per‑token logprobs**: ensure sampler returns logprob for every emitted token; align lengths and masks.
- **Single‑epoch guarantee**: even with `num_substeps>1`, iterate each unique environment’s data exactly once per iteration.
- **Advantage centering**: never apply std or length normalization; keep sequence‑constant `A` if needed.
- **KL as reward shaping**: avoid mixing KL into PPO loss; treat it as a reward penalty before advantage computation.
- **Ray backpressure**: use bounded queues or `ray.wait` to avoid OOM from runaway samplers.
- **vLLM**: if backend can’t expose fine‑grained logprobs, fall back to HF sampler for training; keep vLLM for eval/inference.

---

## 8) Example Pseudocode Snippets

**PPO Loss (token‑level)**
```python
def ppo_loss(target_logprobs, sampling_logprobs, advantages, eps=0.2):
    ratio = torch.exp(target_logprobs - sampling_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    loss = -torch.sum(torch.minimum(unclipped, clipped))
    return loss
```

**Group‑centered advantages**
```python
def center_group(rewards: list[float]) -> torch.Tensor:
    r = torch.tensor(rewards, dtype=torch.float32)
    return r - r.mean()
```

**Ray actor sketch**
```python
@ray.remote(num_gpus=1)
class LearnerActor:
    def __init__(self, model_id, lora_rank): ...
    def forward_backward(self, batch, loss_name, cfg): ...
    def optim_step(self, adam_params): ...
    def save(self, path): ...

@ray.remote(num_gpus=1)
class SamplerActor:
    def __init__(self, model_id, renderer_cfg): ...
    def generate(self, inputs, sampling_params, num_samples): ...  # returns ids, logprobs, text
```

---

## 9) Roadmap (post‑v0.1)

- DPO and custom preference objectives via `forward_backward_custom`.
- KV‑budget‑aware sampler scheduling; dynamic temperature & length penalties.
- Reward packs: Math verifier, code unit tests; renderer packs; eval packs.

---

## 10) How to Work as GPT‑5‑Codex Agents

- **Style**: produce small PRs per task; maintain parity with stated public API.
- **Docs first**: keep docstrings and markdown updated with each module.
- **Bench**: add tiny deterministic benchmarks; plot tokens/s and clip fraction.
- **Review checklist**: losses numeric parity, renderer correctness, Ray resource sanity, eval hooks, checkpoint integrity.

> When in doubt, follow the constraints in Section 0 and keep the API/semantics consistent across SL, RL, and evaluation paths.

