"""Ray actors for the learner, sampler, and reward components."""
from __future__ import annotations

import contextlib
import importlib.util
import os
import queue
import socket
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from ..core import engine
from ..core.lora import LoRAConfig, extract_lora_parameters, merge_lora_weights
from ..core.model_zoo import ModelSpec, build_model
from ..core.types import AdamParams, Datum, ModelInput, SamplingParams
from ..utils.tokenizer import TokenizerProtocol

try:  # pragma: no cover - ray import is optional in unit tests
    import ray
except ImportError as exc:  # pragma: no cover - handled by caller
    raise RuntimeError(
        "Ray is required for the ray_runtime components. "
        "Install with `pip install ray`."
    ) from exc


@dataclass
class ForwardBackwardPayload:
    """Container returned from the learner after a backward pass."""

    loss: float
    metrics: Mapping[str, float]
    loss_fn_outputs: Mapping[str, torch.Tensor] = field(default_factory=dict)
    tokens_processed: int = 0
    gpu_utilization: float | None = None
    duration_s: float | None = None


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            return torch.device("cuda")
    return torch.device("cpu")


def _count_tokens(batch: Sequence[Datum]) -> int:
    total = 0
    for datum in batch:
        if "target_tokens" in datum.loss_fn_inputs:
            tensor = datum.loss_fn_inputs["target_tokens"]
        elif "targets" in datum.loss_fn_inputs:
            tensor = datum.loss_fn_inputs["targets"]
        else:
            continue
        if isinstance(tensor, torch.Tensor):
            total += int(tensor.numel())
    return total


def _query_gpu_utilization(gpu_indices: Iterable[int]) -> float | None:
    try:
        import pynvml  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # pragma: no cover - NVML initialisation may fail in CI
        return None
    try:
        utils: List[float] = []
        for index in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utils.append(float(rates.gpu))
        if not utils:
            return None
        return sum(utils) / len(utils)
    except Exception:  # pragma: no cover - NVML quirks
        return None
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _split_batch(batch: Sequence[Datum], world_size: int) -> List[List[Datum]]:
    if world_size <= 1:
        return [list(batch)]
    shards: List[List[Datum]] = [[] for _ in range(world_size)]
    for index, datum in enumerate(batch):
        shards[index % world_size].append(datum)
    return shards


def _resolve_amp_dtype(value: str | None) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = value.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported AMP dtype '{value}'")
    return mapping[key]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


class _BaseLearnerStrategy:
    def forward_backward(self, batch: Sequence[Datum], loss_fn: str) -> ForwardBackwardPayload:
        raise NotImplementedError

    def forward_backward_custom(
        self,
        batch: Sequence[Datum],
        loss_fn: Callable[[Sequence[Datum], torch.Tensor], engine.CustomLossOutputs],
    ) -> ForwardBackwardPayload:
        raise NotImplementedError

    def optim_step(self, params: AdamParams) -> Mapping[str, float]:
        raise NotImplementedError

    def get_state(self) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    def save_state(self) -> Mapping[str, object]:
        raise NotImplementedError

    def load_state(self, state: Mapping[str, object]) -> None:
        raise NotImplementedError

    def export_for_hf(self) -> Mapping[str, Mapping[str, torch.Tensor]]:
        raise NotImplementedError

    def shutdown(self) -> None:
        return None


class SingleProcessLearnerStrategy(_BaseLearnerStrategy):
    def __init__(
        self,
        *,
        model_spec: ModelSpec,
        tokenizer_vocab_size: int,
        lora_config: LoRAConfig | None,
        amp_dtype: str | None,
        gradient_accumulation_steps: int,
    ) -> None:
        self._device = _resolve_device()
        self._amp_dtype = _resolve_amp_dtype(amp_dtype)
        self._model = build_model(
            model_spec,
            vocab_size=tokenizer_vocab_size,
            lora_config=lora_config,
            amp_dtype=self._amp_dtype,
            device=self._device,
        )
        self._model.train()
        self._optimiser: torch.optim.Optimizer | None = None
        self._grad_scaler = GradScaler(
            enabled=self._amp_dtype == torch.float16 and self._device.type == "cuda"
        )
        self._accum_steps = max(int(gradient_accumulation_steps), 1)
        self._micro_step_progress = 0
        self._global_step = 0
        self._pending_optim_state: Mapping[str, object] | None = None

    def forward_backward(self, batch: Sequence[Datum], loss_fn: str) -> ForwardBackwardPayload:
        result = engine.forward_backward(
            self._model,
            batch,
            loss_fn,
            device=self._device,
            amp_dtype=self._amp_dtype,
            grad_scaler=self._grad_scaler if self._grad_scaler.is_enabled() else None,
            accumulation_steps=self._accum_steps,
        )
        self._micro_step_progress += 1
        return ForwardBackwardPayload(
            loss=result.loss,
            metrics=result.metrics,
            loss_fn_outputs=result.loss_fn_outputs,
            tokens_processed=_count_tokens(batch),
            gpu_utilization=_query_gpu_utilization(ray.get_gpu_ids()),
        )

    def forward_backward_custom(
        self,
        batch: Sequence[Datum],
        loss_fn: Callable[[Sequence[Datum], torch.Tensor], engine.CustomLossOutputs],
    ) -> ForwardBackwardPayload:
        result = engine.forward_backward_custom(
            self._model,
            batch,
            loss_fn,
            device=self._device,
            amp_dtype=self._amp_dtype,
            grad_scaler=self._grad_scaler if self._grad_scaler.is_enabled() else None,
            accumulation_steps=self._accum_steps,
        )
        self._micro_step_progress += 1
        return ForwardBackwardPayload(
            loss=result.loss,
            metrics=result.metrics,
            loss_fn_outputs=result.loss_fn_outputs,
            tokens_processed=_count_tokens(batch),
            gpu_utilization=_query_gpu_utilization(ray.get_gpu_ids()),
        )

    def optim_step(self, params: AdamParams) -> Mapping[str, float]:
        self._optimiser = engine.ensure_adam(self._model, self._optimiser, params)
        if self._pending_optim_state is not None:
            self._optimiser.load_state_dict(self._pending_optim_state)  # type: ignore[arg-type]
            self._pending_optim_state = None
        should_step = self._micro_step_progress >= self._accum_steps
        metrics = engine.optim_step(
            self._model,
            self._optimiser,
            grad_scaler=self._grad_scaler if self._grad_scaler.is_enabled() else None,
            should_step=should_step,
        )
        if metrics.get("stepped"):
            self._global_step += 1
            self._micro_step_progress = 0
        return metrics

    def get_state(self) -> Mapping[str, torch.Tensor]:
        state_dict = self._model.state_dict()
        return {key: value.detach().cpu() for key, value in state_dict.items()}

    def save_state(self) -> Mapping[str, object]:
        optimiser_state = self._optimiser.state_dict() if self._optimiser is not None else None
        grad_scaler_state = self._grad_scaler.state_dict() if self._grad_scaler.is_enabled() else None
        model_state = {key: value.detach().cpu() for key, value in self._model.state_dict().items()}
        return {
            "model": model_state,
            "optimiser": optimiser_state,
            "grad_scaler": grad_scaler_state,
            "global_step": self._global_step,
            "accumulation_progress": self._micro_step_progress,
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        model_state = state.get("model")
        if model_state:
            self._model.load_state_dict(model_state)
        if self._optimiser is not None and state.get("optimiser") is not None:
            self._optimiser.load_state_dict(state["optimiser"])  # type: ignore[arg-type]
        elif state.get("optimiser") is not None:
            self._pending_optim_state = state["optimiser"]  # type: ignore[assignment]
            warnings.warn(
                "Optimiser state deferred until optimiser initialisation",
                RuntimeWarning,
            )
        scaler_state = state.get("grad_scaler")
        if scaler_state and self._grad_scaler.is_enabled():
            self._grad_scaler.load_state_dict(scaler_state)  # type: ignore[arg-type]
        self._global_step = int(state.get("global_step", self._global_step))
        self._micro_step_progress = int(state.get("accumulation_progress", 0))

    def export_for_hf(self) -> Mapping[str, Mapping[str, torch.Tensor]]:
        module = self._model
        return {
            "merged": merge_lora_weights(module),
            "adapters": extract_lora_parameters(module),
        }


class DistributedLearnerStrategy(_BaseLearnerStrategy):
    def __init__(
        self,
        *,
        model_spec: ModelSpec,
        tokenizer_vocab_size: int,
        lora_config: LoRAConfig | None,
        amp_dtype: str | None,
        gradient_accumulation_steps: int,
        mode: str,
        backend: str,
        device_ids: Sequence[int],
    ) -> None:
        self._world_size = max(1, len(device_ids))
        self._amp_dtype = amp_dtype
        self._accum_steps = max(int(gradient_accumulation_steps), 1)
        self._mode = mode
        self._backend = backend
        self._model_spec = model_spec
        self._tokenizer_vocab_size = tokenizer_vocab_size
        self._lora_config = lora_config
        self._ticket = 0
        self._mp_ctx = mp.get_context("spawn")
        self._result_queue: mp.Queue = self._mp_ctx.Queue()
        self._command_queues: List[mp.Queue] = []
        self._processes: List[mp.Process] = []
        self._closed = False
        master_port = _find_free_port()
        for rank, device_id in enumerate(device_ids):
            command_queue: mp.Queue = self._mp_ctx.Queue()
            process = self._mp_ctx.Process(
                target=_distributed_worker_main,
                args=(
                    rank,
                    self._world_size,
                    device_id,
                    command_queue,
                    self._result_queue,
                    model_spec,
                    tokenizer_vocab_size,
                    lora_config,
                    amp_dtype,
                    self._accum_steps,
                    mode,
                    backend,
                    master_port,
                ),
            )
            process.start()
            self._processes.append(process)
            self._command_queues.append(command_queue)

    def forward_backward(self, batch: Sequence[Datum], loss_fn: str) -> ForwardBackwardPayload:
        ticket = self._next_ticket()
        shards = _split_batch(batch, self._world_size)
        for rank, queue in enumerate(self._command_queues):
            queue.put(
                (
                    "forward_backward",
                    {
                        "ticket": ticket,
                        "batch": shards[rank],
                        "loss_fn": loss_fn,
                    },
                )
            )
        results = self._gather("forward_backward", ticket)
        return self._aggregate_forward_backward(results)

    def forward_backward_custom(
        self,
        batch: Sequence[Datum],
        loss_fn: Callable[[Sequence[Datum], torch.Tensor], engine.CustomLossOutputs],
    ) -> ForwardBackwardPayload:
        ticket = self._next_ticket()
        shards = _split_batch(batch, self._world_size)
        for rank, queue in enumerate(self._command_queues):
            queue.put(
                (
                    "forward_backward_custom",
                    {
                        "ticket": ticket,
                        "batch": shards[rank],
                        "loss_fn": loss_fn,
                    },
                )
            )
        results = self._gather("forward_backward_custom", ticket)
        return self._aggregate_forward_backward(results)

    def optim_step(self, params: AdamParams) -> Mapping[str, float]:
        ticket = self._next_ticket()
        for queue in self._command_queues:
            queue.put(("optim_step", {"ticket": ticket, "params": params}))
        results = self._gather("optim_step", ticket)
        aggregated: Dict[str, float] = {}
        for result in results:
            for key, value in result.get("metrics", {}).items():
                aggregated[key] = aggregated.get(key, 0.0) + float(value)
        if aggregated:
            for key in list(aggregated.keys()):
                aggregated[key] /= float(len(results))
        return aggregated

    def get_state(self) -> Mapping[str, torch.Tensor]:
        ticket = self._next_ticket()
        for queue in self._command_queues:
            queue.put(("get_state", {"ticket": ticket}))
        results = self._gather("get_state", ticket)
        for result in results:
            state = result.get("state")
            if state:
                return state
        return {}

    def save_state(self) -> Mapping[str, object]:
        ticket = self._next_ticket()
        for queue in self._command_queues:
            queue.put(("save_state", {"ticket": ticket}))
        results = self._gather("save_state", ticket)
        for result in results:
            state = result.get("state")
            if state:
                return state
        return {}

    def load_state(self, state: Mapping[str, object]) -> None:
        ticket = self._next_ticket()
        for queue in self._command_queues:
            queue.put(("load_state", {"ticket": ticket, "state": state}))
        self._gather("load_state", ticket)

    def export_for_hf(self) -> Mapping[str, Mapping[str, torch.Tensor]]:
        ticket = self._next_ticket()
        for queue in self._command_queues:
            queue.put(("export_for_hf", {"ticket": ticket}))
        results = self._gather("export_for_hf", ticket)
        for result in results:
            export = result.get("export")
            if export:
                return export
        return {}

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for queue in self._command_queues:
            queue.put(("shutdown", {}))
        for process in self._processes:
            process.join(timeout=5)
        self._command_queues.clear()
        self._processes.clear()

    def _next_ticket(self) -> int:
        self._ticket += 1
        return self._ticket

    def _gather(self, command: str, ticket: int) -> List[Mapping[str, object]]:
        results: List[Mapping[str, object]] = []
        while len(results) < self._world_size:
            received_command, payload = self._result_queue.get()
            if received_command != command:
                continue
            if payload.get("ticket") != ticket:
                continue
            results.append(payload)
        return results

    def _aggregate_forward_backward(
        self, results: List[Mapping[str, object]]
    ) -> ForwardBackwardPayload:
        if not results:
            return ForwardBackwardPayload(loss=0.0, metrics={}, loss_fn_outputs={}, tokens_processed=0)
        total_loss = sum(float(item.get("loss", 0.0)) for item in results) / len(results)
        aggregated_metrics: Dict[str, float] = {}
        for item in results:
            metrics = item.get("metrics", {})
            for key, value in metrics.items():
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + float(value)
        if aggregated_metrics:
            for key in list(aggregated_metrics.keys()):
                aggregated_metrics[key] /= float(len(results))
        loss_outputs: Mapping[str, torch.Tensor] = {}
        for item in results:
            outputs = item.get("loss_fn_outputs")
            if outputs:
                loss_outputs = outputs
                break
        tokens = sum(int(item.get("tokens", 0)) for item in results)
        gpu_utils = [item.get("gpu_util") for item in results if item.get("gpu_util") is not None]
        gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
        return ForwardBackwardPayload(
            loss=total_loss,
            metrics=aggregated_metrics,
            loss_fn_outputs=loss_outputs,
            tokens_processed=tokens,
            gpu_utilization=gpu_util,
        )


@ray.remote
class LearnerActor:
    """Learner actor that owns the trainable policy."""

    def __init__(
        self,
        model_spec: ModelSpec,
        *,
        tokenizer_vocab_size: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        amp_dtype: str | None = None,
        gradient_accumulation_steps: int = 1,
        learner_mode: str = "single",
        distributed_backend: str = "nccl",
    ) -> None:
        lora_config = (
            LoRAConfig(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
            if lora_rank > 0
            else None
        )
        requested_mode = (learner_mode or "single").lower()
        device_ids = [int(idx) for idx in ray.get_gpu_ids()]
        if requested_mode in {"ddp", "fsdp"} and len(device_ids) > 1:
            self._strategy: _BaseLearnerStrategy = DistributedLearnerStrategy(
                model_spec=model_spec,
                tokenizer_vocab_size=tokenizer_vocab_size,
                lora_config=lora_config,
                amp_dtype=amp_dtype,
                gradient_accumulation_steps=gradient_accumulation_steps,
                mode=requested_mode,
                backend=distributed_backend,
                device_ids=device_ids,
            )
        else:
            self._strategy = SingleProcessLearnerStrategy(
                model_spec=model_spec,
                tokenizer_vocab_size=tokenizer_vocab_size,
                lora_config=lora_config,
                amp_dtype=amp_dtype,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

    def forward_backward(self, batch: Sequence[Datum], loss_fn: str) -> ForwardBackwardPayload:
        return self._strategy.forward_backward(batch, loss_fn)

    def forward_backward_custom(
        self,
        batch: Sequence[Datum],
        loss_fn: Callable[[Sequence[Datum], torch.Tensor], engine.CustomLossOutputs],
    ) -> ForwardBackwardPayload:
        return self._strategy.forward_backward_custom(batch, loss_fn)

    def optim_step(self, params: AdamParams) -> Mapping[str, float]:
        return self._strategy.optim_step(params)

    def get_state(self) -> Mapping[str, torch.Tensor]:
        return self._strategy.get_state()

    def save_state(self) -> Mapping[str, object]:
        return self._strategy.save_state()

    def load_state(self, state: Mapping[str, object]) -> None:
        self._strategy.load_state(state)

    def export_for_hf(self) -> Mapping[str, Mapping[str, torch.Tensor]]:
        return self._strategy.export_for_hf()

    def __del__(self):  # pragma: no cover - best effort cleanup
        with contextlib.suppress(Exception):
            self._strategy.shutdown()


def _distributed_worker_main(
    rank: int,
    world_size: int,
    device_id: int,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    model_spec: ModelSpec,
    tokenizer_vocab_size: int,
    lora_config: LoRAConfig | None,
    amp_dtype: str | None,
    accumulation_steps: int,
    mode: str,
    backend: str,
    master_port: int,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    dist_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=dist_backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available() and device_id is not None:
        device = torch.device("cuda", int(device_id))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    resolved_amp = _resolve_amp_dtype(amp_dtype)
    model = build_model(
        model_spec,
        vocab_size=tokenizer_vocab_size,
        lora_config=lora_config,
        amp_dtype=resolved_amp,
        device=device,
    )
    if mode == "fsdp":  # pragma: no cover - heavy path
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        wrapped_model = FSDP(model)
    elif mode == "ddp":
        device_ids = [int(device_id)] if device.type == "cuda" else None
        wrapped_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=int(device_id) if device_ids else None,
        )
    else:
        wrapped_model = model
    optimiser: torch.optim.Optimizer | None = None
    grad_scaler = GradScaler(enabled=resolved_amp == torch.float16 and device.type == "cuda")
    micro_step_progress = 0
    global_step = 0
    pending_optim_state: Mapping[str, object] | None = None
    try:
        while True:
            command, payload = command_queue.get()
            if command == "shutdown":
                break
            ticket = payload.get("ticket")
            if command == "forward_backward":
                batch = payload.get("batch", [])
                loss_fn = payload["loss_fn"]
                result = engine.forward_backward(
                    wrapped_model,
                    batch,
                    loss_fn,
                    device=device,
                    amp_dtype=resolved_amp,
                    grad_scaler=grad_scaler if grad_scaler.is_enabled() else None,
                    accumulation_steps=accumulation_steps,
                )
                micro_step_progress += 1
                result_queue.put(
                    (
                        command,
                        {
                            "ticket": ticket,
                            "loss": result.loss,
                            "metrics": result.metrics,
                            "loss_fn_outputs": result.loss_fn_outputs if rank == 0 else None,
                            "tokens": _count_tokens(batch),
                            "gpu_util": _query_gpu_utilization([device_id]) if device.type == "cuda" else None,
                        },
                    )
                )
            elif command == "forward_backward_custom":
                batch = payload.get("batch", [])
                loss_fn = payload["loss_fn"]
                result = engine.forward_backward_custom(
                    wrapped_model,
                    batch,
                    loss_fn,
                    device=device,
                    amp_dtype=resolved_amp,
                    grad_scaler=grad_scaler if grad_scaler.is_enabled() else None,
                    accumulation_steps=accumulation_steps,
                )
                micro_step_progress += 1
                result_queue.put(
                    (
                        command,
                        {
                            "ticket": ticket,
                            "loss": result.loss,
                            "metrics": result.metrics,
                            "loss_fn_outputs": result.loss_fn_outputs if rank == 0 else None,
                            "tokens": _count_tokens(batch),
                            "gpu_util": _query_gpu_utilization([device_id]) if device.type == "cuda" else None,
                        },
                    )
                )
            elif command == "optim_step":
                params = payload["params"]
                optimiser = engine.ensure_adam(wrapped_model, optimiser, params)
                if pending_optim_state is not None:
                    optimiser.load_state_dict(pending_optim_state)  # type: ignore[arg-type]
                    pending_optim_state = None
                should_step = micro_step_progress >= accumulation_steps
                metrics = engine.optim_step(
                    wrapped_model,
                    optimiser,
                    grad_scaler=grad_scaler if grad_scaler.is_enabled() else None,
                    should_step=should_step,
                )
                if metrics.get("stepped"):
                    global_step += 1
                    micro_step_progress = 0
                result_queue.put((command, {"ticket": ticket, "metrics": metrics, "rank": rank}))
            elif command == "get_state":
                module = wrapped_model.module if hasattr(wrapped_model, "module") else wrapped_model
                state = (
                    {key: value.detach().cpu() for key, value in module.state_dict().items()}
                    if rank == 0
                    else None
                )
                result_queue.put((command, {"ticket": ticket, "state": state}))
            elif command == "save_state":
                module = wrapped_model.module if hasattr(wrapped_model, "module") else wrapped_model
                state = None
                if rank == 0:
                    state = {
                        "model": {key: value.detach().cpu() for key, value in module.state_dict().items()},
                        "optimiser": optimiser.state_dict() if optimiser is not None else None,
                        "grad_scaler": grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
                        "global_step": global_step,
                        "accumulation_progress": micro_step_progress,
                    }
                result_queue.put((command, {"ticket": ticket, "state": state}))
            elif command == "load_state":
                module = wrapped_model.module if hasattr(wrapped_model, "module") else wrapped_model
                state = payload.get("state", {})
                model_state = state.get("model")
                if model_state:
                    module.load_state_dict(model_state)
                optim_state = state.get("optimiser")
                if optimiser is not None and optim_state is not None:
                    optimiser.load_state_dict(optim_state)  # type: ignore[arg-type]
                elif optim_state is not None:
                    pending_optim_state = optim_state
                scaler_state = state.get("grad_scaler")
                if scaler_state and grad_scaler.is_enabled():
                    grad_scaler.load_state_dict(scaler_state)  # type: ignore[arg-type]
                global_step = int(state.get("global_step", global_step))
                micro_step_progress = int(state.get("accumulation_progress", micro_step_progress))
                result_queue.put((command, {"ticket": ticket, "ack": True}))
            elif command == "export_for_hf":
                module = wrapped_model.module if hasattr(wrapped_model, "module") else wrapped_model
                export = None
                if rank == 0:
                    export = {
                        "merged": merge_lora_weights(module),
                        "adapters": extract_lora_parameters(module),
                    }
                result_queue.put((command, {"ticket": ticket, "export": export}))
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"Unknown distributed command '{command}'")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@ray.remote
class SamplerActor:
    """Sampler actor responsible for autoregressive generation."""

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        *,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        backend: str = "torch",
    ):
        self._tokenizer = tokenizer
        self._device = _resolve_device()
        lora_config = LoRAConfig(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        self._model = engine.SimpleLanguageModel(
            tokenizer.vocab_size,
            lora_config=lora_config if lora_rank > 0 else None,
        )
        self._model.to(self._device)
        self._model.eval()
        self._stop_sequences: List[str] = []
        self._backend = backend
        if backend == "vllm":
            if importlib.util.find_spec("vllm") is None:
                warnings.warn(
                    "vLLM backend requested but vllm is not installed; falling back to torch sampler",
                    RuntimeWarning,
                )
                self._backend = "torch"
            else:  # pragma: no cover - optional path
                import vllm  # type: ignore

                self._vllm = vllm

    def set_weights(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

    def generate(
        self,
        model_input: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int,
    ) -> List[Mapping[str, object]]:
        prompt_tokens, tokenizer_time_s = self._prepare_prompt(model_input)
        prompt_tokens = prompt_tokens.to(self._device)
        prompt_token_count = int(prompt_tokens.numel())
        outputs: List[Mapping[str, object]] = []
        stop_sequences = sampling_params.stop_sequences or self._stop_sequences
        gpu_util = _query_gpu_utilization(ray.get_gpu_ids())

        for _ in range(num_samples):
            token_ids = prompt_tokens.clone().tolist()
            generated_logprobs: List[float] = []
            for _ in range(sampling_params.max_new_tokens):
                input_tensor = torch.tensor(token_ids, dtype=torch.long, device=self._device).unsqueeze(0)
                logits = self._model(input_tensor)[0, -1]
                token_id, logprob = self._sample_token(logits, sampling_params)
                token_ids.append(int(token_id))
                generated_logprobs.append(float(logprob))

                decoded = self._tokenizer.decode(token_ids[len(prompt_tokens) :])
                if stop_sequences and any(decoded.endswith(stop) for stop in stop_sequences):
                    break

            text = self._tokenizer.decode(token_ids)
            parsed = self._parse_response(model_input, text)
            outputs.append(
                {
                    "text": text,
                    "token_ids": token_ids,
                    "logprobs": generated_logprobs,
                    "parsed_response": parsed,
                    "prompt_tokens": prompt_token_count,
                    "tokenizer_time_s": tokenizer_time_s,
                    "gpu_utilization": gpu_util,
                }
            )
        return outputs

    def _prepare_prompt(self, model_input: ModelInput) -> tuple[torch.Tensor, float]:
        renderer = model_input.metadata.get("renderer") if model_input.metadata else None
        messages = model_input.metadata.get("messages") if model_input.metadata else None
        if renderer and messages:
            prompt_text = renderer.build_generation_prompt(messages)
            self._stop_sequences = renderer.get_stop_sequences()
            start = time.perf_counter()
            encoded = self._tokenizer.encode(prompt_text)
            tokenizer_time = time.perf_counter() - start
            return torch.tensor(encoded, dtype=torch.long), tokenizer_time
        if not model_input.token_chunks:
            raise ValueError("ModelInput must contain at least one token chunk")
        self._stop_sequences = []
        tensor = model_input.token_chunks[0]
        return tensor, 0.0

    def _parse_response(self, model_input: ModelInput, text: str) -> str | None:
        renderer = model_input.metadata.get("renderer") if model_input.metadata else None
        if renderer:
            return renderer.parse_response(text)
        return None

    def _sample_token(self, logits: torch.Tensor, params: SamplingParams) -> tuple[int, float]:
        if params.temperature <= 0:
            raise ValueError("Temperature must be > 0")
        logits = logits / params.temperature
        probs = F.softmax(logits, dim=-1)
        if params.top_k is not None:
            topk = min(params.top_k, probs.numel())
            values, indices = torch.topk(probs, topk)
            probs = values / values.sum()
            token_candidates = indices
        else:
            token_candidates = torch.arange(probs.numel(), device=probs.device)
        if params.top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative <= params.top_p
            if sorted_probs.numel() > 0:
                mask[..., 0] = True
            filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            if filtered.sum() == 0:
                filtered = sorted_probs
            filtered = filtered / filtered.sum()
            choice = torch.multinomial(filtered, 1).item()
            token_id = token_candidates[sorted_idx[choice]].item()
            logprob = torch.log(filtered[choice] + 1e-12).item()
            return token_id, logprob
        sampled_index = torch.multinomial(probs, 1).item()
        token_id = token_candidates[sampled_index].item()
        logprob = torch.log(probs[sampled_index] + 1e-12).item()
        return token_id, logprob


@ray.remote
class RewardActor:
    """Reward actor wrapping a user supplied callable."""

    def __init__(self, reward_fn: Callable[..., float]):
        self._reward_fn = reward_fn

    def compute(self, *args, **kwargs) -> float:
        return float(self._reward_fn(*args, **kwargs))


__all__ = ["LearnerActor", "SamplerActor", "RewardActor", "ForwardBackwardPayload"]
