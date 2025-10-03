"""Utilities for serialising and managing training checkpoints."""
from __future__ import annotations

import json
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Mapping, Optional

import torch
import yaml
from safetensors.torch import save_file

from .tokenizer import SimpleTokenizer


@dataclass(slots=True)
class CheckpointState:
    """Container describing the artefacts saved to disk."""

    path: Path
    step: int
    global_step: int


class CheckpointManager:
    """Coordinates periodic checkpointing during training loops."""

    def __init__(
        self,
        *,
        training_client,
        tokenizer: SimpleTokenizer,
        output_dir: Path,
        every_steps: int = 0,
        keep_last: Optional[int] = None,
        save_optimizer: bool = True,
        save_tokenizer: bool = True,
        save_config: bool = True,
    ) -> None:
        self._training = training_client
        self._tokenizer = tokenizer
        self._output_dir = output_dir
        self._every_steps = max(int(every_steps), 0)
        self._keep_last = keep_last if keep_last is None else max(int(keep_last), 0)
        self._save_optimizer = bool(save_optimizer)
        self._save_tokenizer = bool(save_tokenizer)
        self._save_config = bool(save_config)
        self._retained: Deque[Path] = deque()
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def maybe_save(self, step: int, *, training_config: Mapping[str, object] | None = None) -> CheckpointState | None:
        """Saves a checkpoint if the cadence allows for ``step``."""

        if self._every_steps <= 0:
            return None
        if (step + 1) % self._every_steps != 0:
            return None
        return self.save(step=step, training_config=training_config)

    def save(self, *, step: int, training_config: Mapping[str, object] | None = None) -> CheckpointState:
        """Persists the latest weights and optimiser state to disk."""

        state = self._training.save_state()
        exported = self._training.export_lora_weights()
        adapters = exported.get("adapters", {})
        adapter_tensors = {key: tensor.cpu() for key, tensor in adapters.items()}

        global_step = int(state.get("global_step", step + 1))
        checkpoint_dir = self._output_dir / f"step_{global_step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        save_file(adapter_tensors, str(checkpoint_dir / "adapter.safetensors"))
        torch.save(state, checkpoint_dir / "trainer_state.pt")

        if self._save_optimizer and state.get("optimiser") is not None:
            torch.save(state["optimiser"], checkpoint_dir / "optimizer.pt")

        if self._save_tokenizer:
            self._write_tokenizer(checkpoint_dir / "tokenizer.json")

        metadata = {
            "step_index": step,
            "global_step": global_step,
            "has_optimizer": state.get("optimiser") is not None,
            "accumulation_progress": state.get("accumulation_progress", 0),
        }
        with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        if self._save_config and training_config is not None:
            self._write_config(checkpoint_dir / "config.yaml", training_config)

        self._retained.append(checkpoint_dir)
        self._enforce_retention()

        return CheckpointState(path=checkpoint_dir, step=step, global_step=global_step)

    def list_checkpoints(self) -> Iterable[Path]:
        """Returns the retained checkpoint directories in chronological order."""

        return tuple(self._retained)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write_tokenizer(self, path: Path) -> None:
        payload = {
            "unk_token": self._tokenizer.unk_token,
            "pad_token": self._tokenizer.pad_token,
            "vocab": self._tokenizer.vocab,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_config(self, path: Path, config: Mapping[str, object]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_to_python(config), handle, sort_keys=False)

    def _enforce_retention(self) -> None:
        if self._keep_last is None:
            return
        while len(self._retained) > self._keep_last:
            oldest = self._retained.popleft()
            shutil.rmtree(oldest, ignore_errors=True)


def _to_python(obj):
    if isinstance(obj, Mapping):
        return {key: _to_python(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(value) for value in obj]
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    if hasattr(obj, "__dict__"):
        return _to_python(vars(obj))
    return repr(obj)


__all__ = ["CheckpointManager", "CheckpointState"]
