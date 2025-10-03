"""Minimal future-like wrapper used by the synchronous API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import ray


T = TypeVar("T")


@dataclass
class ImmediateFuture(Generic[T]):
    value: T

    def result(self) -> T:
        return self.value


class RayFuture(Generic[T]):
    """Wrapper around a Ray ObjectRef that mimics the Future API."""

    def __init__(self, ref: "ray.ObjectRef", transform: Callable[[object], T] | None = None) -> None:
        self._ref = ref
        self._transform = transform

    @property
    def object_ref(self) -> "ray.ObjectRef":
        return self._ref

    def result(self) -> T:
        value = ray.get(self._ref)
        if self._transform is not None:
            return self._transform(value)
        return value  # type: ignore[return-value]


__all__ = ["ImmediateFuture", "RayFuture"]
