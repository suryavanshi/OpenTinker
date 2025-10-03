"""Minimal future-like wrapper used by the synchronous API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class ImmediateFuture(Generic[T]):
    value: T

    def result(self) -> T:
        return self.value


__all__ = ["ImmediateFuture"]
