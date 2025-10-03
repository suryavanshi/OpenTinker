"""Renderer abstractions for chat style prompting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class ChatMessage:
    role: str
    content: str


class Renderer:
    """Base renderer for chat-oriented models."""

    def build_generation_prompt(self, messages: Sequence[ChatMessage]) -> str:
        raise NotImplementedError

    def get_stop_sequences(self) -> List[str]:
        return []

    def parse_response(self, text: str) -> str:
        return text


__all__ = ["Renderer", "ChatMessage"]
