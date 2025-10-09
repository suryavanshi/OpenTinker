"""Renderer abstractions for chat style prompting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Literal, Mapping, MutableSequence, Sequence


ChatMessageContent = "ChatMessagePart" | Sequence["ChatMessagePart"] | str


@dataclass
class ChatMessagePart:
    """Represents a segment within a chat message."""

    type: Literal["text", "image"]
    text: str | None = None
    image: Any | None = None
    reference: str | None = None
    mime_type: str | None = None


@dataclass
class ChatMessage:
    role: str
    content: ChatMessageContent


def _iter_message_parts(content: ChatMessageContent) -> Iterable[ChatMessagePart]:
    if isinstance(content, str):
        yield ChatMessagePart(type="text", text=content)
        return
    if isinstance(content, ChatMessagePart):
        yield content
        return
    if isinstance(content, Sequence):
        for part in content:
            if isinstance(part, ChatMessagePart):
                yield part
            else:
                raise TypeError(
                    "Chat message parts must be ChatMessagePart instances or strings"
                )
        return
    raise TypeError("Unsupported chat message content type")


def _collect_text(content: ChatMessageContent) -> str:
    parts: MutableSequence[str] = []
    for part in _iter_message_parts(content):
        if part.type != "text":
            raise ValueError("Text renderer received a non-text content part")
        parts.append(part.text or "")
    return "".join(parts)


class Renderer:
    """Base renderer for chat-oriented models."""

    def build_generation_prompt(
        self,
        messages: Sequence[ChatMessage],
        *,
        attachments: Mapping[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    def build_processor_inputs(
        self,
        messages: Sequence[ChatMessage],
        *,
        attachments: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        """Optional hook to prepare processor-specific inputs for VLMs."""

        return None

    def attach_processor(
        self,
        processor: Any | None,
        *,
        max_pixels: int | None = None,
    ) -> None:
        """Allows the sampler to lazily wire an image processor to the renderer."""

        return None

    def get_stop_sequences(self) -> List[str]:
        return []

    def parse_response(self, text: str) -> str:
        return text


__all__ = [
    "ChatMessage",
    "ChatMessagePart",
    "ChatMessageContent",
    "Renderer",
]

