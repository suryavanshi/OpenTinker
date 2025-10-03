"""Qwen style chat renderer."""
from __future__ import annotations

from typing import List, Sequence

from .base import ChatMessage, Renderer


class QwenRenderer(Renderer):
    SYSTEM_PREFIX = "<|im_start|>system\n"
    USER_PREFIX = "<|im_start|>user\n"
    ASSISTANT_PREFIX = "<|im_start|>assistant\n"
    SUFFIX = "<|im_end|>"

    def build_generation_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt_parts: List[str] = []
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"{self.SYSTEM_PREFIX}{message.content}{self.SUFFIX}\n")
            elif message.role == "user":
                prompt_parts.append(f"{self.USER_PREFIX}{message.content}{self.SUFFIX}\n")
            elif message.role == "assistant":
                prompt_parts.append(f"{self.ASSISTANT_PREFIX}{message.content}{self.SUFFIX}\n")
            else:
                raise ValueError(f"Unsupported role: {message.role}")
        prompt_parts.append(f"{self.ASSISTANT_PREFIX}")
        return "".join(prompt_parts)

    def get_stop_sequences(self) -> List[str]:
        return [self.SUFFIX]

    def parse_response(self, text: str) -> str:
        return text.split(self.SUFFIX, 1)[0]


__all__ = ["QwenRenderer"]
