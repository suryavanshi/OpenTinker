"""LLaMA style chat renderer."""
from __future__ import annotations

from typing import List, Sequence

from .base import ChatMessage, Renderer


class LlamaRenderer(Renderer):
    BOS = "<s>"
    EOS = "</s>"

    def build_generation_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt_parts: List[str] = [self.BOS]
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"<<SYS>>\n{message.content}\n<</SYS>>\n\n")
            elif message.role == "user":
                prompt_parts.append(f"[INST] {message.content} [/INST]")
            elif message.role == "assistant":
                prompt_parts.append(f" {message.content} {self.EOS}")
            else:
                raise ValueError(f"Unsupported role: {message.role}")
        prompt_parts.append(" ")
        return "".join(prompt_parts)

    def get_stop_sequences(self) -> List[str]:
        return [self.EOS]

    def parse_response(self, text: str) -> str:
        return text.split(self.EOS, 1)[0].strip()


__all__ = ["LlamaRenderer"]
