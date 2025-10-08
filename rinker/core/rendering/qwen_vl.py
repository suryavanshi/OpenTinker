"""Renderer for Qwen-style vision-language models."""
from __future__ import annotations

import importlib.util
import math
from typing import Any, List, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F

from .base import ChatMessage, ChatMessagePart, Renderer, _iter_message_parts


class QwenVLMRenderer(Renderer):
    """Renderer that prepares processor inputs for Qwen vision-language models."""

    def __init__(
        self,
        *,
        processor: Any | None = None,
        processor_name: str | None = None,
        max_pixels: int = 1048576,
    ) -> None:
        self._processor = processor
        self._processor_name = processor_name
        self._max_pixels = int(max_pixels)
        if self._processor is None and self._processor_name is not None:
            self._processor = self._load_processor(self._processor_name)

    # ------------------------------------------------------------------
    # Renderer API
    # ------------------------------------------------------------------
    def attach_processor(
        self,
        processor: Any | None,
        *,
        max_pixels: int | None = None,
    ) -> None:
        if processor is None:
            if self._processor is None and self._processor_name is not None:
                self._processor = self._load_processor(self._processor_name)
            return
        self._processor = processor
        if max_pixels is not None:
            self._max_pixels = int(max_pixels)

    def build_generation_prompt(
        self,
        messages: Sequence[ChatMessage],
        *,
        attachments: Mapping[str, Any] | None = None,
    ) -> str:
        if self._processor is None:
            raise RuntimeError(
                "QwenVLMRenderer requires a processor; call attach_processor or "
                "initialise with one"
            )
        normalised, _ = self._normalise_messages(messages, attachments)
        apply_chat = getattr(self._processor, "apply_chat_template", None)
        if apply_chat is None:
            raise AttributeError("Processor is missing 'apply_chat_template'")
        return apply_chat(normalised, add_generation_prompt=True, tokenize=False)

    def build_processor_inputs(
        self,
        messages: Sequence[ChatMessage],
        *,
        attachments: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if self._processor is None:
            raise RuntimeError(
                "QwenVLMRenderer requires a processor; call attach_processor or "
                "initialise with one"
            )
        normalised, images = self._normalise_messages(messages, attachments)
        processor_kwargs: MutableMapping[str, Any] = {
            "messages": normalised,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        encoded = self._processor(**processor_kwargs)
        if isinstance(encoded, Mapping):
            output: MutableMapping[str, Any] = dict(encoded)
        else:
            output = {"input_ids": encoded}
        if images:
            output.setdefault("images", images)
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_processor(self, name: str) -> Any:
        if importlib.util.find_spec("transformers") is None:
            raise RuntimeError(
                "transformers must be installed to load a Qwen vision processor"
            )
        from transformers import AutoProcessor  # type: ignore

        return AutoProcessor.from_pretrained(name, trust_remote_code=True)

    def _normalise_messages(
        self,
        messages: Sequence[ChatMessage],
        attachments: Mapping[str, Any] | None,
    ) -> tuple[List[Mapping[str, Any]], List[Any]]:
        attachments = attachments or {}
        normalised: List[Mapping[str, Any]] = []
        collected_images: List[Any] = []
        for message in messages:
            content_parts: List[Mapping[str, Any]] = []
            for part in _iter_message_parts(message.content):
                if part.type == "text":
                    content_parts.append({"type": "text", "text": part.text or ""})
                    continue
                image_obj = self._resolve_image(part, attachments)
                if image_obj is None:
                    raise ValueError("Image part did not provide data or a valid reference")
                image_obj = self._ensure_max_pixels(image_obj)
                collected_images.append(image_obj)
                content_parts.append({"type": "image", "image": image_obj})
            normalised.append({"role": message.role, "content": content_parts})
        return normalised, collected_images

    def _resolve_image(
        self,
        part: ChatMessagePart,
        attachments: Mapping[str, Any],
    ) -> Any | None:
        if part.image is not None:
            return part.image
        if part.reference is not None:
            return attachments.get(part.reference)
        return None

    def _ensure_max_pixels(self, image: Any) -> Any:
        try:
            from PIL import Image
        except Exception:  # pragma: no cover - optional dependency
            Image = None  # type: ignore

        if Image is not None and isinstance(image, Image.Image):
            width, height = image.size
            if width * height <= self._max_pixels:
                return image
            scale = math.sqrt(self._max_pixels / max(width * height, 1))
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            return image.resize(new_size, Image.BICUBIC)

        if torch.is_tensor(image):
            if image.ndim < 2:
                return image
            height = int(image.shape[-2])
            width = int(image.shape[-1])
            if width * height <= self._max_pixels:
                return image
            scale = math.sqrt(self._max_pixels / max(width * height, 1))
            new_height = max(1, int(height * scale))
            new_width = max(1, int(width * scale))
            return F.interpolate(
                image.unsqueeze(0).float(),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).type_as(image)

        # Fallback: if we cannot reason about the object, return as-is.
        return image


__all__ = ["QwenVLMRenderer"]

