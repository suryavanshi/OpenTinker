from __future__ import annotations

import torch

from rinker.core.rendering import (
    ChatMessage,
    ChatMessagePart,
    QwenRenderer,
    QwenVLMRenderer,
)
from rinker.core.types import ModelInput, SamplingParams
from rinker.ray_runtime.actors import SamplerActor
from rinker.utils.tokenizer import SimpleTokenizer


class DummyProcessor:
    def __init__(self) -> None:
        self.last_messages = None
        self.called_kwargs = None

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        self.last_messages = messages
        assert add_generation_prompt is True
        assert tokenize is False
        return "PROMPT"

    def __call__(self, **kwargs):
        self.called_kwargs = kwargs
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        return {"input_ids": input_ids, "images": kwargs.get("images")}


def test_qwen_renderer_accepts_text_parts():
    renderer = QwenRenderer()
    messages = [
        ChatMessage(role="system", content="Be helpful"),
        ChatMessage(role="user", content=[ChatMessagePart(type="text", text="Hi")]),
    ]
    prompt = renderer.build_generation_prompt(messages)
    assert "Hi" in prompt


def test_qwen_vlm_renderer_prepares_processor_payload():
    processor = DummyProcessor()
    renderer = QwenVLMRenderer(processor=processor, max_pixels=16)
    image = torch.zeros(3, 2, 2)
    messages = [
        ChatMessage(role="system", content="You are a captioner."),
        ChatMessage(
            role="user",
            content=[
                ChatMessagePart(type="text", text="Describe"),
                ChatMessagePart(type="image", image=image),
            ],
        ),
    ]
    payload = renderer.build_processor_inputs(messages)
    assert "input_ids" in payload
    assert payload["images"][0].shape[-1] == 2
    prompt = renderer.build_generation_prompt(messages)
    assert prompt == "PROMPT"
    assert processor.last_messages[1]["content"][1]["type"] == "image"


class _FakeVisionRenderer(QwenRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.attached_processor = None

    def attach_processor(self, processor, *, max_pixels=None):
        self.attached_processor = processor

    def build_processor_inputs(self, messages, *, attachments=None):
        return {"input_ids": torch.tensor([1, 2], dtype=torch.long)}


def test_sampler_actor_emits_token_logprobs_and_processor_inputs():
    tokenizer = SimpleTokenizer()
    renderer = _FakeVisionRenderer()
    messages = [ChatMessage(role="user", content="Hi")]
    prompt_tokens = torch.tensor(tokenizer.encode("Hi"), dtype=torch.long)
    model_input = ModelInput(
        token_chunks=[prompt_tokens],
        metadata={"renderer": renderer, "messages": messages},
    )
    actor_cls = SamplerActor.__ray_metadata__.modified_class
    actor = actor_cls(
        tokenizer,
        hidden_size=32,
        vision_processor={"name": "dummy"},
    )
    params = SamplingParams(max_new_tokens=2, temperature=1.0)
    result = actor.generate(model_input, params, num_samples=1)[0]
    assert renderer.attached_processor == {"name": "dummy"}
    assert len(result["token_logprobs"]) == len(result["token_ids"])
    assert "processor_inputs" in result

