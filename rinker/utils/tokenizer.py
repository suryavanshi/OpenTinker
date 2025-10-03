"""Tokenizer utilities and adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Protocol, Sequence


class TokenizerProtocol(Protocol):
    """Minimal protocol implemented by tokenizers used in the runtime."""

    @property
    def vocab_size(self) -> int:  # pragma: no cover - protocol definition
        ...

    def encode(self, text: str) -> List[int]:  # pragma: no cover - protocol definition
        ...

    def decode(self, token_ids: Iterable[int]) -> str:  # pragma: no cover - protocol definition
        ...


@dataclass
class SimpleTokenizer(TokenizerProtocol):
    """Character-level tokenizer compatible with our toy models."""

    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    vocab: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.vocab:
            # initialise with printable ASCII characters
            import string

            characters = list(string.printable)
            self.vocab = {ch: idx for idx, ch in enumerate(characters)}
            self.vocab[self.unk_token] = len(self.vocab)
            self.vocab[self.pad_token] = len(self.vocab)
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(ch, self.vocab[self.unk_token]) for ch in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.inv_vocab.get(int(idx), self.unk_token) for idx in token_ids)


@dataclass
class HFTokenizerWrapper(TokenizerProtocol):
    """Wraps a Hugging Face tokenizer to match the local protocol."""

    tokenizer: "PreTrainedTokenizerBase"
    decode_skip_special_tokens: bool = False

    @property
    def vocab_size(self) -> int:
        return int(getattr(self.tokenizer, "vocab_size", len(self.tokenizer)))

    def encode(self, text: str) -> List[int]:
        return list(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def decode(self, token_ids: Iterable[int]) -> str:
        if isinstance(token_ids, Sequence):
            ids = list(int(idx) for idx in token_ids)
        else:
            ids = [int(idx) for idx in token_ids]
        return self.tokenizer.decode(
            ids,
            skip_special_tokens=self.decode_skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )


__all__ = ["TokenizerProtocol", "SimpleTokenizer", "HFTokenizerWrapper"]
