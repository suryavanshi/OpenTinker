"""A very small character level tokenizer used for the examples and tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class SimpleTokenizer:
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
        return "".join(self.inv_vocab.get(idx, self.unk_token) for idx in token_ids)


__all__ = ["SimpleTokenizer"]
