import re
import json
from pathlib import Path

__all__ = ["RegexTokenizer"]


class RegexTokenizer:
    def __init__(self, vocab: dict[str, int] | str, pattern: str, max_tokens: int, padding_idx: int) -> None:
        if isinstance(vocab, (str, Path)):
            with open(vocab, "r") as f:
                vocab = json.load(f)
        self.vocab = vocab
        self.padding_idx = padding_idx
        assert padding_idx not in vocab.values()
        self.vocab_size = len(self.vocab) + 1
        self.pattern = pattern
        self.max_tokens = max_tokens

    def __call__(self, string: str) -> list[int]:
        tokens = re.findall(self.pattern, string)[: self.max_tokens]
        tokens = [self.vocab[token] for token in tokens if token in self.vocab]
        pad = self.max_tokens - len(tokens)
        if pad > 0:
            tokens += [self.padding_idx] * pad
        return tokens
