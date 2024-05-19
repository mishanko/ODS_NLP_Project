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


# class TokenizerFromSklearnVectorizer:
#     def __init__(self, vectorizer: TfidfVectorizer, max_tokens: int) -> None:
#         self.vocab = vectorizer.vocabulary_
#         self.pad_value = max(vectorizer.vocabulary_.values()) + 1
#         self.vocab_size = len(self.vocab) + 1
#         self.pattern = vectorizer.token_pattern
#         self.max_tokens = max_tokens

#     def __call__(self, string: str) -> list[int]:
#         tokens = re.findall(self.pattern, string)[: self.max_tokens]
#         tokens = [self.vocab[token] for token in tokens]
#         pad = self.max_tokens - len(tokens)
#         if pad > 0:
#             tokens += [self.pad_value] * pad
#         return tokens
