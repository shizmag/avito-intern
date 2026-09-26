"""Deterministic text representations shared by retrieval and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Callable


@dataclass(frozen=True)
class TextRepresentations:
    raw: str | None
    normalized: str
    tokens: tuple[str, ...]
    lemmatized: str | None
    was_missing: bool
    was_truncated: bool


def normalize_text(
    text: str | None, *, replace_yo: bool = True, max_chars: int | None = None
) -> tuple[str, bool]:
    value = "" if text is None else unicodedata.normalize("NFKC", str(text)).lower()
    if replace_yo:
        value = value.replace("ё", "е")
    value = re.sub(r"\s+", " ", value).strip()
    truncated = max_chars is not None and len(value) > max_chars
    if truncated:
        value = value[:max_chars].rstrip()
    return value, truncated


def tokenize(
    text: str, *, pattern: str = r"[\w]+", min_length: int = 1
) -> tuple[str, ...]:
    if min_length < 1:
        raise ValueError("min_length must be positive")
    return tuple(
        token
        for token in re.findall(pattern, text, flags=re.UNICODE)
        if len(token) >= min_length
    )


def lemmatize_tokens(
    tokens: tuple[str, ...], lemmatizer: Callable[[str], str] | None = None
) -> tuple[str, ...]:
    if lemmatizer is None:
        return tokens
    return tuple(lemmatizer(token) for token in tokens)


def represent_text(
    text: str | None,
    *,
    replace_yo: bool = True,
    token_pattern: str = r"[\w]+",
    min_token_length: int = 1,
    lemma: bool = False,
    lemmatizer: Callable[[str], str] | None = None,
    max_chars: int | None = None,
) -> TextRepresentations:
    normalized, truncated = normalize_text(
        text, replace_yo=replace_yo, max_chars=max_chars
    )
    tokens = tokenize(normalized, pattern=token_pattern, min_length=min_token_length)
    lemma_text = " ".join(lemmatize_tokens(tokens, lemmatizer)) if lemma else None
    return TextRepresentations(
        text, normalized, tokens, lemma_text, text is None, truncated
    )


def compose_text(*parts: str | None, separator: str = " ") -> str:
    return separator.join(part for part in parts if part)
