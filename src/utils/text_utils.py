from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

SPECIAL_TOKENS = ("<pad>", "<start>", "<end>", "<unk>")


def clean_caption(text: str) -> str:
    """Lowercase and remove punctuation-like characters from a caption."""
    normalized = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize_caption(text: str) -> list[str]:
    """Tokenize a caption after cleaning."""
    cleaned = clean_caption(text)
    return cleaned.split() if cleaned else []


def build_vocabulary(
    captions: Iterable[str],
    min_freq: int = 1,
    max_vocab_size: int | None = None,
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build token-index mappings from captions."""
    if min_freq <= 0:
        raise ValueError("min_freq must be >= 1")

    counter: Counter[str] = Counter()
    for caption in captions:
        counter.update(tokenize_caption(caption))

    sorted_tokens = sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))
    filtered_tokens = [token for token, freq in sorted_tokens if freq >= min_freq]

    if max_vocab_size is not None:
        if max_vocab_size <= len(special_tokens):
            raise ValueError("max_vocab_size must be larger than number of special tokens")
        filtered_tokens = filtered_tokens[: max_vocab_size - len(special_tokens)]

    word_to_idx: dict[str, int] = {}
    for token in special_tokens:
        if token in word_to_idx:
            raise ValueError(f"Duplicate special token detected: {token}")
        word_to_idx[token] = len(word_to_idx)

    for token in filtered_tokens:
        if token not in word_to_idx:
            word_to_idx[token] = len(word_to_idx)

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word


def encode_caption(
    caption: str,
    word_to_idx: dict[str, int],
    add_start_end: bool = True,
) -> list[int]:
    """Convert one caption into token IDs."""
    tokens = tokenize_caption(caption)

    if add_start_end:
        tokens = ["<start>", *tokens, "<end>"]

    unk_idx = word_to_idx.get("<unk>")
    if unk_idx is None:
        raise KeyError("word_to_idx must contain '<unk>' token")

    return [word_to_idx.get(token, unk_idx) for token in tokens]


def encode_captions(
    captions: Iterable[str],
    word_to_idx: dict[str, int],
    add_start_end: bool = True,
) -> list[list[int]]:
    """Convert multiple captions into token ID sequences."""
    return [encode_caption(caption, word_to_idx, add_start_end=add_start_end) for caption in captions]


def pad_sequences(
    sequences: Iterable[Iterable[int]],
    max_length: int | None = None,
    pad_value: int = 0,
    padding: str = "post",
    truncating: str = "post",
) -> np.ndarray:
    """Pad or truncate sequences into a fixed-length 2D NumPy array."""
    if padding not in {"pre", "post"}:
        raise ValueError("padding must be either 'pre' or 'post'")
    if truncating not in {"pre", "post"}:
        raise ValueError("truncating must be either 'pre' or 'post'")

    seq_list = [list(seq) for seq in sequences]
    if not seq_list:
        raise ValueError("sequences is empty")

    if max_length is None:
        max_length = max(len(seq) for seq in seq_list)
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    output = np.full((len(seq_list), max_length), pad_value, dtype=np.int32)

    for row_idx, seq in enumerate(seq_list):
        if not seq:
            continue

        if len(seq) > max_length:
            if truncating == "pre":
                seq = seq[-max_length:]
            else:
                seq = seq[:max_length]

        seq_len = len(seq)
        if padding == "post":
            output[row_idx, :seq_len] = seq
        else:
            output[row_idx, -seq_len:] = seq

    return output


def save_vocabulary(word_to_idx: dict[str, int], output_path: str | Path) -> None:
    """Save vocabulary mapping to JSON."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(word_to_idx, file, ensure_ascii=True, indent=2, sort_keys=True)


def load_vocabulary(vocab_path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load vocabulary mapping from JSON."""
    vocab_file = Path(vocab_path)
    with vocab_file.open("r", encoding="utf-8") as file:
        word_to_idx = json.load(file)

    casted = {str(word): int(idx) for word, idx in word_to_idx.items()}
    idx_to_word = {idx: word for word, idx in casted.items()}
    return casted, idx_to_word
