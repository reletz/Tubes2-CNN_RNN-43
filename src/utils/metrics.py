from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import f1_score


def _normalize_caption_tokens(caption: Sequence[str] | str) -> list[str]:
    if isinstance(caption, str):
        return caption.split()
    return [str(token) for token in caption]


def compute_bleu4(
    references: Iterable[Iterable[Sequence[str] | str]],
    hypotheses: Iterable[Sequence[str] | str],
) -> float:
    """Compute corpus BLEU-4.

    Args:
        references: iterable of reference-caption groups. Each group is an
            iterable of token sequences for the same image.
        hypotheses: iterable of predicted token sequences.
    """
    ref_tokens = [
        [_normalize_caption_tokens(ref) for ref in ref_group]
        for ref_group in references
    ]
    hyp_tokens = [_normalize_caption_tokens(hyp) for hyp in hypotheses]
    smoothing = SmoothingFunction().method1
    return float(corpus_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothing))


def compute_meteor(
    references: Iterable[Iterable[Sequence[str] | str]],
    hypotheses: Iterable[Sequence[str] | str],
) -> float:
    """Compute mean METEOR across all samples."""
    scores = []
    for ref_group, hyp in zip(references, hypotheses):
        ref_tokens = [_normalize_caption_tokens(ref) for ref in ref_group]
        hyp_tokens = _normalize_caption_tokens(hyp)
        # nltk.meteor_score uses WordNet for synonym matching. If that corpus is
        # unavailable, fall back to a lightweight METEOR-like score based on
        # exact token overlap.
        try:
            scores.append(meteor_score(ref_tokens, hyp_tokens))
        except LookupError:
            scores.append(_meteor_fallback(ref_tokens, hyp_tokens))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _meteor_fallback(references: list[list[str]], hypothesis: list[str]) -> float:
    """A small METEOR-like fallback without WordNet.

    Uses exact unigram matching only and a fragmentation penalty inspired by
    METEOR. This is intentionally lightweight so the metric works in minimal
    environments.
    """
    if not references or not hypothesis:
        return 0.0

    best_score = 0.0
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        return 0.0

    for ref in references:
        ref_len = len(ref)
        if ref_len == 0:
            continue

        ref_positions = {}
        for idx, token in enumerate(ref):
            ref_positions.setdefault(token, []).append(idx)

        matches = []
        used_ref_positions = set()
        for h_idx, token in enumerate(hypothesis):
            positions = ref_positions.get(token)
            if not positions:
                continue

            chosen = None
            for pos in positions:
                if pos not in used_ref_positions:
                    chosen = pos
                    break
            if chosen is not None:
                used_ref_positions.add(chosen)
                matches.append((h_idx, chosen))

        m = len(matches)
        if m == 0:
            continue

        precision = m / hyp_len
        recall = m / ref_len
        if precision + recall == 0:
            continue

        # METEOR uses weighted F-mean; a common approximation is recall-heavy.
        f_mean = (10 * precision * recall) / (recall + 9 * precision)

        # Fragmentation penalty: contiguous chunks of matches in the hypothesis.
        chunks = 1
        for i in range(1, len(matches)):
            if matches[i][0] != matches[i - 1][0] + 1 or matches[i][1] != matches[i - 1][1] + 1:
                chunks += 1
        frag = chunks / m
        penalty = 0.5 * (frag ** 3)

        score = f_mean * (1.0 - penalty)
        best_score = max(best_score, score)

    return float(best_score)


def compute_macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute macro F1 for classification labels."""
    return float(f1_score(y_true, y_pred, average="macro"))


def caption_length_stats(captions: Iterable[Sequence[str] | str]) -> dict[str, float]:
    """Return summary statistics for caption lengths.

    Counts tokens by splitting strings on spaces or by sequence length for token lists.
    """
    lengths = []
    for caption in captions:
        tokens = _normalize_caption_tokens(caption)
        lengths.append(len(tokens))

    if not lengths:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }

    arr = np.asarray(lengths, dtype=np.float32)
    return {
        "count": float(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }
