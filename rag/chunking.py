"""Page-bounded token chunking.

A chunk never crosses a page boundary, so every citation points at one real page.
Window sizes are measured with the embedding model's own tokenizer, not with
whitespace counts, because the model silently truncates past its limit.
"""

from __future__ import annotations

from collections.abc import Sequence

SENTENCE_ENDINGS = (".", "!", "?", ";")


def _ends_sentence(text: str, offsets: Sequence[tuple[int, int]], index: int) -> bool:
    start, end = offsets[index]
    return text[start:end].rstrip().endswith(SENTENCE_ENDINGS)


def chunk_page(
    text: str,
    offsets: Sequence[tuple[int, int]],
    *,
    min_tokens: int = 400,
    max_tokens: int = 450,
    overlap_ratio: float = 0.12,
    snap_sentences: bool = True,
) -> list[str]:
    """Split one page into overlapping windows of at most ``max_tokens`` tokens.

    ``offsets`` are character spans for each token, so the returned strings are
    slices of the original text and keep its wording.
    """
    if min_tokens < 1 or max_tokens < min_tokens:
        raise ValueError("require 1 <= min_tokens <= max_tokens")
    if not 0.0 <= overlap_ratio < 0.5:
        raise ValueError("overlap_ratio must be in [0, 0.5)")

    total = len(offsets)
    if total == 0:
        return []
    if total <= max_tokens:
        single = text.strip()
        return [single] if single else []

    windows: list[tuple[int, int]] = []
    start = 0
    while start < total:
        hard_end = min(start + max_tokens, total)
        end = hard_end
        if snap_sentences and hard_end < total:
            floor = start + min_tokens
            for index in range(hard_end - 1, floor - 1, -1):
                if _ends_sentence(text, offsets, index):
                    end = index + 1
                    break
        windows.append((start, end))
        if end >= total:
            break
        overlap = int((end - start) * overlap_ratio)
        start = max(start + 1, end - overlap)

    if len(windows) > 1 and windows[-1][1] - windows[-1][0] < min_tokens:
        previous_start = windows[-2][0]
        if total - previous_start <= max_tokens:
            windows[-2:] = [(previous_start, total)]
        else:
            reanchored = max(0, total - max_tokens)
            if reanchored > windows[-2][0]:
                windows[-1] = (reanchored, total)

    chunks: list[str] = []
    for window_start, window_end in windows:
        begin = offsets[window_start][0]
        finish = offsets[window_end - 1][1]
        piece = text[begin:finish].strip()
        if piece:
            chunks.append(piece)
    return chunks


def token_offsets(offsets: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop special-token offsets, which are zero-width and would corrupt slices."""
    return [(start, end) for start, end in offsets if end > start]
