"""Tests for page-bounded token chunking."""

from __future__ import annotations

import itertools
import re

import pytest

from rag.chunking import chunk_page, token_offsets


def offsets_for(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def numbered_text(count: int) -> str:
    return " ".join(f"kata{i}" for i in range(count))


def windows(chunks: list[str]) -> list[tuple[int, int]]:
    """Map each chunk back to its inclusive word range."""
    result = []
    for chunk in chunks:
        words = chunk.split()
        first = int(words[0].removeprefix("kata"))
        result.append((first, first + len(words)))
    return result


def test_short_page_becomes_one_chunk() -> None:
    text = "Laba bersih tumbuh delapan persen."
    chunks = chunk_page(text, offsets_for(text), min_tokens=5, max_tokens=10)
    assert chunks == [text]


def test_empty_page_returns_nothing() -> None:
    assert chunk_page("   ", []) == []


def test_chunks_never_exceed_max_tokens() -> None:
    text = numbered_text(2000)
    chunks = chunk_page(text, offsets_for(text), min_tokens=40, max_tokens=45, snap_sentences=False)
    assert chunks
    for chunk in chunks:
        assert len(chunk.split()) <= 45


def test_chunks_keep_original_wording() -> None:
    text = numbered_text(1200)
    chunks = chunk_page(text, offsets_for(text), min_tokens=40, max_tokens=45, snap_sentences=False)
    for chunk in chunks:
        assert chunk in text


def test_overlap_is_about_twelve_percent() -> None:
    text = numbered_text(1200)
    chunks = chunk_page(
        text,
        offsets_for(text),
        min_tokens=40,
        max_tokens=45,
        overlap_ratio=0.12,
        snap_sentences=False,
    )
    spans = windows(chunks)
    for (_, previous_end), (next_start, _) in itertools.pairwise(spans):
        overlap = previous_end - next_start
        assert overlap == pytest.approx(5, abs=1)


def test_no_chunk_below_min_tokens_when_the_page_allows_it() -> None:
    text = numbered_text(900)
    chunks = chunk_page(text, offsets_for(text), min_tokens=40, max_tokens=45, snap_sentences=False)
    for chunk in chunks[:-1]:
        assert len(chunk.split()) >= 40


def test_sentence_snapping_does_not_break_the_limit() -> None:
    text = " ".join(f"Kata{i} adalah contoh." for i in range(200))
    chunks = chunk_page(text, offsets_for(text), min_tokens=40, max_tokens=45)
    assert chunks
    for chunk in chunks:
        assert len(chunk.split()) <= 45


def test_invalid_bounds_are_rejected() -> None:
    text = numbered_text(10)
    with pytest.raises(ValueError, match="min_tokens"):
        chunk_page(text, offsets_for(text), min_tokens=0, max_tokens=10)
    with pytest.raises(ValueError, match="overlap_ratio"):
        chunk_page(text, offsets_for(text), min_tokens=5, max_tokens=10, overlap_ratio=0.7)


def test_special_token_offsets_are_dropped() -> None:
    assert token_offsets([(0, 0), (0, 4), (4, 9), (9, 9)]) == [(0, 4), (4, 9)]
