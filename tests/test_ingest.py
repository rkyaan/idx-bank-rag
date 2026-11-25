"""Tests for PDF extraction and chunk assembly."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pymupdf
import pytest

from rag.config import Settings
from rag.ingest import (
    SourceDocument,
    build_chunks,
    clean_text,
    discover_documents,
    extract_pages,
    sha256_file,
)
from rag.schemas import Page


class StubTokenizer:
    """Whitespace tokenizer, so tests stay offline and fast."""

    def encode(self, text: str) -> SimpleNamespace:
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return SimpleNamespace(offsets=offsets)


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "BBCA_2024.pdf"
    document = pymupdf.open()
    for page_number in range(1, 4):
        page = document.new_page()
        body = " ".join(f"kata{page_number}x{index}" for index in range(60))
        page.insert_text((72, 72), f"Halaman {page_number} laporan tahunan", fontsize=11)
        page.insert_text((72, 100), body, fontsize=11)
    document.save(path)
    document.close()
    return path


def test_clean_text_drops_page_numbers_and_collapses_space() -> None:
    raw = "Laba bersih\n\n  123  \nnaik\t8 persen.\n"
    assert clean_text(raw) == "Laba bersih naik 8 persen."


def test_clean_text_removes_soft_hyphens() -> None:
    assert clean_text("pem\u00adbiayaan") == "pembiayaan"


def test_discover_documents_matches_expected_names(sample_pdf, tmp_path) -> None:
    (tmp_path / "notes.pdf").write_bytes(b"not a report")
    documents = discover_documents(tmp_path)
    assert [document.doc_id for document in documents] == ["BBCA_2024"]
    assert documents[0].bank == "BBCA"
    assert documents[0].year == 2024


def test_extract_pages_keeps_one_based_page_numbers(sample_pdf) -> None:
    document = SourceDocument("BBCA_2024", "BBCA", 2024, sample_pdf)
    pages = extract_pages(document)
    assert [page.page for page in pages] == [1, 2, 3]
    assert all(page.doc_id == "BBCA_2024" for page in pages)
    assert "laporan tahunan" in pages[0].text


def test_build_chunks_never_crosses_a_page() -> None:
    pages = [
        Page("BBCA_2024", "BBCA", 2024, 1, " ".join(f"a{i}" for i in range(50))),
        Page("BBCA_2024", "BBCA", 2024, 2, " ".join(f"b{i}" for i in range(50))),
    ]
    settings = Settings(
        chunk_min_tokens=10,
        chunk_max_tokens=20,
        chunk_overlap_ratio=0.1,
        min_indexable_tokens=1,
    )
    chunks = build_chunks(pages, StubTokenizer(), settings)

    assert chunks
    for chunk in chunks:
        assert chunk.chunk_id.startswith(f"BBCA_2024-p{chunk.page:04d}-c")
        assert all(word.startswith("a") for word in chunk.text.split()) or all(
            word.startswith("b") for word in chunk.text.split()
        )
        assert chunk.n_tokens <= 20


def test_chunk_ids_are_unique() -> None:
    pages = [Page("BMRI_2023", "BMRI", 2023, 1, " ".join(f"x{i}" for i in range(120)))]
    settings = Settings(
        chunk_min_tokens=10,
        chunk_max_tokens=20,
        chunk_overlap_ratio=0.1,
        min_indexable_tokens=1,
    )
    chunks = build_chunks(pages, StubTokenizer(), settings)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_sha256_is_stable(sample_pdf) -> None:
    first = sha256_file(sample_pdf)
    assert first == sha256_file(sample_pdf)
    assert len(first) == 64


def test_short_pages_are_dropped_from_the_index() -> None:
    pages = [
        Page("BBCA_2024", "BBCA", 2024, 1, "O1 Laporan Manajemen"),
        Page("BBCA_2024", "BBCA", 2024, 2, " ".join(f"y{i}" for i in range(100))),
    ]
    settings = Settings(
        chunk_min_tokens=30,
        chunk_max_tokens=40,
        chunk_overlap_ratio=0.1,
        min_indexable_tokens=30,
    )
    chunks = build_chunks(pages, StubTokenizer(), settings)
    assert chunks
    assert all(chunk.page == 2 for chunk in chunks)
