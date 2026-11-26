"""Tests for dense search, BM25, and reciprocal rank fusion."""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from rag.config import Settings
from rag.retriever import Retriever, expand_query, reciprocal_rank_fusion, tokenize
from rag.schemas import Chunk


class FixedEmbedder:
    """Returns the same vector for every query, so tests stay offline."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.vector


class ExplodingEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        raise AssertionError("BM25-only search must not call the embedding model")


class RecordingEmbedder:
    """Keeps the text it was asked to embed, so tests can inspect the query."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.asarray([1.0, 0.0], dtype=np.float32)


def make_chunk(chunk_id: str, bank: str, text: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"{bank}_2024",
        bank=bank,
        year=2024,
        page=page,
        text=text,
        n_tokens=len(text.split()),
    )


def make_retriever(
    chunks: list[Chunk],
    vectors: list[list[float]],
    embedder,
    settings: Settings | None = None,
) -> Retriever:
    positions: dict[str, list[int]] = {}
    for position, chunk in enumerate(chunks):
        positions.setdefault(chunk.bank, []).append(position)

    indexes = {}
    for bank, owned in positions.items():
        matrix = np.asarray([vectors[position] for position in owned], dtype=np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        indexes[bank] = index

    return Retriever(chunks, indexes, embedder, settings or Settings())


def test_tokenize_lowercases_and_splits_words() -> None:
    assert tokenize("Rasio NPL turun 0,5% pada 2024.") == [
        "rasio",
        "npl",
        "turun",
        "0",
        "5",
        "pada",
        "2024",
    ]


def test_rrf_rewards_documents_found_by_both_methods() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] > fused[1][1]


def test_rrf_is_deterministic_on_ties() -> None:
    first = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    second = reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60)
    assert first == second


def test_vector_search_returns_the_closest_chunk() -> None:
    chunks = [
        make_chunk("c-near", "BBCA", "laba bersih"),
        make_chunk("c-far", "BBCA", "kredit bermasalah"),
    ]
    retriever = make_retriever(chunks, [[1.0, 0.0], [0.0, 1.0]], FixedEmbedder([1.0, 0.0]))
    results = retriever.search("laba", method="vector", top_k=2)
    assert [result.chunk.chunk_id for result in results] == ["c-near", "c-far"]


def test_bank_filter_excludes_other_banks() -> None:
    chunks = [
        make_chunk("bca-1", "BBCA", "laba bersih bank central asia"),
        make_chunk("bri-1", "BBRI", "laba bersih bank rakyat indonesia"),
    ]
    retriever = make_retriever(chunks, [[1.0, 0.0], [1.0, 0.0]], FixedEmbedder([1.0, 0.0]))
    results = retriever.search("laba", banks=["BBRI"], top_k=5)
    assert [result.chunk.bank for result in results] == ["BBRI"]


def test_unknown_bank_is_rejected() -> None:
    retriever = make_retriever(
        [make_chunk("a", "BBCA", "teks")], [[1.0, 0.0]], FixedEmbedder([1.0, 0.0])
    )
    with pytest.raises(ValueError, match="Unknown bank"):
        retriever.search("teks", banks=["XXXX"])


def test_hybrid_keeps_a_chunk_only_bm25_finds() -> None:
    chunks = [
        make_chunk("dense-hit", "BBCA", "pendapatan bunga bersih tumbuh"),
        make_chunk("filler-a", "BBCA", "biaya operasional naik"),
        make_chunk("filler-b", "BBCA", "kredit bermasalah menurun"),
        make_chunk("sparse-hit", "BBCA", "rasio npl membaik menjadi 1,8 persen"),
    ]
    vectors = [[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 1.0]]
    retriever = make_retriever(chunks, vectors, FixedEmbedder([1.0, 0.0]))

    sparse = retriever.search("npl", method="bm25", top_k=1)
    assert [result.chunk.chunk_id for result in sparse] == ["sparse-hit"]

    hybrid = retriever.search("npl", top_k=2)
    assert {result.chunk.chunk_id for result in hybrid} == {"dense-hit", "sparse-hit"}


def test_bm25_method_never_touches_the_embedding_model() -> None:
    chunks = [
        make_chunk("a", "BBCA", "rasio npl membaik"),
        make_chunk("b", "BBCA", "pendapatan bunga bersih tumbuh"),
        make_chunk("c", "BBCA", "biaya operasional naik"),
        make_chunk("d", "BBCA", "kredit bermasalah menurun"),
    ]
    retriever = make_retriever(chunks, [[1.0, 0.0]] * 4, ExplodingEmbedder())
    results = retriever.search("npl", method="bm25")
    assert [result.chunk.chunk_id for result in results] == ["a"]


def test_top_k_is_respected() -> None:
    chunks = [make_chunk(f"c{i}", "BBCA", f"teks nomor {i}") for i in range(10)]
    vectors = [[1.0, float(i)] for i in range(10)]
    retriever = make_retriever(chunks, vectors, FixedEmbedder([1.0, 0.0]))
    assert len(retriever.search("teks", top_k=3)) == 3


def test_empty_query_returns_nothing() -> None:
    retriever = make_retriever(
        [make_chunk("a", "BBCA", "teks")], [[1.0, 0.0]], FixedEmbedder([1.0, 0.0])
    )
    assert retriever.search("   ") == []


def test_available_banks_lists_only_loaded_indexes() -> None:
    chunks = [make_chunk("a", "BBCA", "teks"), make_chunk("b", "BRIS", "teks")]
    retriever = make_retriever(chunks, [[1.0, 0.0], [1.0, 0.0]], FixedEmbedder([1.0, 0.0]))
    assert retriever.available_banks == ["BBCA", "BRIS"]


def test_expansion_appends_domain_terms_for_a_vague_question() -> None:
    expanded = expand_query("Apa risiko utama yang dilaporkan?")

    assert expanded.startswith("Apa risiko utama yang dilaporkan?")
    assert "risiko kredit" in expanded
    assert "profil risiko" in expanded


def test_expansion_leaves_a_specific_question_untouched() -> None:
    assert expand_query("Berapa laba bersih tahun 2024?") == "Berapa laba bersih tahun 2024?"


def test_expansion_ignores_terms_inside_a_longer_word() -> None:
    assert (
        expand_query("Bagaimana bank mengelola berisiko?") == "Bagaimana bank mengelola berisiko?"
    )


def test_expansion_keeps_duplicate_terms_out_of_the_query() -> None:
    table = {"risiko": "risiko kredit"}
    assert expand_query("risiko", table) == "risiko risiko kredit"


def test_search_embeds_the_expanded_query() -> None:
    chunks = [make_chunk(f"c{i}", "BBCA", f"kalimat {i} biasa") for i in range(4)]
    embedder = RecordingEmbedder()
    retriever = make_retriever(chunks, [[1.0, 0.0]] * 4, embedder)
    retriever.search("Apa risiko utama?", top_k=2)

    assert len(embedder.texts) == 1
    assert embedder.texts[0].startswith("Apa risiko utama?")
    assert "profil risiko" in embedder.texts[0]
