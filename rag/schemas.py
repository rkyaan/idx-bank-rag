"""Data records shared by ingestion, retrieval, and generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Page:
    """One extracted PDF page."""

    doc_id: str
    bank: str
    year: int
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A page-bounded passage, small enough for the embedding model."""

    chunk_id: str
    doc_id: str
    bank: str
    year: int
    page: int
    text: str
    n_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> Chunk:
        return cls(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            bank=row["bank"],
            year=int(row["year"]),
            page=int(row["page"]),
            text=row["text"],
            n_tokens=int(row["n_tokens"]),
        )


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by the retriever, with its fused score."""

    chunk: Chunk
    score: float

    def to_source(self, snippet_chars: int = 300) -> dict:
        text = self.chunk.text.strip()
        snippet = text if len(text) <= snippet_chars else text[:snippet_chars].rstrip() + "..."
        return {"doc": self.chunk.doc_id, "page": self.chunk.page, "snippet": snippet}
