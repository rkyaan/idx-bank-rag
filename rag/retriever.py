"""Hybrid retrieval over the annual report index.

Dense search finds paraphrases, BM25 finds exact terms like "NPL" or "CAR", and
reciprocal rank fusion combines the two ranked lists without having to compare
cosine scores against BM25 scores.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from rag.config import BANKS, Settings
from rag.embeddings import Embedder
from rag.schemas import Chunk, RetrievedChunk

WORD_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, good enough for BM25 on Indonesian text."""
    return WORD_PATTERN.findall(text.casefold())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge ranked id lists by rank position, not by raw score."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    """Loads the index once, then answers queries against selected banks."""

    def __init__(
        self,
        chunks: list[Chunk],
        indexes: dict[str, faiss.Index],
        embedder: Embedder,
        settings: Settings,
    ) -> None:
        self.chunks = chunks
        self.indexes = indexes
        self.embedder = embedder
        self.settings = settings

        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._positions: dict[str, list[int]] = {}
        for position, chunk in enumerate(chunks):
            self._positions.setdefault(chunk.bank, []).append(position)

        self._bm25: dict[str, BM25Okapi] = {}
        for bank, positions in self._positions.items():
            corpus = [tokenize(chunks[position].text) for position in positions]
            if corpus:
                self._bm25[bank] = BM25Okapi(corpus)

    @property
    def available_banks(self) -> list[str]:
        return [bank for bank in BANKS if bank in self.indexes]

    @classmethod
    def from_index(
        cls, index_dir: Path, settings: Settings, embedder: Embedder | None = None
    ) -> Retriever:
        chunks_path = index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"No index found at {chunks_path}. Run python -m rag.ingest first."
            )

        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        indexes: dict[str, faiss.Index] = {}
        for bank in BANKS:
            path = index_dir / "faiss" / f"{bank}.faiss"
            if path.exists():
                indexes[bank] = faiss.read_index(str(path))
        if not indexes:
            raise FileNotFoundError(f"No FAISS index files found in {index_dir / 'faiss'}")

        return cls(chunks, indexes, embedder or Embedder(), settings)

    def search(
        self,
        query: str,
        banks: list[str] | None = None,
        top_k: int | None = None,
        method: str | None = None,
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []

        selected = self._selected_banks(banks)
        if not selected:
            return []

        limit = top_k or self.settings.top_k
        mode = method or self.settings.retrieval_method
        candidates = max(limit * 5, 20)

        dense = self._dense(query, selected, candidates) if mode in {"hybrid", "vector"} else []
        if mode == "vector":
            picked = dense[:limit]
        elif mode == "bm25":
            picked = self._sparse(query, selected, candidates)[:limit]
        else:
            sparse = self._sparse(query, selected, candidates)
            if not dense:
                picked = sparse[:limit]
            else:
                fused = reciprocal_rank_fusion(
                    [[chunk_id for chunk_id, _ in dense], [chunk_id for chunk_id, _ in sparse]],
                    self.settings.rrf_k,
                )
                picked = fused[:limit]

        return [
            RetrievedChunk(chunk=self._by_id[chunk_id], score=score)
            for chunk_id, score in picked
            if chunk_id in self._by_id
        ]

    def _selected_banks(self, banks: list[str] | None) -> list[str]:
        if not banks:
            return self.available_banks
        unknown = [bank for bank in banks if bank not in BANKS]
        if unknown:
            raise ValueError(f"Unknown bank(s): {', '.join(unknown)}")
        return [bank for bank in banks if bank in self.indexes]

    def _dense(self, query: str, banks: list[str], k: int) -> list[tuple[str, float]]:
        vector = self.embedder.embed_query(query).reshape(1, -1).astype(np.float32)
        hits: list[tuple[float, int]] = []
        for bank in banks:
            index = self.indexes[bank]
            count = min(k, index.ntotal)
            if count == 0:
                continue
            scores, local_ids = index.search(vector, count)
            for score, local_id in zip(scores[0], local_ids[0], strict=True):
                if local_id < 0:
                    continue
                hits.append((float(score), self._positions[bank][int(local_id)]))

        hits.sort(key=lambda item: (-item[0], item[1]))
        return [(self.chunks[position].chunk_id, score) for score, position in hits[:k]]

    def _sparse(self, query: str, banks: list[str], k: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        hits: list[tuple[float, int]] = []
        for bank in banks:
            bm25 = self._bm25.get(bank)
            if bm25 is None:
                continue
            scores = bm25.get_scores(tokens)
            if len(scores) == 0:
                continue
            count = min(k, len(scores))
            top = np.argsort(scores)[::-1][:count]
            for local_id in top:
                score = float(scores[local_id])
                if score <= 0:
                    continue
                hits.append((score, self._positions[bank][int(local_id)]))

        hits.sort(key=lambda item: (-item[0], item[1]))
        return [(self.chunks[position].chunk_id, score) for score, position in hits[:k]]


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Query the annual report index.")
    parser.add_argument("query", help="question in Indonesian")
    parser.add_argument(
        "--banks", nargs="+", default=None, help="subset of BBCA BBRI BMRI BBNI BRIS"
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--method", choices=["hybrid", "vector", "bm25"], default=None)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    args = parser.parse_args(argv)

    retriever = Retriever.from_index(args.index_dir, settings)
    try:
        results = retriever.search(args.query, args.banks, args.top_k, args.method)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if not results:
        print("No matching chunks.")
        return 0

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        preview = " ".join(chunk.text.split())[:180]
        print(f"{rank}. [{chunk.doc_id}, hlm. {chunk.page}] score={result.score:.4f}")
        print(f"   {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
