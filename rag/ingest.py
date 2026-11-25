"""Build the search index from the annual report PDFs.

Usage:
    python -m rag.ingest            # build, or skip when nothing changed
    python -m rag.ingest --force    # rebuild from scratch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
import pymupdf
from tokenizers import Tokenizer

from rag.chunking import chunk_page, token_offsets
from rag.config import BANKS, EMBED_DIM, Settings
from rag.embeddings import Embedder
from rag.schemas import Chunk, Page

DOC_PATTERN = re.compile(r"^(?P<bank>BBCA|BBRI|BMRI|BBNI|BRIS)_(?P<year>20\d{2})\.pdf$")
PAGE_NUMBER_LINE = re.compile(r"^(?:hlm\.?\s*)?\d{1,4}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    doc_id: str
    bank: str
    year: int
    path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_documents(data_dir: Path) -> list[SourceDocument]:
    """Find files named like ``BBCA_2024.pdf`` in the data directory."""
    documents: list[SourceDocument] = []
    for path in sorted(data_dir.glob("*.pdf")):
        match = DOC_PATTERN.match(path.name)
        if not match:
            continue
        bank = match["bank"]
        year = int(match["year"])
        documents.append(SourceDocument(f"{bank}_{year}", bank, year, path))
    return documents


def clean_text(raw: str) -> str:
    """Collapse layout noise so token counts reflect real content."""
    lines: list[str] = []
    for line in raw.replace("\u00ad", "").splitlines():
        stripped = line.strip()
        if not stripped or PAGE_NUMBER_LINE.match(stripped):
            continue
        lines.append(stripped)
    return " ".join(" ".join(lines).split())


def extract_pages(document: SourceDocument) -> list[Page]:
    pages: list[Page] = []
    with pymupdf.open(document.path) as pdf:
        for number, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text())
            if text:
                pages.append(Page(document.doc_id, document.bank, document.year, number, text))
    return pages


def fit_to_limit(text: str, tokenizer: Tokenizer, max_tokens: int) -> tuple[str, int]:
    """Cut at a token boundary when standalone tokenization pushes past the cap.

    A page and a slice of that page do not always tokenize to the same length,
    so the window can land one token over the limit. Trimming keeps the promise
    that no chunk exceeds ``max_tokens``.
    """
    offsets = token_offsets(tokenizer.encode(text).offsets)
    if len(offsets) <= max_tokens:
        return text, len(offsets)
    trimmed = text[: offsets[max_tokens - 1][1]].rstrip()
    offsets = token_offsets(tokenizer.encode(trimmed).offsets)
    while len(offsets) > max_tokens:
        trimmed = trimmed[: offsets[max_tokens - 1][1]].rstrip()
        offsets = token_offsets(tokenizer.encode(trimmed).offsets)
    return trimmed, len(offsets)


def build_chunks(pages: list[Page], tokenizer: Tokenizer, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        offsets = token_offsets(tokenizer.encode(page.text).offsets)
        pieces = chunk_page(
            page.text,
            offsets,
            min_tokens=settings.chunk_min_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_ratio=settings.chunk_overlap_ratio,
        )
        for index, piece in enumerate(pieces, start=1):
            piece, n_tokens = fit_to_limit(piece, tokenizer, settings.chunk_max_tokens)
            if n_tokens < settings.min_indexable_tokens:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{page.doc_id}-p{page.page:04d}-c{index:02d}",
                    doc_id=page.doc_id,
                    bank=page.bank,
                    year=page.year,
                    page=page.page,
                    text=piece,
                    n_tokens=n_tokens,
                )
            )
    return chunks


def chunking_signature(settings: Settings) -> dict:
    return {
        "min_tokens": settings.chunk_min_tokens,
        "max_tokens": settings.chunk_max_tokens,
        "overlap_ratio": settings.chunk_overlap_ratio,
    }


def index_is_current(index_dir: Path, documents: list[SourceDocument], settings: Settings) -> bool:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("embedding_model") != settings.embed_model:
        return False
    if manifest.get("chunking") != chunking_signature(settings):
        return False
    recorded = {doc["doc_id"]: doc["sha256"] for doc in manifest.get("documents", [])}
    current = {doc.doc_id: sha256_file(doc.path) for doc in documents}
    if recorded != current:
        return False
    banks = {doc.bank for doc in documents}
    return all((index_dir / "faiss" / f"{bank}.faiss").exists() for bank in banks)


def build_faiss_index(vectors: np.ndarray, mask: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(np.ascontiguousarray(vectors[mask], dtype=np.float32))
    return index


def write_index(
    index_dir: Path,
    chunks: list[Chunk],
    vectors: np.ndarray,
    records: list[dict],
    settings: Settings,
    elapsed: float,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir = index_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = index_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    banks = np.array([chunk.bank for chunk in chunks])
    for bank in BANKS:
        mask = banks == bank
        if not mask.any():
            continue
        faiss.write_index(build_faiss_index(vectors, mask), str(faiss_dir / f"{bank}.faiss"))

    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "build_seconds": round(elapsed, 1),
        "embedding_model": settings.embed_model,
        "embed_dim": EMBED_DIM,
        "chunking": chunking_signature(settings),
        "documents": records,
        "totals": {
            "documents": len(records),
            "pages": sum(record["pages"] for record in records),
            "chunks": len(chunks),
            "tokens": sum(chunk.n_tokens for chunk in chunks),
        },
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Build the annual report search index.")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--force", action="store_true", help="rebuild even when unchanged")
    args = parser.parse_args(argv)

    documents = discover_documents(args.data_dir)
    if not documents:
        print(f"No PDFs matching BBCA_2024.pdf etc. found in {args.data_dir}", file=sys.stderr)
        return 1

    found = {(doc.bank, doc.year) for doc in documents}
    missing = [
        f"{bank}_{year}" for bank in BANKS for year in (2023, 2024) if (bank, year) not in found
    ]
    if missing:
        print(f"Warning: missing reports: {', '.join(missing)}", file=sys.stderr)

    if not args.force and index_is_current(args.index_dir, documents, settings):
        print(f"Index in {args.index_dir} is up to date. Use --force to rebuild.")
        return 0

    started = time.perf_counter()
    tokenizer = Tokenizer.from_pretrained(settings.embed_model)
    embedder = Embedder(settings.embed_model)

    all_chunks: list[Chunk] = []
    all_vectors: list[np.ndarray] = []
    records: list[dict] = []

    for document in documents:
        doc_started = time.perf_counter()
        pages = extract_pages(document)
        chunks = build_chunks(pages, tokenizer, settings)
        vectors = embedder.embed_passages([chunk.text for chunk in chunks])
        all_chunks.extend(chunks)
        all_vectors.append(vectors)
        records.append(
            {
                "doc_id": document.doc_id,
                "bank": document.bank,
                "year": document.year,
                "file": document.path.name,
                "sha256": sha256_file(document.path),
                "pages": len(pages),
                "chunks": len(chunks),
                "tokens": sum(chunk.n_tokens for chunk in chunks),
            }
        )
        print(
            f"{document.doc_id}: {len(pages)} pages, {len(chunks)} chunks "
            f"in {time.perf_counter() - doc_started:.1f}s"
        )

    vectors = np.vstack(all_vectors) if all_vectors else np.zeros((0, EMBED_DIM), np.float32)
    elapsed = time.perf_counter() - started
    write_index(args.index_dir, all_chunks, vectors, records, settings, elapsed)

    size_mb = (args.index_dir / "chunks.jsonl").stat().st_size / 1e6
    print(
        f"Indexed {len(all_chunks)} chunks from {len(records)} documents in {elapsed:.0f}s "
        f"({size_mb:.1f} MB of text)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
