"""Checks for packing and fetching the release index bundle."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from rag import index_bundle
from rag.config import BANKS


def make_index(root: Path) -> Path:
    """Write the smallest directory tree that counts as a complete index."""
    (root / "faiss").mkdir(parents=True)
    (root / "chunks.jsonl").write_text('{"chunk_id": "a"}\n', encoding="utf-8")
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    for bank in BANKS:
        (root / "faiss" / f"{bank}.faiss").write_bytes(b"faiss")
    return root


def test_a_fresh_clone_is_not_ready(tmp_path: Path) -> None:
    assert index_bundle.is_ready(tmp_path) is False


def test_a_missing_faiss_file_is_not_ready(tmp_path: Path) -> None:
    make_index(tmp_path)
    (tmp_path / "faiss" / "BRIS.faiss").unlink()

    assert index_bundle.is_ready(tmp_path) is False


def test_pack_then_fetch_round_trips_the_index(tmp_path: Path) -> None:
    source = make_index(tmp_path / "source")
    archive = index_bundle.pack(source, tmp_path / "bundle.tar.gz")

    target = tmp_path / "target"
    index_bundle.ensure_index(
        target,
        url=archive.as_uri(),
        sha256=index_bundle.sha256_of(archive),
    )

    assert index_bundle.is_ready(target)
    assert (target / "chunks.jsonl").read_text(encoding="utf-8") == '{"chunk_id": "a"}\n'
    assert (target / "faiss" / "BBRI.faiss").read_bytes() == b"faiss"


def test_an_existing_index_is_not_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_index(tmp_path / "index")

    def explode(*args, **kwargs):
        raise AssertionError("should not download when the index is already there")

    monkeypatch.setattr(index_bundle, "_download", explode)

    index_bundle.ensure_index(tmp_path / "index", url="https://example.invalid/x.tar.gz")


def test_a_bad_checksum_leaves_nothing_behind(tmp_path: Path) -> None:
    source = make_index(tmp_path / "source")
    archive = index_bundle.pack(source, tmp_path / "bundle.tar.gz")
    target = tmp_path / "target"

    with pytest.raises(index_bundle.IndexUnavailable, match="checksum mismatch"):
        index_bundle.ensure_index(target, url=archive.as_uri(), sha256="0" * 64)

    assert not index_bundle.is_ready(target)


def test_a_failed_download_is_reported_as_unavailable(tmp_path: Path) -> None:
    with pytest.raises(index_bundle.IndexUnavailable, match="Could not download"):
        index_bundle.ensure_index(
            tmp_path / "index",
            url=(tmp_path / "missing.tar.gz").as_uri(),
            sha256="0" * 64,
        )


def test_a_bundle_without_a_checksum_is_refused(tmp_path: Path) -> None:
    with pytest.raises(index_bundle.IndexUnavailable, match="checksum"):
        index_bundle.ensure_index(
            tmp_path / "index", url="https://example.invalid/x.tar.gz", sha256=""
        )


def test_extraction_rejects_paths_outside_the_target(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="../escaped.txt")
        info.size = 0
        tar.addfile(info)

    with pytest.raises(index_bundle.IndexUnavailable, match="unsafe path"):
        index_bundle._extract(archive, tmp_path / "unpacked")

    assert not (tmp_path / "escaped.txt").exists()


def test_packing_refuses_an_incomplete_index(tmp_path: Path) -> None:
    with pytest.raises(index_bundle.IndexUnavailable, match="No complete index"):
        index_bundle.pack(tmp_path / "empty", tmp_path / "bundle.tar.gz")


def test_the_published_checksum_is_a_real_digest() -> None:
    assert len(index_bundle.INDEX_BUNDLE_SHA256) == 64
    assert all(character in "0123456789abcdef" for character in index_bundle.INDEX_BUNDLE_SHA256)
