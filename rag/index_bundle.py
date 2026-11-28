"""Ship the retrieval index as a downloadable bundle.

The index is about 80 MB of derived data: far too large to keep in git, and far too
slow to rebuild on a hosted machine (the PDFs are not redistributed either). It is
published once as a release asset, and the app downloads it on first start, checks
the checksum, and unpacks it into ``index/``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from rag.config import (
    BANKS,
    DEFAULT_INDEX_DIR,
    INDEX_BUNDLE_NAME,
    INDEX_BUNDLE_SHA256,
    INDEX_BUNDLE_URL,
)

CHUNK_SIZE = 1 << 20


class IndexUnavailable(RuntimeError):
    """The index is missing and could not be downloaded."""


def is_ready(index_dir: Path) -> bool:
    """True when every file the retriever opens is already on disk."""
    if not (index_dir / "chunks.jsonl").is_file():
        return False
    return all((index_dir / "faiss" / f"{bank}.faiss").is_file() for bank in BANKS)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def _extract(archive: Path, destination: Path) -> None:
    """Unpack the bundle, rejecting absolute paths and parent-directory escapes."""
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise IndexUnavailable(f"unsafe path in index bundle: {member.name}")
        tar.extractall(destination, filter="data")


def ensure_index(
    index_dir: Path = DEFAULT_INDEX_DIR,
    *,
    url: str | None = None,
    sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Download and unpack the index bundle unless a complete index already exists."""
    if not force and is_ready(index_dir):
        return index_dir

    url = url or INDEX_BUNDLE_URL
    sha256 = sha256 if sha256 is not None else INDEX_BUNDLE_SHA256
    if not url:
        raise IndexUnavailable("No index bundle URL is configured")
    if not sha256:
        raise IndexUnavailable("No index bundle checksum is configured")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=index_dir.parent, prefix=".index-download-") as tmp:
        archive = Path(tmp) / INDEX_BUNDLE_NAME
        try:
            _download(url, archive)
        except OSError as exc:
            raise IndexUnavailable(f"Could not download the index bundle: {exc}") from exc

        actual = sha256_of(archive)
        if actual != sha256:
            raise IndexUnavailable(
                f"Index bundle checksum mismatch: expected {sha256}, got {actual}"
            )

        staging = Path(tmp) / "unpacked"
        staging.mkdir()
        _extract(archive, staging)
        for entry in staging.iterdir():
            target = index_dir / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)

    if not is_ready(index_dir):
        raise IndexUnavailable(f"Index bundle unpacked but incomplete in {index_dir}")
    return index_dir


def pack(index_dir: Path = DEFAULT_INDEX_DIR, output: Path | None = None) -> Path:
    """Write a compressed bundle of the index and print its checksum."""
    if not is_ready(index_dir):
        raise IndexUnavailable(f"No complete index to pack in {index_dir}")

    output = Path(output or Path("dist") / INDEX_BUNDLE_NAME)
    output.parent.mkdir(parents=True, exist_ok=True)

    members = [index_dir / "chunks.jsonl", index_dir / "manifest.json"]
    members += [index_dir / "faiss" / f"{bank}.faiss" for bank in BANKS]
    with tarfile.open(output, "w:gz", compresslevel=6) as tar:
        for path in sorted(members):
            tar.add(path, arcname=str(path.relative_to(index_dir)))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack or fetch the retrieval index bundle.")
    parser.add_argument("command", choices=("pack", "fetch"))
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output", type=Path, default=None, help="pack: archive path")
    parser.add_argument("--url", default=None, help="fetch: bundle URL")
    parser.add_argument("--sha256", default=None, help="fetch: expected checksum")
    parser.add_argument("--force", action="store_true", help="fetch: ignore an existing index")
    args = parser.parse_args(argv)

    try:
        if args.command == "pack":
            archive = pack(args.index_dir, args.output)
            size = archive.stat().st_size / (1 << 20)
            print(f"{archive}  {size:.1f} MB  sha256={sha256_of(archive)}")
        else:
            path = ensure_index(args.index_dir, url=args.url, sha256=args.sha256, force=args.force)
            print(f"Index ready at {path}")
    except IndexUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
