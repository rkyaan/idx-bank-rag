"""Embedding wrapper around multilingual-e5-small.

The model ships as ONNX weights and runs through onnxruntime, so no PyTorch
install is needed. E5 models require an asymmetric prefix: ``passage:`` for
documents and ``query:`` for questions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from rag.config import EMBED_DIM, EMBED_MODEL

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

_registered: set[str] = set()


def _is_builtin(model_name: str) -> bool:
    return any(entry["model"] == model_name for entry in TextEmbedding.list_supported_models())


def _register(model_name: str, dim: int) -> None:
    """Teach fastembed about e5-small, which is not in its built-in catalog."""
    if model_name in _registered or _is_builtin(model_name):
        _registered.add(model_name)
        return
    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=model_name),
        dim=dim,
        model_file="onnx/model.onnx",
        description="Multilingual E5 small text embeddings (ONNX)",
        license="MIT",
        size_in_gb=0.45,
    )
    _registered.add(model_name)


class Embedder:
    """Turns text into normalized 384-dimensional vectors."""

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        *,
        dim: int = EMBED_DIM,
        cache_dir: str | Path | None = None,
        threads: int | None = None,
    ) -> None:
        _register(model_name, dim)
        self.model_name = model_name
        self.dim = dim
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
            threads=threads,
        )

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._embed(texts, PASSAGE_PREFIX, batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], QUERY_PREFIX, 1)[0]

    def _embed(self, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prepared = [prefix + text for text in texts]
        vectors = np.asarray(
            list(self._model.embed(prepared, batch_size=batch_size)), dtype=np.float32
        )
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        return vectors
