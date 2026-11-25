"""Configuration for the IDX bank report RAG pipeline.

Values are read from environment variables first, then from Streamlit secrets when
the app runs on Streamlit Community Cloud. Nothing secret is ever logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"

load_dotenv(PROJECT_ROOT / ".env", override=False)

BANKS: tuple[str, ...] = ("BBCA", "BBRI", "BMRI", "BBNI", "BRIS")

BANK_NAMES: dict[str, str] = {
    "BBCA": "PT Bank Central Asia Tbk",
    "BBRI": "PT Bank Rakyat Indonesia (Persero) Tbk",
    "BMRI": "PT Bank Mandiri (Persero) Tbk",
    "BBNI": "PT Bank Negara Indonesia (Persero) Tbk",
    "BRIS": "PT Bank Syariah Indonesia Tbk",
}

EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_DIM = 384
EMBED_MAX_TOKENS = 512

RETRIEVAL_METHODS: tuple[str, ...] = ("hybrid", "vector")


def _lookup(key: str) -> str | None:
    """Return a config value from the environment, then Streamlit secrets."""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st

        value = st.secrets.get(key)
    except Exception:
        return None
    if value in (None, ""):
        return None
    return str(value)


def _as_int(key: str, default: int) -> int:
    raw = _lookup(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _as_float(key: str, default: float) -> float:
    raw = _lookup(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings."""

    llm_model: str | None = None
    llm_api_key: str | None = None
    data_dir: Path = DEFAULT_DATA_DIR
    index_dir: Path = DEFAULT_INDEX_DIR
    embed_model: str = EMBED_MODEL
    embed_dim: int = EMBED_DIM
    chunk_min_tokens: int = 400
    chunk_max_tokens: int = 450
    chunk_overlap_ratio: float = 0.12
    min_indexable_tokens: int = 30
    top_k: int = 6
    rrf_k: int = 60
    retrieval_method: str = "hybrid"
    session_cooldown_seconds: float = 3.0
    daily_request_cap: int = 200
    cache_size: int = 128

    def __post_init__(self) -> None:
        if self.chunk_min_tokens < 1:
            raise ValueError("CHUNK_MIN_TOKENS must be positive")
        if self.chunk_max_tokens > EMBED_MAX_TOKENS:
            raise ValueError(f"CHUNK_MAX_TOKENS must be <= {EMBED_MAX_TOKENS} for {EMBED_MODEL}")
        if self.chunk_min_tokens > self.chunk_max_tokens:
            raise ValueError("CHUNK_MIN_TOKENS must be <= CHUNK_MAX_TOKENS")
        if not 0.0 <= self.chunk_overlap_ratio < 0.5:
            raise ValueError("CHUNK_OVERLAP_RATIO must be in [0, 0.5)")
        if self.top_k < 1:
            raise ValueError("TOP_K must be positive")
        if self.retrieval_method not in RETRIEVAL_METHODS:
            raise ValueError(
                f"RETRIEVAL_METHOD must be one of {RETRIEVAL_METHODS}, "
                f"got {self.retrieval_method!r}"
            )
        if self.daily_request_cap < 0:
            raise ValueError("DAILY_REQUEST_CAP must be >= 0")
        if self.min_indexable_tokens < 0:
            raise ValueError("MIN_INDEXABLE_TOKENS must be >= 0")

    @property
    def llm_configured(self) -> bool:
        """True when both the API key and the model ID are present."""
        return bool(self.llm_api_key and self.llm_model)

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the environment and Streamlit secrets."""
        return cls(
            llm_model=_lookup("LLM_MODEL"),
            llm_api_key=_lookup("LLM_API_KEY") or _lookup("GEMINI_API_KEY"),
            chunk_min_tokens=_as_int("CHUNK_MIN_TOKENS", 400),
            chunk_max_tokens=_as_int("CHUNK_MAX_TOKENS", 450),
            chunk_overlap_ratio=_as_float("CHUNK_OVERLAP_RATIO", 0.12),
            min_indexable_tokens=_as_int("MIN_INDEXABLE_TOKENS", 30),
            top_k=_as_int("TOP_K", 6),
            rrf_k=_as_int("RRF_K", 60),
            retrieval_method=_lookup("RETRIEVAL_METHOD") or "hybrid",
            session_cooldown_seconds=_as_float("SESSION_COOLDOWN_SECONDS", 3.0),
            daily_request_cap=_as_int("DAILY_REQUEST_CAP", 200),
            cache_size=_as_int("CACHE_SIZE", 128),
        )
