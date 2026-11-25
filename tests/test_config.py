"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from rag.config import BANKS, Settings

ENV_KEYS = (
    "LLM_MODEL",
    "LLM_API_KEY",
    "GEMINI_API_KEY",
    "CHUNK_MIN_TOKENS",
    "CHUNK_MAX_TOKENS",
    "CHUNK_OVERLAP_RATIO",
    "MIN_INDEXABLE_TOKENS",
    "TOP_K",
    "RRF_K",
    "RETRIEVAL_METHOD",
    "SESSION_COOLDOWN_SECONDS",
    "DAILY_REQUEST_CAP",
    "CACHE_SIZE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_bank_list_is_the_five_idx_banks() -> None:
    assert BANKS == ("BBCA", "BBRI", "BMRI", "BBNI", "BRIS")


def test_defaults_match_the_approved_plan() -> None:
    settings = Settings.from_env()
    assert settings.chunk_min_tokens == 400
    assert settings.chunk_max_tokens == 450
    assert settings.chunk_overlap_ratio == pytest.approx(0.12)
    assert settings.min_indexable_tokens == 30
    assert settings.top_k == 6
    assert settings.retrieval_method == "hybrid"
    assert settings.session_cooldown_seconds == pytest.approx(3.0)
    assert settings.llm_model is None
    assert settings.llm_configured is False


def test_environment_overrides_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("RETRIEVAL_METHOD", "vector")
    monkeypatch.setenv("CHUNK_OVERLAP_RATIO", "0.15")

    settings = Settings.from_env()

    assert settings.llm_model == "some-model"
    assert settings.llm_api_key == "test-key"
    assert settings.llm_configured is True
    assert settings.top_k == 9
    assert settings.retrieval_method == "vector"
    assert settings.chunk_overlap_ratio == pytest.approx(0.15)


def test_gemini_api_key_is_accepted_as_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "alias-key")
    assert Settings.from_env().llm_api_key == "alias-key"


def test_non_numeric_integer_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "six")
    with pytest.raises(ValueError, match="TOP_K must be an integer"):
        Settings.from_env()


def test_chunk_size_above_embedding_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="CHUNK_MAX_TOKENS"):
        Settings(chunk_max_tokens=600)


def test_inverted_chunk_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="CHUNK_MIN_TOKENS must be <="):
        Settings(chunk_min_tokens=450, chunk_max_tokens=400)


def test_overlap_must_stay_below_half() -> None:
    with pytest.raises(ValueError, match="CHUNK_OVERLAP_RATIO"):
        Settings(chunk_overlap_ratio=0.6)


def test_unknown_retrieval_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="RETRIEVAL_METHOD"):
        Settings(retrieval_method="bm25-only")


def test_negative_daily_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="DAILY_REQUEST_CAP"):
        Settings(daily_request_cap=-1)
