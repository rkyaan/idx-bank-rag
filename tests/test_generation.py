"""Tests for prompting, citation validation, refusals, caching, and the quota cap."""

from __future__ import annotations

from datetime import date

import pytest

from rag.config import Settings
from rag.generator import (
    REFUSAL_TEXT,
    DailyCapReached,
    Generator,
    LLMUnavailable,
    with_retry,
)
from rag.pipeline import Answerer, build_prompt, is_refusal, validate_answer
from rag.schemas import Chunk, RetrievedChunk


class FakeModel:
    """Records every prompt and replays a prepared answer."""

    def __init__(self, reply: str = "Laba bersih naik [BBCA_2024, hlm. 112].") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class FailingModel:
    def generate(self, prompt: str) -> str:
        raise LLMUnavailable("quota exhausted")


class StubRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, list[str] | None]] = []

    def search(self, query: str, banks: list[str] | None = None, **kwargs: object):
        self.calls.append((query, banks))
        return list(self.results)


def make_result(
    doc_id: str = "BBCA_2024",
    page: int = 112,
    text: str = "Laba bersih tumbuh 12,7 persen menjadi Rp54,8 triliun.",
    score: float = 0.03,
) -> RetrievedChunk:
    bank, year = doc_id.split("_")
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{doc_id}-p{page:04d}-c1",
            doc_id=doc_id,
            bank=bank,
            year=int(year),
            page=page,
            text=text,
            n_tokens=len(text.split()),
        ),
        score=score,
    )


def make_answerer(
    reply: str = "Laba bersih naik [BBCA_2024, hlm. 112].",
    results: list[RetrievedChunk] | None = None,
    settings: Settings | None = None,
    today: date | None = None,
) -> tuple[Answerer, FakeModel, StubRetriever]:
    settings = settings or Settings()
    model = FakeModel(reply)
    retriever = StubRetriever(results if results is not None else [make_result()])
    answerer = Answerer(
        retriever, Generator(model, settings, today=lambda: today or date(2026, 1, 1)), settings
    )
    return answerer, model, retriever


def test_grounded_answer_keeps_its_citation() -> None:
    answerer, _, _ = make_answerer()
    result = answerer.answer("Berapa laba bersih 2024?")

    assert result["refused"] is False
    assert "[BBCA_2024, hlm. 112]" in result["answer"]
    assert result["sources"] == [
        {
            "doc": "BBCA_2024",
            "page": 112,
            "snippet": "Laba bersih tumbuh 12,7 persen menjadi Rp54,8 triliun.",
        }
    ]


def test_invented_citation_is_removed_but_real_one_survives() -> None:
    answerer, _, _ = make_answerer(
        reply="Laba naik [BBCA_2024, hlm. 112] dan beban turun [BBNI_2019, hlm. 999]."
    )
    result = answerer.answer("Bagaimana laba?")

    assert "[BBCA_2024, hlm. 112]" in result["answer"]
    assert "BBNI_2019" not in result["answer"]
    assert "999" not in result["answer"]
    assert [source["doc"] for source in result["sources"]] == ["BBCA_2024"]


def test_answer_without_any_valid_citation_becomes_a_refusal() -> None:
    answerer, _, _ = make_answerer(reply="Laba bersih naik 12,7 persen.")
    result = answerer.answer("Berapa laba bersih?")

    assert result["refused"] is True
    assert result["answer"] == REFUSAL_TEXT
    assert result["sources"] == []


def test_model_refusal_is_passed_through_without_sources() -> None:
    answerer, _, _ = make_answerer(reply=REFUSAL_TEXT)
    result = answerer.answer("Siapa direktur utama tahun 2030?")

    assert result["refused"] is True
    assert result["answer"] == REFUSAL_TEXT
    assert result["sources"] == []


def test_near_miss_refusal_wording_is_still_a_refusal() -> None:
    answerer, _, _ = make_answerer(reply="Informasi itu tidak ada di dokumen yang tersedia.")
    result = answerer.answer("Berapa harga saham besok?")

    assert result["refused"] is True
    assert result["answer"] == REFUSAL_TEXT


def test_empty_question_is_refused_without_calling_the_model() -> None:
    answerer, model, retriever = make_answerer()
    result = answerer.answer("   ")

    assert result["refused"] is True
    assert model.prompts == []
    assert retriever.calls == []


def test_bank_filter_reaches_the_retriever() -> None:
    answerer, _, retriever = make_answerer()
    answerer.answer("Bagaimana NPL?", banks=["BBRI", "BMRI"])

    assert retriever.calls == [("Bagaimana NPL?", ["BBRI", "BMRI"])]


def test_no_retrieved_passages_refuses_before_spending_a_request() -> None:
    answerer, model, _ = make_answerer(results=[])
    result = answerer.answer("Apa itu bank?")

    assert result["refused"] is True
    assert model.prompts == []


def test_relevance_floor_refuses_weak_matches() -> None:
    settings = Settings(min_relevance_score=0.02)
    answerer, model, _ = make_answerer(results=[make_result(score=0.001)], settings=settings)
    result = answerer.answer("Pertanyaan tidak relevan?")

    assert result["refused"] is True
    assert model.prompts == []


def test_prompt_carries_the_citation_label_and_the_question() -> None:
    results = [make_result(), make_result(page=113, text="Beban pajak turun.")]
    prompt = build_prompt("Berapa laba bersih?", results, 14000)

    assert "[BBCA_2024, hlm. 112]" in prompt
    assert "[BBCA_2024, hlm. 113]" in prompt
    assert prompt.rstrip().endswith("PERTANYAAN: Berapa laba bersih?\n\nJAWABAN:")


def test_prompt_stops_adding_passages_past_the_context_budget() -> None:
    results = [make_result(page=page, text="x" * 400) for page in range(1, 11)]
    prompt = build_prompt("Pertanyaan", results, 1200)

    assert "[BBCA_2024, hlm. 1]" in prompt
    assert "[BBCA_2024, hlm. 10]" not in prompt


def test_answer_is_trimmed_of_the_leading_label() -> None:
    answerer, _, _ = make_answerer(reply="Jawaban: Laba naik [BBCA_2024, hlm. 112].")
    result = answerer.answer("Bagaimana laba?")

    assert result["answer"].startswith("Laba naik")


def test_llm_failure_is_raised_for_the_ui_to_translate() -> None:
    settings = Settings()
    retriever = StubRetriever([make_result()])
    answerer = Answerer(retriever, Generator(FailingModel(), settings), settings)

    with pytest.raises(LLMUnavailable):
        answerer.answer("Berapa laba bersih?")


def test_cache_returns_the_same_answer_without_a_second_model_call() -> None:
    answerer, model, _ = make_answerer()
    first = answerer.answer("Berapa laba bersih?")
    second = answerer.answer("Berapa laba bersih?")

    assert first == second
    assert len(model.prompts) == 1


def test_distinct_questions_each_call_the_model_once() -> None:
    answerer, model, _ = make_answerer()
    answerer.answer("Berapa laba bersih?")
    answerer.answer("Bagaimana rasio NPL?")

    assert len(model.prompts) == 2


def test_daily_cap_stops_the_extra_request() -> None:
    settings = Settings(daily_request_cap=2)
    model = FakeModel()
    generator = Generator(model, settings, today=lambda: date(2026, 1, 1))

    generator.generate("pertanyaan satu")
    generator.generate("pertanyaan dua")
    with pytest.raises(DailyCapReached):
        generator.generate("pertanyaan tiga")

    assert len(model.prompts) == 2
    assert generator.requests_used_today == 2


def test_daily_cap_resets_on_a_new_day() -> None:
    settings = Settings(daily_request_cap=1)
    today = date(2026, 1, 1)
    generator = Generator(FakeModel(), settings, today=lambda: today)

    generator.generate("pertanyaan satu")
    with pytest.raises(DailyCapReached):
        generator.generate("pertanyaan dua")

    today = date(2026, 1, 2)
    assert generator.generate("pertanyaan dua")


def test_cap_of_zero_means_unlimited() -> None:
    settings = Settings(daily_request_cap=0)
    generator = Generator(FakeModel(), settings, today=lambda: date(2026, 1, 1))

    for index in range(5):
        generator.generate(f"pertanyaan {index}")

    assert generator.requests_used_today == 0


def test_cache_stays_within_its_size_limit() -> None:
    settings = Settings(cache_size=2)
    generator = Generator(FakeModel(), settings, today=lambda: date(2026, 1, 1))

    for index in range(5):
        generator.generate(f"pertanyaan {index}")

    assert generator.cached_answers == 2


def test_is_refusal_ignores_case_and_extra_space() -> None:
    assert is_refusal("  INFORMASI   itu tidak ada")
    assert not is_refusal("Laba bersih naik 12,7 persen.")


def test_validate_answer_keeps_page_order_of_first_mention() -> None:
    results = [make_result(page=112), make_result(page=113)]
    answer, cited = validate_answer(
        "Naik [BBCA_2024, hlm. 113], lalu turun [BBCA_2024, hlm. 112].", results
    )

    assert [result.chunk.page for result in cited] == [113, 112]
    assert answer.count("[BBCA_2024") == 2


def test_retry_recovers_from_a_transient_failure() -> None:
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise LLMUnavailable("model call failed with status DEADLINE_EXCEEDED", retryable=True)
        return "jawaban"

    assert with_retry(flaky, attempts=2, backoff_seconds=0, sleep=lambda _: None) == "jawaban"
    assert len(calls) == 2


def test_retry_gives_up_and_raises_the_last_error() -> None:
    def always_fails() -> str:
        raise LLMUnavailable("model call failed with status UNAVAILABLE", retryable=True)

    with pytest.raises(LLMUnavailable, match="UNAVAILABLE"):
        with_retry(always_fails, attempts=3, backoff_seconds=0, sleep=lambda _: None)


def test_retry_does_not_repeat_a_permanent_failure() -> None:
    calls: list[int] = []

    def quota_exhausted() -> str:
        calls.append(1)
        raise LLMUnavailable("model call failed with status RESOURCE_EXHAUSTED")

    with pytest.raises(LLMUnavailable):
        with_retry(quota_exhausted, attempts=3, backoff_seconds=0, sleep=lambda _: None)
    assert len(calls) == 1


def test_retry_backs_off_between_attempts() -> None:
    waits: list[float] = []

    def always_fails() -> str:
        raise LLMUnavailable("busy", retryable=True)

    with pytest.raises(LLMUnavailable):
        with_retry(always_fails, attempts=3, backoff_seconds=0.5, sleep=waits.append)

    assert waits == [0.5, 1.0]


def test_missing_llm_configuration_is_reported_before_any_call() -> None:
    settings = Settings(llm_model=None, llm_api_key=None)
    assert settings.llm_configured is False
