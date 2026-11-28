"""Tests for the gold set, the retrieval metrics, and the answer scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config import PROJECT_ROOT, Settings
from rag.evaluate import (
    GoldQuestion,
    answerable_pages,
    display_path,
    evaluate_answers,
    evaluate_retrieval,
    hit_at_k,
    load_gold,
    ranked_pages,
    reciprocal_rank,
    rescore_answers,
    verify_gold,
    write_report,
)
from rag.generator import LLMUnavailable
from rag.schemas import Chunk, RetrievedChunk

GOLD_PATH = PROJECT_ROOT / "eval" / "gold_questions.jsonl"
INDEX_CHUNKS = PROJECT_ROOT / "index" / "chunks.jsonl"

needs_index = pytest.mark.skipif(
    not INDEX_CHUNKS.exists(), reason="the derived index is not present"
)


class StubRetriever:
    """Returns a fixed ranking per query, so metrics are checked without an index."""

    def __init__(self, ranking: list[RetrievedChunk], settings: Settings | None = None) -> None:
        self.ranking = ranking
        self.settings = settings or Settings()

    def search(self, query: str, banks=None, top_k=None, method=None):
        return list(self.ranking)


class ScriptedAnswerer:
    """Replays prepared answers, in order, without touching the network."""

    def __init__(self, answers: list[dict]) -> None:
        self.answers = list(answers)

    def answer(self, question: str, banks=None) -> dict:
        return self.answers.pop(0)


def result(doc_id: str, page: int) -> RetrievedChunk:
    bank, year = doc_id.split("_")
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{doc_id}-p{page:04d}-c1",
            doc_id=doc_id,
            bank=bank,
            year=int(year),
            page=page,
            text="teks",
            n_tokens=1,
        ),
        score=0.02,
    )


def answered(doc: str, page: int, refused: bool = False) -> dict:
    if refused:
        return {"answer": "tidak ada", "sources": [], "refused": True}
    return {
        "answer": "jawaban",
        "sources": [{"doc": doc, "page": page, "snippet": "s"}],
        "refused": False,
    }


def test_gold_file_loads_with_unique_ids() -> None:
    questions = load_gold(GOLD_PATH)

    assert len(questions) >= 30
    assert len({question.id for question in questions}) == len(questions)


def test_gold_file_covers_every_document_and_both_kinds_of_question() -> None:
    questions = load_gold(GOLD_PATH)
    documents = {question.doc_id for question in questions if not question.out_of_scope}

    assert documents == {
        f"{bank}_{year}"
        for bank in ("BBCA", "BBRI", "BMRI", "BBNI", "BRIS")
        for year in (2023, 2024)
    }
    assert any(question.out_of_scope for question in questions)


def test_in_scope_questions_must_declare_a_page_and_an_expected_value() -> None:
    with pytest.raises(ValueError, match="needs doc_id, pages, and must_contain"):
        GoldQuestion.from_dict({"id": "x", "question": "Berapa laba?", "doc_id": "BBCA_2024"})


def test_gold_question_rejects_an_unknown_bank() -> None:
    with pytest.raises(ValueError, match="unknown bank"):
        GoldQuestion.from_dict(
            {
                "id": "x",
                "question": "Berapa laba?",
                "banks": ["XXXX"],
                "doc_id": "BBCA_2024",
                "pages": [1],
                "must_contain": ["1"],
            }
        )


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    row = {
        "id": "dupe",
        "question": "Berapa laba?",
        "doc_id": "BBCA_2024",
        "pages": [1],
        "must_contain": ["1"],
    }
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_gold(path)


@needs_index
def test_every_expected_value_really_appears_on_its_page() -> None:
    assert verify_gold(load_gold(GOLD_PATH), PROJECT_ROOT / "index") == []


def test_ranked_pages_collapses_repeated_pages_and_keeps_order() -> None:
    ranked = ranked_pages([result("BBCA_2024", 5), result("BBCA_2024", 5), result("BBCA_2023", 9)])

    assert ranked == [("BBCA_2024", 5), ("BBCA_2023", 9)]


def test_hit_at_k_respects_the_cutoff() -> None:
    ranked = [("BBCA_2024", 1), ("BBCA_2024", 2), ("BBCA_2024", 3)]
    gold = {("BBCA_2024", 3)}

    assert hit_at_k(ranked, gold, 3) is True
    assert hit_at_k(ranked, gold, 2) is False


def test_reciprocal_rank_rewards_higher_positions() -> None:
    assert reciprocal_rank([("BBCA_2024", 1)], {("BBCA_2024", 1)}) == pytest.approx(1.0)
    assert reciprocal_rank([("BBCA_2024", 1), ("BBCA_2024", 2)], {("BBCA_2024", 2)}) == (
        pytest.approx(0.5)
    )
    assert reciprocal_rank([("BBCA_2024", 1)], {("BBCA_2024", 9)}) == 0.0


def test_retrieval_report_scores_the_strict_and_the_answerable_page() -> None:
    questions = [
        GoldQuestion(
            id="q1",
            question="Berapa laba?",
            banks=["BBCA"],
            doc_id="BBCA_2024",
            pages=(10,),
            must_contain=("99",),
        )
    ]
    retriever = StubRetriever([result("BBCA_2024", 12), result("BBCA_2024", 10)])
    answerable = {"q1": {("BBCA_2024", 12), ("BBCA_2024", 10)}}

    report = evaluate_retrieval(retriever, questions, ["hybrid"], 6, answerable)["methods"][
        "hybrid"
    ]

    assert report["hit_rate"] == 1.0
    assert report["mrr"] == pytest.approx(0.5)
    assert report["answerable_hit_rate"] == 1.0
    assert report["answerable_mrr"] == pytest.approx(1.0)
    assert report["missed"] == []


def test_retrieval_report_lists_misses() -> None:
    questions = [
        GoldQuestion(
            id="q1",
            question="Berapa laba?",
            banks=["BBCA"],
            doc_id="BBCA_2024",
            pages=(10,),
            must_contain=("99",),
        )
    ]
    report = evaluate_retrieval(StubRetriever([result("BBCA_2024", 3)]), questions, ["bm25"], 6)

    assert report["methods"]["bm25"]["hit_rate"] == 0.0
    assert report["methods"]["bm25"]["missed"] == ["q1"]


def test_out_of_scope_questions_are_left_out_of_retrieval_scores() -> None:
    questions = [GoldQuestion(id="oos", question="Cuaca?", out_of_scope=True)]
    report = evaluate_retrieval(StubRetriever([]), questions, ["hybrid"], 6)

    assert report["questions"] == 0
    assert report["methods"]["hybrid"]["hit_rate"] == 0.0


def test_answerable_pages_scan_the_whole_document() -> None:
    questions = [
        GoldQuestion(
            id="q1",
            question="Berapa laba?",
            doc_id="BBCA_2024",
            pages=(10,),
            must_contain=("99",),
        )
    ]
    page_text = {
        ("BBCA_2024", 10): "laba 99",
        ("BBCA_2024", 11): "laba 99 juga",
        ("BBCA_2024", 12): "tidak menyebut",
        ("BBRI_2024", 10): "laba 99 milik bank lain",
    }

    assert answerable_pages(questions, page_text) == {"q1": {("BBCA_2024", 10), ("BBCA_2024", 11)}}


def test_answer_scoring_counts_an_answer_that_cites_the_gold_page() -> None:
    questions = [
        GoldQuestion(
            id="q1",
            question="Berapa laba?",
            doc_id="BBCA_2024",
            pages=(10,),
            must_contain=("99",),
        ),
        GoldQuestion(id="oos", question="Cuaca?", out_of_scope=True),
    ]
    report = evaluate_answers(
        ScriptedAnswerer([answered("BBCA_2024", 10), answered("", 0, True)]), questions
    )

    assert report["in_scope"]["answer_rate"] == 1.0
    assert report["in_scope"]["citation_hit_rate"] == 1.0
    assert report["out_of_scope"]["refusal_rate"] == 1.0


def test_answer_scoring_flags_a_citation_that_misses_the_gold_page() -> None:
    questions = [
        GoldQuestion(
            id="q1",
            question="Berapa laba?",
            doc_id="BBCA_2024",
            pages=(10,),
            must_contain=("99",),
        )
    ]
    report = evaluate_answers(ScriptedAnswerer([answered("BBCA_2024", 999)]), questions)

    assert report["in_scope"]["citation_hit_rate"] == 0.0
    assert report["in_scope"]["failures"] == ["q1"]


def test_answer_scoring_flags_a_question_the_model_refused() -> None:
    questions = [
        GoldQuestion(
            id="q1",
            question="Berapa laba?",
            doc_id="BBCA_2024",
            pages=(10,),
            must_contain=("99",),
        )
    ]
    report = evaluate_answers(ScriptedAnswerer([answered("", 0, True)]), questions)

    assert report["in_scope"]["refused"] == 1
    assert report["in_scope"]["answer_rate"] == 0.0
    assert report["in_scope"]["citation_hit_rate"] == 0.0


def test_answer_scoring_flags_an_out_of_scope_question_that_was_answered() -> None:
    questions = [GoldQuestion(id="oos", question="Cuaca?", out_of_scope=True)]
    report = evaluate_answers(ScriptedAnswerer([answered("BBCA_2024", 10)]), questions)

    assert report["out_of_scope"]["refusal_rate"] == 0.0
    assert report["out_of_scope"]["wrongly_answered"] == ["oos"]


class RateLimitedAnswerer:
    """Raises a quota error the given number of times, then answers."""

    def __init__(self, failures: int, then: dict) -> None:
        self.failures = failures
        self.then = then
        self.calls = 0

    def answer(self, question: str, banks=None) -> dict:
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise LLMUnavailable("model call failed with status RESOURCE_EXHAUSTED")
        return self.then


def a_question() -> GoldQuestion:
    return GoldQuestion(
        id="q1",
        question="Berapa laba?",
        doc_id="BBCA_2024",
        pages=(10,),
        must_contain=("99",),
    )


def test_a_rate_limited_question_is_retried_after_a_pause() -> None:
    answerer = RateLimitedAnswerer(1, answered("BBCA_2024", 10))
    waits: list[float] = []

    report = evaluate_answers(answerer, [a_question()], 0.0, waits.append)

    assert waits == [20.0]
    assert report["in_scope"]["answer_rate"] == 1.0
    assert report["in_scope"]["errors"] == 0


def test_a_persistent_rate_limit_is_counted_as_an_error_not_a_refusal() -> None:
    answerer = RateLimitedAnswerer(2, answered("BBCA_2024", 10))
    report = evaluate_answers(answerer, [a_question()], 0.0, lambda _: None)

    assert report["in_scope"]["errors"] == 1
    assert report["in_scope"]["refused"] == 0
    assert report["in_scope"]["answer_rate"] == 0.0


def test_questions_are_paced_to_stay_under_the_per_minute_limit() -> None:
    waits: list[float] = []
    questions = [
        GoldQuestion(
            id=f"q{index}",
            question="Berapa laba?",
            doc_id="BBCA_2024",
            pages=(10,),
            must_contain=("99",),
        )
        for index in range(3)
    ]
    answerer = ScriptedAnswerer([answered("BBCA_2024", 10)] * 3)

    evaluate_answers(answerer, questions, pace_seconds=4.5, sleep=waits.append)

    assert waits == [4.5, 4.5]


def write_fake_index(directory: Path, rows: list[dict]) -> Path:
    index_dir = directory / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row) for row in rows]
    (index_dir / "chunks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_dir


def chunk_row(doc_id: str, page: int, text: str) -> dict:
    bank, year = doc_id.split("_")
    return {
        "chunk_id": f"{doc_id}-p{page:04d}-c1",
        "doc_id": doc_id,
        "bank": bank,
        "year": int(year),
        "page": page,
        "text": text,
        "n_tokens": len(text.split()),
    }


def test_verify_gold_accepts_a_consistent_set(tmp_path: Path) -> None:
    index_dir = write_fake_index(tmp_path, [chunk_row("BBCA_2024", 10, "laba 99 triliun")])
    gold = [
        GoldQuestion(
            id="q1",
            question="Berapa laba?",
            doc_id="BBCA_2024",
            pages=(10,),
            must_contain=("99",),
        )
    ]

    assert verify_gold(gold, index_dir) == []


def test_verify_gold_flags_an_expected_value_that_is_not_on_the_page(tmp_path: Path) -> None:
    index_dir = write_fake_index(tmp_path, [chunk_row("BBCA_2024", 10, "laba 12 triliun")])
    gold = [
        GoldQuestion(
            id="q1",
            question="Berapa laba?",
            doc_id="BBCA_2024",
            pages=(10,),
            must_contain=("99",),
        )
    ]

    problems = verify_gold(gold, index_dir)

    assert len(problems) == 1
    assert "'99'" in problems[0] or "99" in problems[0]


def test_verify_gold_flags_a_page_that_is_missing_from_the_index(tmp_path: Path) -> None:
    index_dir = write_fake_index(tmp_path, [chunk_row("BBCA_2024", 10, "laba 99")])
    gold = [
        GoldQuestion(
            id="q1",
            question="Berapa laba?",
            doc_id="BBCA_2024",
            pages=(99,),
            must_contain=("99",),
        )
    ]

    assert "is not in the index" in verify_gold(gold, index_dir)[0]


def test_verify_gold_rejects_an_out_of_scope_question_the_documents_actually_answer(
    tmp_path: Path,
) -> None:
    index_dir = write_fake_index(tmp_path, [chunk_row("BBCA_2024", 10, "kami memakai blockchain")])
    gold = [
        GoldQuestion(
            id="oos",
            question="Apa itu blockchain?",
            out_of_scope=True,
            absent_terms=("blockchain",),
        )
    ]

    problems = verify_gold(gold, index_dir)

    assert len(problems) == 1
    assert "mention" in problems[0]


def test_absent_terms_are_optional_for_out_of_scope_questions(tmp_path: Path) -> None:
    index_dir = write_fake_index(tmp_path, [chunk_row("BBCA_2024", 10, "laba 99")])
    gold = [GoldQuestion(id="oos", question="Cuaca?", out_of_scope=True)]

    assert verify_gold(gold, index_dir) == []


def test_answer_scoring_counts_a_citation_of_any_page_holding_the_value() -> None:
    questions = [a_question()]
    report = evaluate_answers(
        ScriptedAnswerer([answered("BBCA_2024", 11)]),
        questions,
        answerable={"q1": {("BBCA_2024", 11), ("BBCA_2024", 12)}},
    )

    assert report["in_scope"]["citation_hit_rate"] == 0.0
    assert report["in_scope"]["citation_answerable_rate"] == 1.0


def test_answer_scoring_records_what_each_answer_cited() -> None:
    questions = [a_question()]
    report = evaluate_answers(ScriptedAnswerer([answered("BBCA_2024", 10)]), questions)

    assert report["details"] == [{"id": "q1", "refused": False, "cited": [("BBCA_2024", 10)]}]


def test_rescoring_recomputes_metrics_from_a_stored_report() -> None:
    questions = [a_question(), GoldQuestion(id="oos", question="Cuaca?", out_of_scope=True)]
    stored = {
        "in_scope": {"errors": 0},
        "out_of_scope": {"errors": 0},
        "details": [
            {"id": "q1", "refused": False, "cited": [["BBCA_2024", 11]]},
            {"id": "oos", "refused": True, "cited": []},
        ],
    }

    rescored = rescore_answers(stored, questions, {"q1": {("BBCA_2024", 11)}})

    assert rescored["in_scope"]["answered"] == 1
    assert rescored["in_scope"]["citation_hit_rate"] == 0.0
    assert rescored["in_scope"]["citation_answerable_rate"] == 1.0
    assert rescored["in_scope"]["failures"] == []
    assert rescored["out_of_scope"]["refusal_rate"] == 1.0


def test_rescoring_keeps_a_refusal_out_of_the_citation_rate() -> None:
    stored = {
        "in_scope": {"errors": 0},
        "out_of_scope": {"errors": 0},
        "details": [{"id": "q1", "refused": True, "cited": []}],
    }

    rescored = rescore_answers(stored, [a_question()])

    assert rescored["in_scope"]["answered"] == 0
    assert rescored["in_scope"]["citation_answerable_rate"] == 0.0
    assert rescored["in_scope"]["refused_ids"] == ["q1"]


def test_report_is_written_as_readable_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "results.json"
    write_report(path, {"questions": 3})

    assert json.loads(path.read_text(encoding="utf-8")) == {"questions": 3}


def test_report_paths_are_relative_to_the_project() -> None:
    assert display_path(GOLD_PATH) == "eval/gold_questions.jsonl"
    assert display_path(Path("/tmp/elsewhere/questions.jsonl")) == "questions.jsonl"


def test_the_committed_report_contains_no_local_paths() -> None:
    report = PROJECT_ROOT / "eval" / "results" / "latest.json"
    if not report.exists():
        pytest.skip("no measured report has been committed")

    text = report.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "C:\\Users" not in text
