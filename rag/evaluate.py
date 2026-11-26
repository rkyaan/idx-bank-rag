"""Measurement against a hand-checked gold question set.

Retrieval is scored two ways. The strict measure asks whether the specific page we
picked by hand shows up in the top k. The answerable measure is wider: any page that
actually contains the figure a correct answer must cite counts, because a bank states
the same number in its highlights, its management discussion, and its statements.
That second measure needs no model call, so it can run as often as we like. Answer
quality is scored separately, because refusals and citations spend free-tier quota.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.config import BANKS, PROJECT_ROOT, RETRIEVAL_METHODS, Settings
from rag.generator import LLMUnavailable
from rag.pipeline import Answerer, get_answerer
from rag.retriever import Retriever
from rag.schemas import RetrievedChunk

GOLD_PATH = PROJECT_ROOT / "eval" / "gold_questions.jsonl"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "eval" / "results" / "latest.json"

PageKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class GoldQuestion:
    """One judged question, with the pages a correct answer must cite."""

    id: str
    question: str
    banks: list[str] | None = None
    doc_id: str | None = None
    pages: tuple[int, ...] = ()
    must_contain: tuple[str, ...] = ()
    absent_terms: tuple[str, ...] = ()
    topic: str | None = None
    out_of_scope: bool = False

    @property
    def page_keys(self) -> set[PageKey]:
        if self.doc_id is None:
            return set()
        return {(self.doc_id, page) for page in self.pages}

    @classmethod
    def from_dict(cls, row: dict) -> GoldQuestion:
        out_of_scope = bool(row.get("out_of_scope", False))
        question = cls(
            id=row["id"],
            question=row["question"],
            banks=row.get("banks"),
            doc_id=row.get("doc_id"),
            pages=tuple(int(page) for page in row.get("pages", [])),
            must_contain=tuple(row.get("must_contain", [])),
            absent_terms=tuple(row.get("absent_terms", [])),
            topic=row.get("topic"),
            out_of_scope=out_of_scope,
        )
        if out_of_scope:
            return question
        if not question.doc_id or not question.pages or not question.must_contain:
            raise ValueError(f"gold question {question.id!r} needs doc_id, pages, and must_contain")
        unknown = [bank for bank in question.banks or [] if bank not in BANKS]
        if unknown:
            raise ValueError(f"gold question {question.id!r} names unknown bank(s) {unknown}")
        return question


def load_gold(path: Path = GOLD_PATH) -> list[GoldQuestion]:
    """Read the gold set, rejecting duplicate ids."""
    questions = [
        GoldQuestion.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(f"duplicate gold question id: {question.id}")
        seen.add(question.id)
    return questions


def ranked_pages(results: Sequence[RetrievedChunk]) -> list[PageKey]:
    """Distinct (document, page) pairs in rank order, so two chunks of one page count once."""
    ranked: list[PageKey] = []
    seen: set[PageKey] = set()
    for result in results:
        key = (result.chunk.doc_id, result.chunk.page)
        if key not in seen:
            seen.add(key)
            ranked.append(key)
    return ranked


def hit_at_k(ranked: Sequence[PageKey], gold: set[PageKey], k: int) -> bool:
    return any(key in gold for key in ranked[:k])


def recall_at_k(ranked: Sequence[PageKey], gold: set[PageKey], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold.intersection(ranked[:k])) / len(gold)


def reciprocal_rank(ranked: Sequence[PageKey], gold: set[PageKey]) -> float:
    for rank, key in enumerate(ranked, start=1):
        if key in gold:
            return 1.0 / rank
    return 0.0


def index_page_text(index_dir: Path) -> dict[PageKey, str]:
    """Concatenate every chunk of a page, so a value can be found wherever it sits."""
    chunks_path = index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No index found at {chunks_path}")

    page_text: dict[PageKey, str] = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["doc_id"], int(row["page"]))
        page_text[key] = page_text.get(key, "") + " " + row["text"]
    return page_text


def answerable_pages(
    questions: Sequence[GoldQuestion], page_text: dict[PageKey, str]
) -> dict[str, set[PageKey]]:
    """Every page that really holds an expected value, not just the one we picked by hand."""
    answerable: dict[str, set[PageKey]] = {}
    for question in questions:
        if question.out_of_scope:
            continue
        answerable[question.id] = {
            key
            for key, text in page_text.items()
            if key[0] == question.doc_id and any(value in text for value in question.must_contain)
        }
    return answerable


def evaluate_retrieval(
    retriever: Retriever,
    questions: Sequence[GoldQuestion],
    methods: Sequence[str] = RETRIEVAL_METHODS,
    top_k: int | None = None,
    answerable: dict[str, set[PageKey]] | None = None,
) -> dict:
    """Score each retrieval method on the strict gold, and on every answerable page."""
    in_scope = [question for question in questions if not question.out_of_scope]
    k = top_k or retriever.settings.top_k
    report: dict = {"top_k": k, "questions": len(in_scope), "methods": {}}

    for method in methods:
        hits = 0
        recalls = 0.0
        rr_total = 0.0
        wide_hits = 0
        wide_rr = 0.0
        misses: list[str] = []
        for question in in_scope:
            gold = question.page_keys
            ranked = ranked_pages(retriever.search(question.question, question.banks, k, method))
            if hit_at_k(ranked, gold, k):
                hits += 1
            else:
                misses.append(question.id)
            recalls += recall_at_k(ranked, gold, k)
            rr_total += reciprocal_rank(ranked, gold)

            wide = (answerable or {}).get(question.id) or gold
            if hit_at_k(ranked, wide, k):
                wide_hits += 1
            wide_rr += reciprocal_rank(ranked, wide)

        total = len(in_scope)
        report["methods"][method] = {
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "recall": round(recalls / total, 4) if total else 0.0,
            "mrr": round(rr_total / total, 4) if total else 0.0,
            "answerable_hit_rate": round(wide_hits / total, 4) if total else 0.0,
            "answerable_mrr": round(wide_rr / total, 4) if total else 0.0,
            "missed": misses,
        }
    return report


def evaluate_answers(
    answerer: Answerer,
    questions: Sequence[GoldQuestion],
    pace_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    retry_wait_seconds: float = 20.0,
    answerable: dict[str, set[PageKey]] | None = None,
) -> dict:
    """Score refusal behaviour and citation accuracy; this spends model quota.

    The free tier allows only a handful of requests per minute, so a bulk run paces
    itself and gives a rate-limited question one more chance after a pause.
    """
    in_scope = [question for question in questions if not question.out_of_scope]
    out_of_scope = [question for question in questions if question.out_of_scope]

    asked = 0

    def ask(question: GoldQuestion) -> dict | None:
        nonlocal asked
        if asked and pace_seconds:
            sleep(pace_seconds)
        asked += 1
        try:
            return answerer.answer(question.question, question.banks)
        except LLMUnavailable:
            sleep(max(pace_seconds, retry_wait_seconds))
            asked += 1
            try:
                return answerer.answer(question.question, question.banks)
            except LLMUnavailable:
                return None

    answered = 0
    cited_gold = 0
    cited_any = 0
    answer_failures: list[str] = []
    refused_ids: list[str] = []
    in_scope_errors = 0
    details: list[dict] = []
    for question in in_scope:
        result = ask(question)
        if result is None:
            in_scope_errors += 1
            continue
        record = {"id": question.id, "refused": bool(result["refused"]), "cited": []}
        if result["refused"]:
            refused_ids.append(question.id)
            answer_failures.append(question.id)
            details.append(record)
            continue
        answered += 1
        cited = {(source["doc"], source["page"]) for source in result["sources"]}
        record["cited"] = sorted(cited)
        details.append(record)
        wide = (answerable or {}).get(question.id) or question.page_keys
        if cited.intersection(question.page_keys):
            cited_gold += 1
        if cited.intersection(wide):
            cited_any += 1
        if not cited.intersection(wide):
            answer_failures.append(question.id)

    wrongly_answered: list[str] = []
    out_of_scope_errors = 0
    for question in out_of_scope:
        result = ask(question)
        if result is None:
            out_of_scope_errors += 1
            continue
        details.append(
            {
                "id": question.id,
                "refused": bool(result["refused"]),
                "cited": sorted((source["doc"], source["page"]) for source in result["sources"]),
            }
        )
        if not result["refused"]:
            wrongly_answered.append(question.id)

    scored = len(out_of_scope) - out_of_scope_errors
    refused_out_of_scope = scored - len(wrongly_answered)
    return {
        "in_scope": {
            "questions": len(in_scope),
            "answered": answered,
            "refused": len(refused_ids),
            "errors": in_scope_errors,
            "answer_rate": round(answered / len(in_scope), 4) if in_scope else 0.0,
            "citation_hit_rate": round(cited_gold / answered, 4) if answered else 0.0,
            "citation_answerable_rate": round(cited_any / answered, 4) if answered else 0.0,
            "failures": answer_failures,
            "refused_ids": refused_ids,
        },
        "out_of_scope": {
            "questions": len(out_of_scope),
            "refused": refused_out_of_scope,
            "errors": out_of_scope_errors,
            "refusal_rate": round(refused_out_of_scope / scored, 4) if scored else 0.0,
            "wrongly_answered": wrongly_answered,
        },
        "details": details,
    }


def verify_gold(gold: list[GoldQuestion], index_dir: Path) -> list[str]:
    """Check the gold set against the real index: values present, out-of-scope terms absent."""
    page_text = index_page_text(index_dir)
    everything = " ".join(page_text.values()).casefold()

    problems: list[str] = []
    for question in gold:
        if question.out_of_scope:
            for term in question.absent_terms:
                if term.casefold() in everything:
                    problems.append(
                        f"{question.id}: marked out of scope but the documents mention {term!r}"
                    )
            continue
        for key in sorted(question.page_keys):
            if key not in page_text:
                problems.append(f"{question.id}: page {key[1]} of {key[0]} is not in the index")
                continue
            if not any(value in page_text[key] for value in question.must_contain):
                values = list(question.must_contain)
                problems.append(f"{question.id}: none of {values} found on {key[1]} of {key[0]}")
    return problems


def rescore_answers(
    report: dict, gold: Sequence[GoldQuestion], answerable: dict[str, set[PageKey]] | None = None
) -> dict:
    """Recompute the answer metrics from a stored report, so scoring costs no quota."""
    by_id = {question.id: question for question in gold}
    answered = cited_gold = cited_any = 0
    failures: list[str] = []
    refused_ids: list[str] = []
    in_scope = 0
    for entry in report.get("details", []):
        question = by_id.get(entry["id"])
        if question is None or question.out_of_scope:
            continue
        in_scope += 1
        if entry["refused"]:
            refused_ids.append(question.id)
            failures.append(question.id)
            continue
        answered += 1
        cited = {tuple(pair) for pair in entry["cited"]}
        wide = (answerable or {}).get(question.id) or question.page_keys
        if cited.intersection(question.page_keys):
            cited_gold += 1
        if cited.intersection(wide):
            cited_any += 1
        else:
            failures.append(question.id)

    out_of_scope = [
        entry
        for entry in report.get("details", [])
        if by_id.get(entry["id"]) is not None and by_id[entry["id"]].out_of_scope
    ]
    wrongly_answered = [entry["id"] for entry in out_of_scope if not entry["refused"]]
    scored = len(out_of_scope)
    previous = report.get("in_scope", {})
    return {
        "in_scope": {
            "questions": in_scope,
            "answered": answered,
            "refused": len(refused_ids),
            "errors": previous.get("errors", 0),
            "answer_rate": round(answered / in_scope, 4) if in_scope else 0.0,
            "citation_hit_rate": round(cited_gold / answered, 4) if answered else 0.0,
            "citation_answerable_rate": round(cited_any / answered, 4) if answered else 0.0,
            "failures": failures,
            "refused_ids": refused_ids,
        },
        "out_of_scope": {
            "questions": scored,
            "refused": scored - len(wrongly_answered),
            "errors": previous.get("out_of_scope", {}).get("errors", 0),
            "refusal_rate": round((scored - len(wrongly_answered)) / scored, 4) if scored else 0.0,
            "wrongly_answered": wrongly_answered,
        },
        "details": report.get("details", []),
    }


def print_retrieval(report: dict) -> None:
    print(f"retrieval over {report['questions']} questions, top-{report['top_k']}")
    print(f"{'method':<8} {'strict hit':>10} {'strict mrr':>10} {'any page':>9} {'any mrr':>8}")
    for method, stats in report["methods"].items():
        print(
            f"{method:<8} {stats['hit_rate']:>10.3f} {stats['mrr']:>10.3f} "
            f"{stats['answerable_hit_rate']:>9.3f} {stats['answerable_mrr']:>8.3f}"
        )
    for method, stats in report["methods"].items():
        if stats["missed"]:
            print(f"  {method} missed: {', '.join(stats['missed'])}")


def print_answers(report: dict) -> None:
    inside = report["in_scope"]
    outside = report["out_of_scope"]
    print(f"in-scope questions           : {inside['questions']}")
    print(f"  answered                   : {inside['answered']} ({inside['answer_rate']:.3f})")
    print(f"  refused                    : {inside['refused']}")
    print(f"  ran out of quota           : {inside['errors']}")
    print(f"  cited our hand-picked page : {inside['citation_hit_rate']:.3f} of answers")
    print(f"  cited a page holding it    : {inside['citation_answerable_rate']:.3f} of answers")
    if inside["refused_ids"]:
        print(f"  refused anyway             : {', '.join(inside['refused_ids'])}")
    unusable = [item for item in inside["failures"] if item not in inside["refused_ids"]]
    if unusable:
        print(f"  cited nothing usable       : {', '.join(unusable)}")
    print(f"out-of-scope questions       : {outside['questions']}")
    print(f"  correctly refused          : {outside['refusal_rate']:.3f}")
    if outside["wrongly_answered"]:
        print(f"  wrongly answered           : {', '.join(outside['wrongly_answered'])}")


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        description="Score retrieval and answers against the gold set."
    )
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--methods", nargs="+", choices=RETRIEVAL_METHODS, default=list(RETRIEVAL_METHODS)
    )
    parser.add_argument("--retrieval-only", action="store_true", help="skip every model call")
    parser.add_argument(
        "--verify-gold", action="store_true", help="check the gold set against the index"
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=4.5,
        help="pause between questions, because the free tier allows about 15 per minute",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help="recompute the answer metrics of an earlier report without calling the model",
    )
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)

    if args.rescore:
        stored = json.loads(args.rescore.read_text(encoding="utf-8"))
        answerable = answerable_pages(gold, index_page_text(args.index_dir))
        stored["answers"] = rescore_answers(stored["answers"], gold, answerable)
        print_answers(stored["answers"])
        write_report(args.rescore, stored)
        return 0

    if args.verify_gold:
        problems = verify_gold(gold, args.index_dir)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1
        checked = sum(len(question.pages) for question in gold if not question.out_of_scope)
        print(f"gold set is consistent: {len(gold)} questions, {checked} pages checked")
        return 0

    retriever = Retriever.from_index(args.index_dir, settings)
    answerable = None
    try:
        answerable = answerable_pages(gold, index_page_text(args.index_dir))
    except FileNotFoundError as exc:
        print(f"{exc} (strict scores only)", file=sys.stderr)

    payload: dict = {
        "gold": str(args.gold),
        "questions": len(gold),
        "retrieval": evaluate_retrieval(retriever, gold, args.methods, args.top_k, answerable),
    }
    print_retrieval(payload["retrieval"])

    if not args.retrieval_only:
        answerer: Answerer = get_answerer(settings, args.index_dir)
        payload["answers"] = evaluate_answers(
            answerer, gold, args.pace_seconds, answerable=answerable
        )
        print()
        print_answers(payload["answers"])

    if args.output:
        write_report(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
