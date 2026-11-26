"""Turns a question into a cited Indonesian answer.

Retrieval provides the passages, the model writes the answer, and this module is
the only place that decides whether that answer is allowed to reach the user.
Every citation is checked against the passages that were actually retrieved, so a
made-up page number is dropped instead of printed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from rag.config import Settings
from rag.generator import REFUSAL_TEXT, GeminiClient, Generator, LLMUnavailable, TextModel
from rag.retriever import Retriever
from rag.schemas import RetrievedChunk

CITATION_PATTERN = re.compile(r"\[\s*([A-Za-z]{2,5}_\d{4})\s*,\s*hlm\.?\s*(\d+)\s*\]")
DOUBLED_SPACES = re.compile(r"[ \t]{2,}")
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:])")
REFUSAL_PREFIX = "informasi itu tidak ada"
ANSWER_PREFIX = re.compile(r"^\s*(jawaban|answer)\s*:\s*", re.IGNORECASE)

PROMPT_TEMPLATE = """KUTIPAN:
{context}

PERTANYAAN: {question}

JAWABAN:"""


def build_prompt(question: str, results: list[RetrievedChunk], max_context_chars: int) -> str:
    """Render retrieved passages into labelled context blocks."""
    blocks: list[str] = []
    used = 0
    for result in results:
        chunk = result.chunk
        block = f"[{chunk.doc_id}, hlm. {chunk.page}]\n{chunk.text.strip()}"
        if blocks and used + len(block) > max_context_chars:
            break
        blocks.append(block)
        used += len(block)
    return PROMPT_TEMPLATE.format(context="\n\n".join(blocks), question=question)


def normalize_answer(text: str) -> str:
    """Collapse whitespace and clean up the gaps left by removed citations."""
    text = " ".join(text.replace("\r", "\n").split())
    text = DOUBLED_SPACES.sub(" ", text)
    text = SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    return ANSWER_PREFIX.sub("", text).strip()


def is_refusal(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return (
        normalized.startswith(REFUSAL_PREFIX)
        or " ".join(REFUSAL_TEXT.casefold().split()) in normalized
    )


def validate_answer(raw: str, results: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
    """Strip citations the model invented, then report which passages it used."""
    allowed: dict[tuple[str, int], RetrievedChunk] = {
        (result.chunk.doc_id.upper(), result.chunk.page): result for result in results
    }
    cited: list[RetrievedChunk] = []

    def replace(match: re.Match[str]) -> str:
        key = (match.group(1).upper(), int(match.group(2)))
        result = allowed.get(key)
        if result is None:
            return ""
        if result not in cited:
            cited.append(result)
        return f"[{key[0]}, hlm. {key[1]}]"

    text = normalize_answer(CITATION_PATTERN.sub(replace, raw))
    if is_refusal(text) or not cited:
        return REFUSAL_TEXT, []
    return text, cited


class Answerer:
    """Connects the retriever to the generator and enforces the grounding rules."""

    def __init__(self, retriever: Retriever, generator: Generator, settings: Settings) -> None:
        self.retriever = retriever
        self.generator = generator
        self.settings = settings

    def answer(self, question: str, banks: list[str] | None = None) -> dict:
        """Return {"answer", "sources", "refused"} for a question."""
        question = question.strip()
        if not question:
            return refusal()

        results = [
            result
            for result in self.retriever.search(question, banks)
            if result.score >= self.settings.min_relevance_score
        ]
        if not results:
            return refusal()

        prompt = build_prompt(question, results, self.settings.max_context_chars)
        answer, cited = validate_answer(self.generator.generate(prompt), results)
        if not cited:
            return refusal()
        return {
            "answer": answer,
            "sources": [result.to_source() for result in cited],
            "refused": False,
        }


def refusal() -> dict:
    """The single refusal response shape, used by every failing path."""
    return {"answer": REFUSAL_TEXT, "sources": [], "refused": True}


_ANSWERER: Answerer | None = None


def get_answerer(settings: Settings | None = None, index_dir: Path | None = None) -> Answerer:
    """Build the pipeline once per process; the UI caches this with st.cache_resource."""
    global _ANSWERER
    if _ANSWERER is None:
        settings = settings or Settings.from_env()
        if not settings.llm_configured:
            raise LLMUnavailable("LLM_MODEL and LLM_API_KEY must be set before answering")
        assert settings.llm_api_key is not None and settings.llm_model is not None
        retriever = Retriever.from_index(index_dir or settings.index_dir, settings)
        model: TextModel = GeminiClient(settings.llm_api_key, settings.llm_model, settings)
        _ANSWERER = Answerer(retriever, Generator(model, settings), settings)
    return _ANSWERER


def answer_question(question: str, banks: list[str] | None = None) -> dict:
    """Answer one question using the process-wide pipeline."""
    return get_answerer().answer(question, banks)


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Ask the annual report index a question.")
    parser.add_argument("question", help="question in Indonesian")
    parser.add_argument(
        "--banks", nargs="+", default=None, help="subset of BBCA BBRI BMRI BBNI BRIS"
    )
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    args = parser.parse_args(argv)

    try:
        result = get_answerer(settings, args.index_dir).answer(args.question, args.banks)
    except LLMUnavailable as exc:
        print(f"LLM unavailable: {exc}", file=sys.stderr)
        return 1

    print(result["answer"])
    if result["sources"]:
        print()
        for source in result["sources"]:
            print(f"- {source['doc']}, hlm. {source['page']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
