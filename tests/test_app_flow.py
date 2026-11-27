"""End-to-end UI checks driven by Streamlit's own test harness.

The pipeline is replaced by a stub, so the whole conversation flow can be exercised
without loading the index or spending free-tier quota.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

import app
import rag.pipeline
from rag.config import PROJECT_ROOT, Settings
from rag.generator import LLMUnavailable

APP_PATH = PROJECT_ROOT / "app.py"

ANSWER = {
    "answer": "Laba bersih naik [BBCA_2024, hlm. 647].",
    "sources": [{"doc": "BBCA_2024", "page": 647, "snippet": "Laba bersih 54,8 triliun."}],
    "refused": False,
}
REFUSAL_RESULT = {"answer": "tidak ada di dokumen", "sources": [], "refused": True}


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch) -> tuple[AppTest, list[tuple[str, list[str]]]]:
    """Return a running app whose pipeline answers from a stub."""
    calls: list[tuple[str, list[str]]] = []

    def fake_answer_question(question: str, banks: list[str] | None = None) -> dict:
        calls.append((question, list(banks or [])))
        return REFUSAL_RESULT if "saham" in question else ANSWER

    monkeypatch.setattr(rag.pipeline, "answer_question", fake_answer_question)
    monkeypatch.setattr(rag.pipeline, "get_answerer", lambda *args, **kwargs: None)

    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    return app, calls


def ask(app: AppTest, question: str) -> None:
    """Submit a question without tripping the three second throttle."""
    app.session_state["last_request"] = 0.0
    app.chat_input[0].set_value(question).run()


def test_layout_matches_the_specification(chat) -> None:
    app, calls = chat

    assert not app.exception
    assert [title.value for title in app.title] == ["Chatbot laporan bank IDX"]
    assert [button.label for button in app.sidebar.button] == [
        "Berapa laba bersih tahun 2024?",
        "Bagaimana tren rasio NPL?",
        "Apa risiko utama yang dilaporkan?",
        "Hapus percakapan",
    ]
    assert app.sidebar.multiselect[0].value == ["BBCA", "BBRI", "BMRI", "BBNI", "BRIS"]
    assert app.chat_input[0].placeholder == "Tanyakan sesuatu tentang laporan bank"
    assert calls == []


def test_the_empty_state_is_never_a_blank_page(chat) -> None:
    app, _ = chat

    shown = [block.value for block in app.markdown]
    assert any("Pilih contoh pertanyaan" in text for text in shown)
    assert next(caption.value for caption in app.caption) == (
        "Bukan nasihat keuangan. Jawaban bersumber dari laporan tahunan publik."
    )


def test_an_example_button_answers_and_lists_the_sources(chat) -> None:
    app, calls = chat

    app.sidebar.button[0].click().run()

    assert not app.exception
    assert calls == [("Berapa laba bersih tahun 2024?", ["BBCA", "BBRI", "BMRI", "BBNI", "BRIS"])]
    messages = app.session_state["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Berapa laba bersih tahun 2024?"
    rendered = [block.value for block in app.chat_message[1].markdown]
    assert "Laba bersih naik [BBCA_2024, hlm. 647]." in rendered
    assert "**BBCA_2024**, hlm. 647" in rendered


def test_a_typed_question_uses_the_selected_banks(chat) -> None:
    app, calls = chat

    app.sidebar.multiselect[0].set_value(["BBRI", "BMRI"]).run()
    ask(app, "Bagaimana rasio NPL?")

    assert calls == [("Bagaimana rasio NPL?", ["BBRI", "BMRI"])]


def test_a_refusal_hides_the_sources(chat) -> None:
    app, _ = chat

    ask(app, "Berapa harga saham besok?")

    last = app.session_state["messages"][-1]
    assert last["role"] == "assistant"
    assert last["sources"] == []
    assert last["content"].startswith("Informasi itu tidak ada di dokumen yang tersedia.")


def test_a_second_question_inside_the_window_is_throttled(chat) -> None:
    app, calls = chat

    ask(app, "Bagaimana rasio NPL?")
    app.chat_input[0].set_value("Dan bagaimana laba?").run()

    assert [warning.value for warning in app.warning] == [
        "Terlalu cepat. Tunggu beberapa detik, lalu coba lagi."
    ]
    assert len(calls) == 1


def test_asking_without_any_bank_warns_and_does_not_call_the_pipeline(chat) -> None:
    app, calls = chat

    app.sidebar.multiselect[0].set_value([]).run()
    ask(app, "Bagaimana rasio NPL?")

    assert [warning.value for warning in app.warning] == ["Pilih minimal satu bank di sidebar."]
    assert calls == []


def test_clearing_the_conversation_empties_the_chat(chat) -> None:
    app, _ = chat
    ask(app, "Bagaimana rasio NPL?")
    assert app.session_state["messages"]

    app.sidebar.button[-1].click().run()

    assert app.session_state["messages"] == []


def test_a_busy_model_shows_the_friendly_message(chat, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = chat

    def unavailable(question: str, banks: list[str] | None = None) -> dict:
        raise LLMUnavailable("quota exhausted")

    monkeypatch.setattr(rag.pipeline, "answer_question", unavailable)
    ask(app, "Bagaimana rasio NPL?")

    assert [error.value for error in app.error] == [
        "Layanan AI sedang sibuk atau kuotanya habis. Coba lagi sebentar lagi."
    ]


def test_a_missing_index_shows_the_setup_message(chat, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = chat

    def missing(question: str, banks: list[str] | None = None) -> dict:
        raise FileNotFoundError("no index")

    monkeypatch.setattr(rag.pipeline, "answer_question", missing)
    ask(app, "Bagaimana rasio NPL?")

    assert [error.value for error in app.error] == [
        "Dokumen atau kunci API belum siap, jadi jawaban belum bisa dibuat."
    ]


def test_a_missing_api_key_is_reported_as_a_setup_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unconfigured = Settings(llm_model=None, llm_api_key=None)
    monkeypatch.setattr(app.Settings, "from_env", classmethod(lambda cls: unconfigured))
    app.load_answerer.clear()

    with pytest.raises(app.SetupError):
        app.load_answerer()
