"""Guards for the Streamlit front end: real backend, spec copy, and throttle maths."""

from __future__ import annotations

import tomllib

import pytest

import app
from rag.config import BANKS, PROJECT_ROOT
from rag.pipeline import answer_question

CONFIG_PATH = PROJECT_ROOT / ".streamlit" / "config.toml"


def test_the_app_calls_the_real_pipeline() -> None:
    assert app.answer_question is answer_question


def test_bank_list_matches_the_pipeline() -> None:
    assert tuple(BANKS) == ("BBCA", "BBRI", "BMRI", "BBNI", "BRIS")


def test_copy_matches_the_specification() -> None:
    assert app.APP_TITLE == "Chatbot laporan bank IDX"
    assert app.SUBTITLE == "Tanya jawab dari laporan tahunan publik"
    assert app.REFUSAL == (
        "Informasi itu tidak ada di dokumen yang tersedia. "
        "Saya hanya menjawab dari laporan tahunan yang sudah diindeks."
    )
    assert app.THROTTLE_MSG == "Terlalu cepat. Tunggu beberapa detik, lalu coba lagi."
    assert app.API_ERROR_MSG == (
        "Layanan AI sedang sibuk atau kuotanya habis. Coba lagi sebentar lagi."
    )
    assert app.DISCLAIMER == (
        "Bukan nasihat keuangan. Jawaban bersumber dari laporan tahunan publik."
    )


def test_refusal_wording_is_shared_with_the_backend() -> None:
    from rag.generator import REFUSAL_TEXT

    assert app.REFUSAL == REFUSAL_TEXT


def test_example_questions_match_the_specification() -> None:
    assert app.EXAMPLES == (
        "Berapa laba bersih tahun 2024?",
        "Bagaimana tren rasio NPL?",
        "Apa risiko utama yang dilaporkan?",
    )


def test_input_limit_and_throttle_match_the_specification() -> None:
    assert app.MAX_INPUT_CHARS == 500
    assert app.MIN_SECONDS_BETWEEN_REQUESTS == 3


def test_throttle_blocks_a_second_request_inside_the_window() -> None:
    assert app.is_throttled(last_request=100.0, now=101.0) is True
    assert app.is_throttled(last_request=100.0, now=103.0) is False
    assert app.is_throttled(last_request=0.0, now=1.0) is False


def test_user_facing_copy_avoids_exclamation_marks_and_please() -> None:
    for text in (app.REFUSAL, app.THROTTLE_MSG, app.API_ERROR_MSG, app.DISCLAIMER, app.EMPTY_STATE):
        assert "!" not in text
        assert "mohon" not in text.casefold()


def test_theme_file_matches_the_specification() -> None:
    theme = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))["theme"]

    assert theme == {
        "base": "light",
        "primaryColor": "#1F4E79",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F3F5F8",
        "textColor": "#1B2430",
        "font": "sans serif",
    }


@pytest.mark.parametrize("name", ["answer", "render_message", "load_answerer", "main"])
def test_expected_entry_points_exist(name: str) -> None:
    assert callable(getattr(app, name))
