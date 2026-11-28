"""Streamlit front end for the bank report chatbot.

Layout, copy, and states follow the project UI specification. Retrieval and prompting
live in rag/, so this file only calls answer_question(question, banks).
"""

from __future__ import annotations

import time

import streamlit as st

from rag.config import BANKS, Settings
from rag.generator import LLMUnavailable
from rag.index_bundle import IndexUnavailable, ensure_index
from rag.pipeline import answer_question, get_answerer


class SetupError(RuntimeError):
    """The app cannot answer because the index or the API key is missing."""


APP_TITLE = "Chatbot laporan bank IDX"
SUBTITLE = "Tanya jawab dari laporan tahunan publik"
EMPTY_STATE = "Pilih contoh pertanyaan di sidebar atau ketik pertanyaan Anda di bawah."
EXAMPLES = (
    "Berapa laba bersih tahun 2024?",
    "Bagaimana tren rasio NPL?",
    "Apa risiko utama yang dilaporkan?",
)
MAX_INPUT_CHARS = 500
MIN_SECONDS_BETWEEN_REQUESTS = 3

REFUSAL = (
    "Informasi itu tidak ada di dokumen yang tersedia. "
    "Saya hanya menjawab dari laporan tahunan yang sudah diindeks."
)
THROTTLE_MSG = "Terlalu cepat. Tunggu beberapa detik, lalu coba lagi."
API_ERROR_MSG = "Layanan AI sedang sibuk atau kuotanya habis. Coba lagi sebentar lagi."
SETUP_MSG = "Dokumen atau kunci API belum siap, jadi jawaban belum bisa dibuat."
NO_BANK_MSG = "Pilih minimal satu bank di sidebar."
DISCLAIMER = "Bukan nasihat keuangan. Jawaban bersumber dari laporan tahunan publik."


def is_throttled(
    last_request: float, now: float, min_seconds: float = MIN_SECONDS_BETWEEN_REQUESTS
) -> bool:
    """True when the previous request is too recent to send another one."""
    if last_request <= 0:
        return False
    return now - last_request < min_seconds


@st.cache_resource(show_spinner=False)
def load_answerer():
    """Build the retriever, the index, and the model client once per app process."""
    settings = Settings.from_env()
    if not settings.llm_configured:
        raise SetupError("LLM_MODEL and LLM_API_KEY are not set")
    ensure_index(
        settings.index_dir,
        url=settings.index_bundle_url,
        sha256=settings.index_bundle_sha256,
    )
    return get_answerer(settings)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources") or []
        if sources:
            with st.expander(f"Sumber ({len(sources)})"):
                for source in sources:
                    st.markdown(f"**{source['doc']}**, hlm. {source['page']}")
                    st.caption(source["snippet"])


def answer(question: str, banks: list[str]) -> None:
    """Ask the pipeline and render the answer, the sources, or a friendly failure."""
    with st.chat_message("assistant"):
        try:
            with st.spinner("Mencari di dokumen"):
                load_answerer()
                result = answer_question(question, banks)
        except LLMUnavailable:
            st.error(API_ERROR_MSG)
            return
        except (FileNotFoundError, SetupError, IndexUnavailable):
            st.error(SETUP_MSG)
            return
        except Exception:
            st.error(API_ERROR_MSG)
            return

        content = REFUSAL if result["refused"] else result["answer"]
        sources = [] if result["refused"] else result["sources"]
        st.markdown(content)
        if sources:
            with st.expander(f"Sumber ({len(sources)})"):
                for source in sources:
                    st.markdown(f"**{source['doc']}**, hlm. {source['page']}")
                    st.caption(source["snippet"])
        st.session_state.messages.append(
            {"role": "assistant", "content": content, "sources": sources}
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="centered")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader(APP_TITLE)
        st.caption(SUBTITLE)
        selected_banks = st.multiselect("Bank", BANKS, default=BANKS)
        st.write("Contoh pertanyaan")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                st.session_state.pending = example
        if st.button("Hapus percakapan", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title(APP_TITLE)
    if not st.session_state.messages:
        st.write(EMPTY_STATE)

    for message in st.session_state.messages:
        render_message(message)

    typed = st.chat_input("Tanyakan sesuatu tentang laporan bank", max_chars=MAX_INPUT_CHARS)
    question = typed or st.session_state.pop("pending", None)

    if question:
        now = time.time()
        if is_throttled(st.session_state.get("last_request", 0.0), now):
            st.warning(THROTTLE_MSG)
        elif not selected_banks:
            st.warning(NO_BANK_MSG)
        else:
            st.session_state.last_request = now
            user_message = {"role": "user", "content": question}
            st.session_state.messages.append(user_message)
            render_message(user_message)
            answer(question, selected_banks)

    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
