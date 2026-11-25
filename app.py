"""Streamlit entry point.

The chat interface is not wired up yet. This placeholder only confirms that the
pinned Streamlit version and the theme load correctly.
"""

import streamlit as st

from rag.config import BANKS

st.set_page_config(page_title="Chatbot laporan bank IDX", layout="centered")

st.title("Chatbot laporan bank IDX")
st.caption("Tanya jawab dari laporan tahunan publik")
st.write("Indeks bank yang tersedia:", ", ".join(BANKS))
st.caption("Bukan nasihat keuangan. Jawaban bersumber dari laporan tahunan publik.")
