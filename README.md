# IDX Bank Report RAG Chatbot

A retrieval-augmented chatbot that answers Indonesian questions about the 2023 and
2024 annual reports of five Indonesian banks: BBCA, BBRI, BMRI, BBNI, and BRIS.

Every answer cites the document and page it came from, and the app refuses to
answer when the indexed reports do not contain the evidence.

> Work in progress.

## Local setup

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
cp .env.example .env
```

## Checks

```bash
ruff check .
ruff format --check .
pytest
```
