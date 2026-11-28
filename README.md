# IDX Bank Report RAG Chatbot

A retrieval-augmented chatbot that answers Indonesian questions about the 2023 and
2024 annual reports of five Indonesian banks: BBCA, BBRI, BMRI, BBNI, and BRIS.

Every answer cites the report and PDF page it came from, and the app refuses to
answer when the indexed reports do not contain the evidence.

**Live demo:** <https://idx-bank-rag-xfgell9rh2rypslppule7h.streamlit.app/>

Example questions:

- Berapa laba bersih BBCA tahun 2024?
- Bagaimana tren rasio NPL Bank Mandiri?
- Apa risiko utama yang dilaporkan BRIS?
- Berapa CAR BBNI tahun 2023?

## How it works

```
PDF ──► page text ──► overlapping chunks ──► e5-small embeddings ──► FAISS per bank
                                                    │
question ──► query expansion ──► BM25 + vector search ──► RRF ──► top 6 passages
                                                    │
                                              Gemini writes an answer
                                                    │
                                    citation check ──► answer or refusal
```

1. Each PDF page is extracted with PyMuPDF and split into 400–450 token chunks with
   12% overlap. A chunk never crosses a page boundary, so a citation always points at
   the page that holds the text.
2. Chunks are embedded with `intfloat/multilingual-e5-small` through ONNX Runtime on
   the CPU, so no GPU or PyTorch install is needed.
3. Retrieval is hybrid: dense search finds paraphrases, BM25 finds exact terms such as
   `NPL` or `CAR`, and reciprocal rank fusion merges the two rankings.
4. The six best passages are sent to Gemini with instructions to answer only from
   them. Every citation in the reply is checked against the passages that were
   actually retrieved; an answer without a valid citation becomes a refusal.

## Data

| Bank | Reports | Chunks |
|---|---|---|
| BBCA — Bank Central Asia | 2023, 2024 | 2,908 |
| BBRI — Bank Rakyat Indonesia | 2023, 2024 | 5,264 |
| BMRI — Bank Mandiri | 2023, 2024 | 6,458 |
| BBNI — Bank Negara Indonesia | 2023, 2024 | 6,495 |
| BRIS — Bank Syariah Indonesia | 2023, 2024 | 2,729 |

10 reports, 11,440 indexed pages, 23,854 chunks. Pages without extractable text are
skipped during indexing, so the page counts here are slightly lower than the PDF page
counts. The PDFs are not redistributed in this repository; `data/raw/README.md` lists
the sources.

## Results

Measured on a hand-judged set of 36 questions (30 answerable, 6 out of scope) with
top-6 retrieval:

| Retrieval | Any-page hit | Any-page MRR |
|---|---|---|
| Hybrid (shipped) | 0.933 | 0.712 |
| BM25 only | 0.833 | 0.608 |
| Vector only | 0.800 | 0.653 |

- 27 of 30 answerable questions answered (0.900).
- 25 of those 27 cited a page that really contains the figure (0.926).
- 6 of 6 out-of-scope questions refused (1.000).

Three answerable questions are refused because the passage holding the figure is not
in the top six, and two answers cite a page that does not hold the figure. No answer
invented a number. Reproduce with:

```bash
python -m rag.evaluate --top-k 6 --pace-seconds 5
```

## Limitations

- Citations use PDF page numbers, not the printed page numbers in the report.
- Tables and charts are read optimistically. A chart-heavy page may be extracted as a
  run of numbers without their column headers, and the answer can then associate a
  figure with the wrong period.
- The free Gemini tier allows 500 requests per day and roughly 15 per minute. The app
  throttles one question every three seconds per session and caches repeated
  questions, but a busy demo can still hit the daily cap.
- Indonesian is the only supported language, because that is the language of the
  reports and of the interface.
- Each question is answered on its own. The chat keeps the history on screen, but a
  follow-up such as "kalau 2023?" is not linked to the question before it.
- With several banks selected, a question that names no bank is answered from the six
  best passages, which may cover only some of the selected banks. Pick a bank in the
  sidebar for a focused answer.

## Local setup

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
cp .env.example .env          # then set LLM_API_KEY and LLM_MODEL
```

The index is downloaded on first use. To run the app:

```bash
streamlit run app.py
```

## Index bundle

The index is about 80 MB of derived data: too large for git, and too slow to rebuild
on a hosted machine. It is published as the `index-v1` release asset. On first start
the app downloads it, verifies the SHA-256 checksum, and unpacks it into `index/`.

```bash
python -m rag.index_bundle fetch   # download the published bundle
python -m rag.index_bundle pack    # rebuild the release asset from index/
```

To rebuild the index from the PDFs instead, put them in `data/raw/` and run:

```bash
python -m rag.ingest
```

That takes roughly 30 minutes on a laptop CPU.

## Deploy

The app runs on Streamlit Community Cloud with no paid services.

1. Create the app from this repository with `app.py` as the entry point, and select
   Python 3.11 under Advanced settings. `.python-version` records the same version for
   local runs.
2. Add `LLM_MODEL` and `LLM_API_KEY` in the app's secrets editor; the format is shown
   in `.streamlit/secrets.toml.example`.
3. Deploy. The first question downloads the index bundle and loads the embedding model,
   which takes under a minute; later answers take a few seconds.

## Checks

```bash
ruff check .
ruff format --check .
pytest
```

The UI tests drive the real Streamlit app through `streamlit.testing`, with the model
stubbed, so they run offline and do not spend API quota.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — data flow, component design, and the
  reasoning behind each major decision.
- [`docs/evaluation.md`](docs/evaluation.md) — the gold question set, metric
  definitions, and measured results.

## Project layout

```
app.py               Streamlit interface
rag/config.py        settings, bank metadata, query expansions
rag/ingest.py        PDF discovery, page extraction, index build
rag/chunking.py      token-aware overlapping chunks
rag/embeddings.py    e5-small through fastembed/ONNX
rag/retriever.py     BM25, FAISS, reciprocal rank fusion
rag/generator.py     Gemini client, retries, daily cap, cache
rag/pipeline.py      prompt assembly, citation validation, refusals
rag/index_bundle.py  release-asset packing and fetching
rag/evaluate.py      retrieval and answer scoring
eval/                gold questions and measured results
docs/                architecture and evaluation notes
```
