# Evaluation

The gold set and the scoring harness exist to answer one question: does the system
answer correctly and cite the page it actually used, or does it only look plausible?
Every question was checked against the PDFs by hand.

## Gold set

`eval/gold_questions.jsonl` holds 36 questions: 30 answerable and 6 deliberately out of
scope.

```json
{
  "id": "bbca-2023-profit",
  "question": "Berapa laba bersih BCA tahun 2023?",
  "banks": ["BBCA"],
  "doc_id": "BBCA_2023",
  "pages": [39],
  "must_contain": ["48,6", "48.658"],
  "topic": "profit"
}
```

| Field | Meaning |
|---|---|
| `pages` | The page picked by hand as the best place to answer from |
| `must_contain` | Spellings of the answer figure as printed in the report |
| `absent_terms` | For out-of-scope questions, terms that must not appear in an answer |
| `topic` | Profit, capital, assets, NPL, deposits, and similar |

The set covers all five banks and both years, and mixes direct questions with
paraphrased ones.

## Metrics

Retrieval is measured without calling the model, so it can run as often as needed.

- **Strict hit** — the hand-picked page appears in the top k. This is the hardest
  measure: a bank states the same figure in its highlights, its management discussion,
  and its financial statements, so a correct answer can come from a page other than the
  one picked by hand.
- **Any-page hit** — any retrieved page contains one of the `must_contain` spellings.
  This is the measure that matters for answer quality, because the model can only cite
  what it was given.
- **MRR** — mean reciprocal rank of the first correct page. A page at rank 1 scores
  1.0, rank 2 scores 0.5, and so on.

Answer quality needs model calls, so it is measured less often and paced to stay
inside the free tier:

- **Answer rate** — answered questions out of the answerable set.
- **Citation hit rate** — answers that cited the hand-picked page.
- **Citation answerable rate** — answers that cited a page that really contains the
  figure. This is the honest citation measure.
- **Refusal rate on out-of-scope questions** — higher is better.

## Running it

```bash
python -m rag.evaluate --top-k 6 --pace-seconds 5
```

Retrieval is scored for all three methods (`hybrid`, `vector`, `bm25`). Answer quality
runs only on the shipped method. `--pace-seconds` is the gap between questions; 5
seconds keeps a run under the 15 requests per minute limit.

To rescore stored answers after changing the gold set, without spending quota:

```bash
python -m rag.evaluate --rescore eval/results/latest.json
```

## Results

Measured at temperature 0 with top-6 retrieval:

| Method | Strict hit | Strict MRR | Any-page hit | Any-page MRR |
|---|---|---|---|---|
| Hybrid (shipped) | 0.567 | 0.336 | 0.933 | 0.712 |
| BM25 only | 0.433 | 0.297 | 0.833 | 0.608 |
| Vector only | 0.300 | 0.211 | 0.800 | 0.653 |

Answer quality:

| Measure | Result |
|---|---|
| Answerable questions answered | 27 / 30 (0.900) |
| Answers citing the hand-picked page | 0.407 |
| Answers citing a page holding the figure | 25 / 27 (0.926) |
| Out-of-scope questions refused | 6 / 6 (1.000) |

## Reading the numbers

Hybrid retrieval finds a usable page for 28 of 30 questions. The strict hit rate is
lower because the gold page is one of several valid places for a figure, not the only
one.

Three answerable questions are refused: `bbri-2023-assets`, `bbni-2023-profit`, and
`bris-2023-car`. These are stable across runs, so they are retrieval misses rather than
model noise: the passage that holds the figure is not in the top six.

Two answers cite a page that does not hold the figure: `bmri-2024-profit` and
`bris-2024-profit`. Both come from table-heavy pages where the extracted text loses
column headers, which is the known chart and table failure mode.

The low strict citation rate (0.407) is expected for the same reason as the low strict
hit rate: the model often cites a highlights page while the hand-picked page is the
statement of financial position. The answerable citation rate (0.926) is the number
that reflects whether a reader can verify the answer.

## Cost control

- One run costs 30 model requests, about three minutes at 5 seconds per question.
- Retrieval-only scoring is free and runs on every change.
- Stored answers can be rescored offline when the gold set gains spellings.
