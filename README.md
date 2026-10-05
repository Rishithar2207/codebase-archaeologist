# Codebase Archaeologist

[![tests](https://github.com/Rishithar2207/codebase-archaeologist/actions/workflows/tests.yml/badge.svg)](https://github.com/Rishithar2207/codebase-archaeologist/actions/workflows/tests.yml)

Ask a Python repository questions in English. Get answers cited to file and line —
or an honest "not in the retrieved code."

```
$ python arch.py ask ~/code/crypto-ingestion "how does it decide a price movement is worth flagging"

query: 'how does it decide a price movement is worth flagging'  (paraphrase -> vector)

  retrieved tests/test_anomaly.py:24-31
  retrieved tests/test_anomaly.py:34-42
  retrieved app/anomaly.py:34-85
  retrieved app/anomaly.py:1-23
  retrieved tests/test_idempotency.py:32-47

A price movement is flagged as a "price_spike" when it meets both statistical
outlier and meaningful percentage criteria: its absolute z-score must be >=
Z_THRESHOLD (3.0) and its percentage move from the baseline mean must be >=
MIN_PCT_MOVE (2.0%) (app/anomaly.py:49-58, app/anomaly.py:11-20).
```

---

## The result

Most projects like this stop at "I implemented hybrid search." This one measured it,
and hybrid search turned out to be the wrong answer.

Evaluated on **41 labelled questions** against a 14-file Python service:

| | R@1 | R@5 | MRR |
|---|---|---|---|
| BM25 | 0.24 | 0.39 | 0.30 |
| Vector (MiniLM) | 0.37 | 0.71 | 0.50 |
| Hybrid (RRF) | 0.27 | 0.59 | 0.37 |
| **Routed** | **0.44** | **0.73** | **0.55** |

**Routing beats fusion by 49% on MRR, and beats the better single retriever by 10%.**

Split by query type, the reason is obvious:

| | BM25 MRR | Vector MRR | Hybrid MRR | Routed MRR |
|---|---|---|---|---|
| Identifier queries (n=10) | **1.00** | 0.78 | 0.90 | **1.00** |
| Paraphrase queries (n=31) | 0.07 | **0.41** | 0.20 | **0.41** |

Each retriever is excellent at one kind of question and poor at the other. Unweighted
Reciprocal Rank Fusion gives them equal votes, so on every query the weaker one is
evicting the stronger one's answers. Routing picks the right tool per query instead.

Reproduce it:

```bash
python arch.py eval ~/code/your-repo questions.json
```

---

## Quickstart

```bash
git clone https://github.com/Rishithar2207/codebase-archaeologist.git
cd codebase-archaeologist
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# retrieval only -- no API key needed
python arch.py route ~/code/some-repo "where do duplicate rows get prevented"

# natural-language answers
export GEMINI_API_KEY='your-key'
python arch.py ask ~/code/some-repo "where do duplicate rows get prevented"
```

First run downloads ~90MB of model weights. Embeddings are cached to `.cache/`,
keyed by a hash of the chunk set, so re-running on an unchanged repo skips the
model entirely.

### Commands

| | |
|---|---|
| `python arch.py chunks <repo>` | chunk statistics |
| `python arch.py bm25 <repo> <query>` | keyword retrieval only |
| `python arch.py vec <repo> <query>` | embedding retrieval only |
| `python arch.py hybrid <repo> <query>` | all three rankings side by side |
| `python arch.py route <repo> <query>` | routed retrieval (the recommended path) |
| `python arch.py ask <repo> <query>` | routed retrieval → Gemini → cited answer |
| `python arch.py eval <repo> <questions.json>` | the full evaluation |
| `python arch.py serve <repo> [questions.json]` | FastAPI on :8000, web UI at / and docs at /docs |

Passing the question set to `serve` turns on the evaluation view in the browser.

### The web UI

`serve` puts a page on `http://127.0.0.1:8000` whose point is to make the routing
decision visible *as you type*. Typing `jsonable_encoder` turns the badge orange and
names the token it matched; typing "how does it encode json" turns it green and says no
symbol was named. Classification is its own endpoint — it runs the router and nothing
else, so it costs a regex and a set lookup per keystroke, with no retrieval and no model.

Every BM25 result is shown **with its arithmetic**: each query term's contribution,
its term frequency and idf, and whether `NAME_BOOST` fired. Querying `jsonable_encoder`
against FastAPI, the definition scores 51.5 and the runner-up 16.6, and the breakdown
says exactly why — `tf 35` against `tf 1`, times a boost the other chunk does not get.
The matched terms are highlighted in the source. Vector hits show no such breakdown,
because cosine similarity does not have one; the UI says so rather than inventing an
explanation, and that asymmetry is a fair summary of the trade between the two.

The itemisation is a second implementation of the BM25 formula rather than a refactor
of the scoring loop — building a per-term dict for all 557 chunks would cost more than
the retrieval it describes. A test asserts the two agree for every chunk over several
queries, which is what stops them drifting apart.

**Compare mode** is the argument of this README, run live on one query. It puts BM25,
the vector index and their RRF fusion side by side over the same input, marks the
column the router would have picked, and then states where fusion's top result came
from — typically something one retriever ranked 4th and the other never returned at
all, now sitting above a chunk the right retriever had at rank 1. The aggregate table
above says fusion loses; this shows the single-query mechanism by which it loses.

**The evaluation view** is the table at the top of this README, live, with the
per-question ranks underneath it. Every labelled question is a row showing where each
of the four retrievers put the expected chunk — 1, a worse rank, or a dash for a miss.
Filter to the unanswered ones to see the ceiling directly. Clicking any row loads that
question into compare mode, so an aggregate you distrust is two clicks from the
retrieval that produced it.

Below everything, a grid of one square per chunk in the index shows which 5 of the
several hundred were retrieved, coloured by which retriever found them. Hovering a
square names its chunk, highlights it in the results and lights up the rest of its
file; hovering a result finds its square. It makes the selectivity of retrieval
obvious in a way a list of five filenames does not.

`/` focuses the query box, `Esc` clears it, and the URL carries the query and mode —
so a specific demo can be linked rather than described.

| | |
|---|---|
| `GET /` | the page |
| `GET /classify?q=` | which retriever would take this query, and why |
| `GET /search?q=&k=` | retrieved chunks, with per-term scoring for BM25 |
| `GET /compare?q=&k=` | all three rankings over one query |
| `GET /ask?q=&k=` | cited answer |
| `GET /eval` | the full evaluation, aggregates and per-question ranks |
| `GET /chunks` | every chunk in the index, without its source |
| `GET /health` | chunk count and sample symbols from the index |
| `/docs` | Swagger |

---

## How it works

```
repo ──> tree-sitter ──> chunks ──┬──> BM25 index ────┐
                                  │                   ├──> router ──> top-k ──> Gemini ──> cited answer
                                  └──> MiniLM vectors ─┘
```

**Chunking.** tree-sitter parses each file to an AST; every function and class
becomes a chunk, tagged with its path and line range. Not fixed-size windows — a
chunk is always a complete unit.

**Routing.** A query goes to BM25 when it *types out* a symbol that exists in the
repo — the token must both match a real name and be written like code
(`snake_case`, `camelCase`, `CapWords`). Everything else goes to the embedding index.
On the 41-question set this classifies **40/40 correctly** against hand-written type
labels. The code-shape requirement is not decoration; see finding 6.

**Answering.** The top-k chunks go to Gemini with their file and line headers, under a
prompt that forbids using outside knowledge and requires a citation per claim. Because
measured Recall@5 is 0.73, roughly one query in four hands the model the wrong chunks —
so it is instructed to answer `"Not in the retrieved code."` rather than guess.

---

## Decisions worth explaining

**Decorators are inside the chunk.** A naive AST walk starts a function chunk at `def`,
which drops `@app.get("/prices/latest")` — the single most searchable line in a route
handler. Traversal yields `(definition, outer)` pairs and slices from `outer`.

**Classes are indexed as stubs**, not bodies: signature, docstring, `# fields: a, b, c`
and `# methods: x, y, z`. Methods are already chunked individually, so emitting the
full class body would index the same source twice and let one large class dominate
results. Fields are summarised rather than pasted for a reason that had to be measured
-- see finding 7.

**Module-level code is chunked too.** Constants and the comments explaining them live
outside any function. `MIN_PCT_MOVE = 2.0` and its three-line justification were
invisible to a function-only chunker — see the findings below.

**RRF uses ranks, never raw scores.** BM25 returns unbounded positives; cosine returns
-1 to 1. Adding them needs a calibration set that doesn't exist. k=60 follows Cormack
et al. (2009); rankings are fused to depth 20 and the top 5 returned, so a chunk ranked
12th by both retrievers can beat one ranked 2nd by a single retriever.

**Brute-force cosine, no vector database.** Measured over 557 chunks of FastAPI's
source: BM25 retrieval **2.3ms**, vector retrieval **60ms** — and almost all of that
60ms is the embedding model encoding the query string, not the similarity search. The
search itself is a numpy dot product against a normalised matrix. A vector database
would optimise the part that was never the bottleneck. It gets added when a
measurement demands one. The real first-query cost is neither: the embedding model is
loaded lazily, so the first paraphrase query in a fresh process waits several seconds
for weights and every one after it is in the tens of milliseconds.

**Gemini over plain REST.** `httpx` was already a dependency and the REST contract is
stable, which avoids taking on an SDK whose import path has changed repeatedly.

---

## What the measurements taught

**1. Fusion can be worse than its inputs.** With BM25 at 0.07 MRR on paraphrase
queries and the vector index at 0.41, RRF produced 0.20. The arithmetic: BM25's rank-1
chunk scores 1/61 = 0.01639, the vector index's rank-2 chunk scores 1/62 = 0.01613. A
wrong answer the weak retriever ranked first outranks the right answer the strong one
ranked second.

**2. An eval set can smuggle in your assumptions.** The first 30 questions all avoided
code vocabulary — realistic for a newcomer, but precisely BM25's weakness. BM25 scored
0.17 and looked useless. Adding 10 identifier-style questions and reporting per type
revealed it scores 1.00 on the queries it is actually for.

**3. BM25 ranked definitions below their own callers.** `"where is check_asset
defined"` returned three tests before the implementation: the tests call the function
so they contain the token, and they are far shorter, so length normalisation rewards
them. A `NAME_BOOST` multiplier when a query token exactly matches a chunk's name fixed
it — identifier MRR 0.73 → 1.00.

**4. The boost value came from a sweep, not taste.** Measured at 1, 3, 5 and 8:
identifier MRR reaches 1.00 at 3 and is *identical* at 5 and 8. It saturates once the
definition clears its callers, so 3 is the smallest value that works. Paraphrase scores
are unchanged at every setting, because the boost only fires on queries containing real
symbol names.

**5. The router broke on the second repository I tried.** It scored 40/40 on the repo
it was built against, where functions are called `store_prices` and `check_asset`.
Pointed at FastAPI it collapsed — that framework defines methods named `get`, `post`,
`put`, `head`, `options` and `trace`, so "what happens when a request fails" matched a
symbol and was routed to BM25. The heuristic had silently assumed function names are
not ordinary English words. Fix: require the matched token to be written like code.
Re-running the original evaluation confirmed no regression — still 1.00 on identifier
queries, still 0.55 overall — so the fix is a strict improvement rather than a trade.
This is the clearest argument in the project for keeping an eval harness around: the
bug was invisible on the corpus the heuristic was designed against.

**6. Recall and answer quality are not the same thing.** The module-preamble chunks
(constants and their comments) changed paraphrase MRR by 0.01 — nothing. By the metric,
the change failed. But in `ask`, the model now retrieves both the implementation *and*
the constants, and answers "z-score >= 3.0 and move >= 2.0%" with real values instead
of naming constants whose values it cannot see. The metric measured whether the
expected chunk was retrieved. It never measured whether the answer was good.

**7. Two wrong fixes taught more than the right one.** The evaluation view lists every
question no retriever answered. Reading it, three of ten were model classes —
`PriceReading`, `Asset`, `AnomalyOut` — and the chunker indexes classes as stubs. A
class with no methods therefore reduced to `class PriceReading(Base):` and a docstring,
with the field declarations discarded and indexed nowhere else. "What is stored for a
single observation" was asking for exactly what had been deleted.

The obvious fix was to emit such classes whole. Measured, it did not work: overall MRR
did not move, R@1 rose while R@5 fell, and a question that *had* been answered —
"what gets recorded when something unusual happens" → `models.Anomaly` — stopped being
answered. Adding the real content made the chunk harder to find. An embedding is a mean
over tokens, so a two-line stub carrying a descriptive docstring is nearly pure signal,
and ten lines of `Column(Float, nullable=False)` pull the vector toward generic ORM
boilerplate.

Summarising the fields the way methods were already summarised — `# fields: symbol,
price, ts` — is what worked: **MRR 0.54 → 0.55, R@1 0.41 → 0.44, unanswered 10 → 9**,
with R@5 unchanged. Worth noting that the stated reason was still half wrong: the gain
is entirely in the embedding index, and BM25's paraphrase scores did not move at all,
though adding searchable tokens was the justification given.

Three things follow. Chunking decisions that are right for one kind of class can be
wrong for another, and only a per-question eval surfaces which. More text in a chunk
can lower its recall. And a correct prediction is not the same as a correct
explanation — the second experiment succeeded for reasons partly different from the
ones that motivated it.

---

## Limitations

- 41 questions, 10 of them identifier-style. Small. "10/10", not "100%".
- Two repositories, one language: a 45-chunk service and 557 chunks of FastAPI's
  source. Nothing here is validated at real scale.
- Nine paraphrase questions are unanswered by every retriever. The ceiling is in what
  gets indexed, not how it's ranked.
- Untested ideas for that ceiling: a code-trained embedding model; one chunk per
  top-level constant carrying its own comment block; an LLM-written summary indexed
  alongside each chunk's source.
- Python only. tree-sitter has grammars for everything else; the chunker doesn't use
  them yet.
- The index is built at startup and never invalidated — re-index by restarting.
- **Whole-repository questions are out of scope.** "What is this codebase about?" has
  no locatable answer; retrieval hands the model five chunks and it generalises from
  whatever those happened to be. This tool answers questions that have an answer
  *somewhere specific*. Summarisation is a different architecture.
- Routing accuracy depends on a repo's naming conventions. Validated on two
  codebases; a third could break it again in a way the current rule doesn't cover.

---

## Tests

```bash
pytest -q
```

Run on every push against Python 3.11 and 3.12. CI installs `pytest`, `numpy` and the
two tree-sitter packages — deliberately not `requirements.txt`. The suite touches no
network and no embedding model, so a run finishes in under a minute instead of pulling
two gigabytes of torch for code it never executes. It also pins a design decision in
place: `sentence-transformers`, `fastapi` and `httpx` are imported lazily inside the
functions that need them, so `arch.py chunks` starts instantly. Move one of those
imports to the top of the file and CI fails on `ImportError`, which is when you want
to find out.

36 tests, no network and no model. They cover the parts that can be wrong in ways a
human wouldn't notice: tokenisation, chunk boundaries, fusion arithmetic, metric
definitions, routing, and the evaluation harness itself. The embedding model is a
dependency, not this project's code — what's tested is how its output is combined.

The test that earns its place most is `test_explain_sums_to_the_score_it_claims_to_
explain`. The UI shows a per-term breakdown of every BM25 score, computed by a
different code path from the scoring loop for performance reasons. An explanation that
quietly stops matching the thing it explains is worse than no explanation, so the test
walks every chunk over several queries and asserts the itemised contributions still
multiply out to the score actually used for ranking.

---

## Stack

Python 3.12 · tree-sitter · sentence-transformers (all-MiniLM-L6-v2) · numpy ·
FastAPI · Gemini · pytest
