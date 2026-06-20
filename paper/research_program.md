# Research Program — "Can Bayes Help IR for LLMs?" (exhaustive plan)

A living roadmap for the multi-week exhaustive study feeding `tas_bayes_ir.md` (TAS article) and the
CS/JASA method track. Goal: test the Bayes-helps-via-structure-not-uncertainty thesis across *settings*
and *Bayes angles*, especially the regimes our BIRD/Spider evidence missed.

## The central methodological insight (drives the whole program)

BIRD/Spider databases are **near-orthogonal** (distinct sectors, disjoint vocab, few near-duplicate
tables). That is the regime *least* favorable to Bayesian structure: no redundancy for diversity/DPP to
exploit, easy cross-domain routing, mild distractor confusability. Realistic corpora are the opposite —
a single company's many overlapping tables; one document chunked into similar adjacent sections. So our
"diversity loses / UQ loses / routing easy" findings are **scoped to orthogonal corpora and may flip
under correlation.** Testing correlated + multi-hop settings is where the Bayesian-structure story has
its best (and untested) shot.

## The matrix (settings × Bayes angles)

Angles: (a) calibrated relevance/UQ, (b) signal fusion, (c) structured/graph joint selection,
(d) set-level diversity/DPP, (e) adaptive/online updating.

| setting | a UQ | b fusion | c structured | d diversity | e adaptive |
|---|---|---|---|---|---|
| **S1 SQL orthogonal** (BIRD/Spider) | done: loses (cosine-maxout 0.763 > post 0.700) | done: loses on clean text | **done: WINS** (FK-MRF +5.7pp EX) | done: loses* | untested |
| **S2 SQL correlated** (BEAVER/Spider 2.0) | — | — | — | — | — |
| **S3 RAG single-hop** (chunked Wiki/docs) | — | — | — | — | — |
| **S4 RAG multi-hop** (HotpotQA/2Wiki/MuSiQue) | — | — | — | — | — |
| **S5 Graph RAG** (entity/KG graph) | — | — | — | — | — |

\* likely an **orthogonality artifact** — re-test under correlation (S2/S3).

## Settings to add (datasets + setup notes)

- **S2 SQL correlated — BEAVER** (arXiv:2409.02038; enterprise warehouse, ~812 tables, single-company
  correlation; partially public, SQLite-friendlier than Spider 2.0). Doubles as large-schema validation
  (JASA gate) AND the correlated-SQL regime. Spider 2.0-Lite as a harder alternative.
- **S3 RAG single-hop** — Natural Questions / TriviaQA over chunked Wikipedia, or a long-document QA
  set; chunk to induce **adjacent-chunk redundancy** (the realistic correlated-chunk case).
- **S4 RAG multi-hop** — **HotpotQA** (has built-in distractor passages → tests structure AND
  distractor-sensitivity), 2WikiMultiHopQA, MuSiQue. The direct RAG analog of multi-hop SQL: retrieve a
  *connected* passage set; the entity/passage graph is the analog of the FK graph.
- **S5 Graph RAG** — entity-graph or KG-augmented retrieval (GraphRAG-style); passage graph from
  shared entities/links as the structural prior.

## Hypotheses for the high-value open cells

- **S4-c (structured multi-hop RAG):** a passage/entity-graph prior recovers bridge passages cosine
  misses → higher answer-passage recall and downstream answer accuracy. *If true, the SQL structural
  win generalizes — the program's biggest result.*
- **S3-d / S4-d (diversity under redundancy):** DPP/repulsion finally helps when chunks are redundant
  (unlike orthogonal tables) → "it depends on corpus correlation."
- **S2 (correlated SQL):** structured win is *larger* and the MRF pulls ahead of the FK-closure
  heuristic (dense/correlated graphs make blind closure over-include → selectivity matters).
- **a UQ across settings:** posterior UQ keeps losing to simple signals (proposal-not-posterior) —
  expected to replicate; would solidify lesson #2.
- **e adaptive:** structural prior + online posterior updating from feedback beats fixed cosine as
  feedback accrues — the one genuinely-load-bearing-Bayes route.

## Prioritized queue

- **P0:** S4-c multi-hop RAG structural test (generalization — highest value); S3-d/S4-d diversity
  under redundancy (tests the orthogonality-artifact claim).
- **P1:** S2 BEAVER (correlated + large-schema, JASA gate); S3 vanilla-RAG baseline battery.
- **P2:** S5 graph RAG; e adaptive/online Bayesian retriever.

## Already done (S1 row) — see `retrieval_exploration.md`
UQ (cosine-maxout beats posterior), fusion (cosine wins clean), structured (FK-MRF wins, heuristic
captures most, +5.7pp downstream EX), diversity (cosine-coupling no help — orthogonality-scoped),
cosine≠FK structure, conditional on schema richness.

## Honesty / scoping principles
- One setting at a time, fully (front-load the harness, then run the 5-angle battery cheaply).
- Always include the trivial baseline (cosine, RRF, MMR, shortest-path closure) before claiming a model
  is needed.
- Report negatives — each cell is a row in the master table whether positive or not.
- Mark every claim's scope (which setting it holds in); resist over-generalizing from one regime.
- Safe-by-default on API; cache everything; keep costs logged.

## Cadence (multi-week)
Week-by-week: stand up one setting's harness, run the battery, log results into the matrix + master
table, update `tas_bayes_ir.md`. Revisit the thesis after each row; the "barely and sometimes" answer
may sharpen into "it depends on corpus correlation and hop-count," which is a stronger TAS finding.
