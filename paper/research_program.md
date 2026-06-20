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

## The Bayes-angle FAMILIES (consolidated from a long method menu)

A wide method list (PRF, BMA, active retrieval, stopping rules, multi-fidelity, graph-GP, hierarchical
priors, DPP, metadata priors, negative-evidence, latent intent, causal, BED, ...) collapses into FIVE
families. The thesis question is *which family lets Bayes beat the trivial baseline*.

1. **Structure** — generalize the one win. Methods: FK-MRF (done); **graph-GP / diffusion prior**
   (ADD; brings the PageRank/diffusion baseline = the "is it just diffusion?" check); metadata/authority
   priors (variant, for real corpora). Trivial baselines: shortest-path closure, PageRank.
2. **Decision & cost under budget** — posterior *allocates/decides*, not calibrates (sidesteps the UQ
   failure). Methods: **context-window utility** (ADD; retrieval as E[utility]−λ·tokens; oracle>full
   already motivates it); **cost-aware sequential retrieval** (ADD; active scoring + stopping/VOI +
   multi-fidelity as one cell); coverage/credible sets (skeptical variant — posterior-completeness
   already failed). Trivial baselines: top-k, fixed budget, score-gap stopping.
3. **Adaptation** — genuinely Bayesian; load-bearing only under correlation/repeated workloads.
   Methods: **hierarchical priors across DB/user/corpus** (ADD); online feedback updating; BMA-over-
   retrievers; PRF/query-model updating (variant — don't claim novelty over RM3/Indri). Trivial
   baselines: fixed retriever, fine-tune-on-feedback.
4. **Diversity / precision under redundancy** — the orthogonality-artifact retest family. Methods:
   **DPP / repulsion** (retest under redundancy, S3/S4); **negative-evidence / facet-completeness**
   (ADD; models absence/contradiction, not just positive relevance). Trivial baseline: MMR.
5. **Uncertainty / calibration** — the family that keeps losing. Methods: posterior UQ (done, loses);
   calibration splits doc/set/answer-level (variant). Trivial baseline: cosine max-out / verifier.

**Predicted outcome (to test):** families 1–2 deliver; 3–4 deliver only under correlation; 5 loses.
That prediction *is* the TAS spine.

## The matrix (settings × the 5 families)

Columns below = the five families (a→structure, b→decision/cost, c→adaptation, d→diversity, e→UQ).
(Earlier "fusion" folds into adaptation/structure; earlier "structured" = family a.)

| setting | a structure | b decision/cost | c adaptation | d diversity | e UQ |
|---|---|---|---|---|---|
| **S1 SQL orthogonal** (BIRD/Spider) | **done: WINS** (FK-MRF +5.7pp EX) | untested (oracle>full motivates) | untested | done: loses* | done: loses (maxout 0.763 > post 0.700) |
| **S2 SQL correlated** (BEAVER/Spider 2.0) | — | — | — | — | — |
| **S3 RAG single-hop** (chunked Wiki/docs) | — | — | — | — | — |
| **S4 RAG multi-hop** (HotpotQA/2Wiki/MuSiQue) | — | — | — | — | — |
| **S5 Graph RAG** (entity/KG graph) | — | — | — | — | — |

\* likely an **orthogonality artifact** — re-test under correlation (S2/S3).

## Future directions (deferred — interesting but premature / poor benchmark fit / overlapping)
- **Causal / intervention-aware retrieval** (Bayesian-network/causal-graph priors for "what caused X"):
  genuinely different, but too big and no clean dataset yet.
- **BNP intent discovery** (CRP/PYP over query intents): we found BNP-intent is for corpus
  diagnostics/novelty, not retrieval wins — keep as workload characterization, not a core cell.
- **Bayesian experimental design for query reformulation** (pick the rewrite maximizing EIG): strong
  concept, overlaps active retrieval; revisit if the decision/cost family pays off.
- **Latent-intent mixtures / subquestion coverage**: overlaps multi-hop structured retrieval; test as a
  variant there, not standalone.
- **Coverage / credible context sets**: posterior-as-completeness already failed (Phase 3); keep only a
  single skeptical conformal test under family b.

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

## Noted variations (mark-and-try as they arise)
- Diffusion/MRF on the COSINE-similarity graph vs the FK graph (test smoothing on the "wrong"
  structure; expect failure → confirms cosine≠FK at the diffusion level). [from S1-a]
- PageRank seed = raw cosine vs learned unary (does the learned unary matter, or does diffusion from
  cosine suffice?). [from S1-a]
- unary + PageRank-score + MRF-marginal as features in a small meta-ranker (does diffusion add as a
  feature beyond being a ranker?). [from S1-a]
- "Structure-exploitability" meta-predictor: per-DB/query, predict whether structure will help from
  FK-density × hop-count × distractor-count → adaptively gate the structural prior. [from per-DB + S1-a]
- Restart rate / coupling strength as a function of unary confidence (strong seeds → less diffusion).

## Progress log (executing the program)
- [done] S1-a structure: graph-GP/diffusion. PageRank≈MRF≫cosine; Laplacian-GP over-smooths.
  Verdict: structural win robust to method; MRF not uniquely needed (2nd simple baseline matches it).
