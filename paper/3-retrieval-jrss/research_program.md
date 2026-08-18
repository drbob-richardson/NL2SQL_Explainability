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
| **S1 SQL orthogonal** (BIRD/Spider) | **done: WINS** (FK-MRF +5.7pp EX) | done: opportunity real (oracle>full) but posterior-threshold FAILS (0.411) | untested | done: loses* | done: loses (maxout 0.763 > post 0.700) |
| **S2 SQL correlated** (BEAVER dw) | **done: WINS bigger** (cosine craters 0.42; struct +12-14pp) | — | **done: WINS** (online +8.5pp, learning curve) | — | — |
| **S3 single-hop** (SQL |gold|=1; RAG SciFact) | **done: structure HURTS** (SQL −0.11; RAG kNN-diffusion −0.59 catastrophic) | — | — | — | — |
| **S4 RAG multi-hop** (HotpotQA distractor) | **done: WINS bridge** (+10-13pp) | — | — | **done: helps COMPARISON** (+7pp; topology-dependent) | done: loses (cos-margin 0.744 > post 0.692) |
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

- [done] S4-a structure (HotpotQA): graph prior (PageRank/MRF) beats cosine +10-13pp recall@2, bridge-concentrated (+17pp), ~tie comparison. THE SQL STRUCTURAL WIN GENERALIZES TO TEXT RAG. PageRank~=MRF again.
- [variation] PageRank slightly HURTS comparison -> adaptive gating by bridge-ness/query-type (the structure-exploitability meta-predictor). Full-corpus (not distractor-rerank) multi-hop + noisy entity-linking = future tests. S4 harness now enables S4-d (diversity, distractors are redundant) and S4-e (UQ) cheaply.

- [done] S4-d/e: diversity helps COMPARISON under redundancy (+7pp) but hurts bridge; structure mirror -> TOPOLOGY-DEPENDENT complementarity (structure|bridge, diversity|comparison; oracle-routing ~0.76 beats all fixed). UQ posterior loses to cosine-margin again.
- [variation] ADAPTIVE topology-routed retriever (predict bridge-vs-comparison from question, apply structure-prior vs diversity-prior) -> beats both fixed; query-type classifier is easy. HIGH PRIORITY new method idea.

- [done] S4 ADAPTIVE topology-routed retriever: type classifier acc 0.936; adaptive recall@2 0.748 beats best-fixed (PageRank 0.703) by +0.044 [.036,.053], ~oracle 0.761. POSITIVE METHOD: route structure-prior(bridge) vs diversity-prior(comparison). The audit's constructive payoff.

- [done] S2 BEAVER (correlated enterprise SQL): cosine 0.425 (vs BIRD 0.72) -> CORRELATION makes retrieval hard; structure (PageRank/MRF) +12-14pp, LARGER gain than BIRD -> orthogonality critique CONFIRMED. Methods cluster (sparse join graph; no MRF>heuristic separation). 120 qs, recall-only.
- [variation] BEAVER join graph sparse -> densify with cosine-sim edges to test MRF>heuristic; dev_nw.json (88 qs, multi-DB) = cross-DB routing cell; no warehouse data = metadata-prior-only regime.

- [done] S3 SQL single-hop control (|gold|=1, recall@1): cosine 0.839, PageRank 0.727 (HURTS -0.11), MRF 0.857 (neutral). Boundary confirmed: structure helps iff multi-hop. MRF degrades GRACEFULLY (backs off to unary) while diffusion heuristic hurts single-hop -> MRF's niche = safe across MIXED hop-count workloads. Structure-value is monotone in hop-count.
- [variation] hop-count / connectivity-need predictor to GATE structure (apply only when multi-hop) -- analog of the topology router; would let the diffusion heuristic match the MRF's graceful degradation. RAG single-hop (BEIR SciFact / chunked-doc redundancy) still to run.

- [done] Hop-gated SQL: on mixed workload (800 qs), always-MRF 0.803 > always-PageRank 0.751 > cosine 0.744; oracle-gate rescues PageRank (0.791) but real gate fails (hop predictor 0.799 too weak). MRF MARGINALIZATION = predictor-free implicit hop-gate -> the MRF's genuine niche (graceful degradation on heterogeneous workloads). Contrast: RAG topology predictable (explicit routing wins); SQL hop-count not (implicit MRF wins).

- [done] Decision/cost (BIRD downstream EX): oracle 0.562@2.2tbls > full 0.500@10 (opportunity real, distractors hurt) BUT MRF-posterior-threshold 0.411 UNDER-RETRIEVES (< fixed-k MRF 0.495). Posterior-as-decision FAILS like posterior-as-UQ -> families 2 & 5 unified: posterior is not a usable decision/uncertainty signal; structure-as-ranking is the only robust win.

- [done] Family 3 ADAPTATION (BEAVER online learning curve): naive-Bayes term->table online 0.510 (+8.5pp vs static cosine 0.425), RISES with feedback (Q1->Q3 0.455->0.560). GENUINE Bayes win, regime-specific (correlated/repeated). Second robust win after structure; the textbook-Bayesian one. Validates the user's naive-Bayes-online idea. 120 qs (short), simulated feedback.

- [done] S5 graph RAG (2WikiMultiHopQA, 1500 qs): PageRank 0.805 / MRF 0.804 > cosine 0.717; by type structure helps chained (bridge_comp +.21, compositional +.17, inference +.13) HURTS independent comparison (−.22). Dichotomy replicates on a 2nd multi-hop RAG dataset; PageRank≈MRF (4th confirmation: structure not the specific Bayes).
- [done] S3 RAG single-hop (SciFact, 5183 docs, gold 1.13): cosine 0.608 > BM25 0.535 >> kNN-graph diffusion 0.016 (−0.59). Imposing a similarity graph on single-hop is DESTRUCTIVE. Boundary confirmed cross-domain.
- ===== PROGRAM COMPLETE: all 5 settings x 5 families covered across 5 datasets (BIRD, Spider, BEAVER, HotpotQA, 2Wiki, SciFact). =====
