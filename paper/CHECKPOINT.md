# Project Checkpoint — Goals, Hypotheses, Results, Theory, Directions

**Date:** 2026-06-20 · **Scope:** the full arc from "BNP priors on query graphs for NL2SQL
uncertainty" through the correctness paper, the ambiguity/decision exploration, and the retrieval
pivot. Detailed records: `paper/tex/paper1_correctness.tex` (Paper 1), `paper2_bnp_decision_exploration.md`,
`paper2_options_and_roadmap.md`, `paper2_theorem1_sketch.md`, `retrieval_exploration.md`. Total API
spend across the whole exploration ≈ $6 (all cached/reproducible).

---

## 1. Goals

- **Original:** place **Bayesian-nonparametric priors on query graphs** to produce a **posterior over
  SQL query graphs**, giving calibrated **uncertainty quantification for text-to-SQL** — a
  methodological/statistical contribution, not just an applied win.
- **Evolved (as evidence came in):**
  1. Ship a solid empirical paper on **what predicts correctness** in text-to-SQL (Paper 1). ✔ done.
  2. Find where BNP / structured Bayes is *load-bearing* rather than decorative.
  3. After BNP-for-correctness and ambiguity failed, pivot to **retrieval**: can structured/Bayesian
     methods choose context (RAG) or tables/DBs (text-to-SQL) better than cosine/hybrid?
- **Standing aim:** a defensible contribution that plays to a Bayesian-statistics strength and
  iterates fast.

---

## 2. Hypotheses tested (with verdicts)

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | Black-box agreement signals predict SQL correctness well | **FALSE** — they plateau (~0.61–0.68 AUROC); a reasoning verifier is needed (Paper 1) |
| H2 | A BNP graph-posterior improves correctness UQ | **FALSE** — graph features 0.58–0.65, no marginal value over the verifier |
| H3 | The gold-query motif space is open-world (BNP-justified) | **PARTLY TRUE** — real power law at the semantic level (d≈0.49, ~15% discovery) after deflating syntactic/granularity inflation |
| H4 | Ambiguity can be detected from the generator's posterior | **FALSE** — sampling collapses (coverage-both 1%), divergence AUROC 0.475 |
| H5 | The model can *discover* interpretations even if it can't realize them | **TRUE** — interpretation-first recall ≫ SQL recall; realization is the bottleneck |
| H6 | A Bayes execute/clarify/abstain layer has value | **TRUE in principle** (oracle sim ~5× loss cut) but blocked by realization, and the PYP-reserve was no better than baselines for the actual decision |
| H7 | Bayesian fusion beats cosine/hybrid for retrieval ranking | **FALSE in easy regimes** — cosine wins; fusion saturated |
| H8 | A graph-structured posterior (FK prior) beats cosine for *multi-hop* table retrieval | **TRUE** — the project's main positive result |
| H9 | The structured win translates to end-task SQL accuracy | **TRUE** — +5.7pp EX, significant |
| H10 | The subgraph posterior is a good abstention/UQ signal | **FALSE** — a trivial cosine heuristic beats it |

---

## 3. Results

### 3A. Paper 1 — "What Predicts Correctness in Text-to-SQL?" (SHIPPED-READY)
- Black-box signals (string/structural/execution self-consistency, schema-relevance, executability)
  plateau **0.61–0.68 AUROC**; white-box logprob doesn't beat them.
- **Verification breaks the ceiling:** LLM judge 0.72–0.78; two-provider ensemble **0.82 AUROC,
  ECE 0.03**; supports abstention frontiers self-consistency can't.
- Trained verifiers work in-domain (~0.77–0.79) but **don't transfer** across schemas (~0.66); only a
  large frozen reasoning judge transfers.
- Errors concentrate in **computation/composition** (arithmetic, nesting, CASE, GROUP BY), not schema
  linking — explains why schema-relevance fails and reasoning verification works.
- Replicated across two benchmarks (BIRD + Spider). **In Overleaf, TMLR format, ~12pp.**

### 3B. BNP-for-correctness — DEAD
- Graph-posterior features (skeleton entropy, node-min posterior, etc.) = 0.58–0.65 AUROC; combined
  with the verifier Δ = −0.028 [−0.054, −0.004]. No value over the verifier.

### 3C. Open-world motif tail (PYP) — PARTIAL / ALIVE
- Skeleton level: d=0.16, ~52% discovery — but **degenerate fit** (θ≫N, near-linear accumulation =
  curation/syntactic inflation).
- **Semantic "canon" level (the credible one):** clean power law **d=0.49**, ~15% discovery,
  accumulation exponent 0.68 (R²=0.997); does *not* collapse to closed clause-set. Survives
  denotational quotienting. **Open co-gate:** curation inflation (needs a real repetitive workload).

### 3D. Ambiguity / decision-making (AMBROSIA) — cheap version DEAD
- AMBROSIA gold interpretations: **100% material**; localization type-dependent (vague 83% single-slot,
  scope/attachment 0% — structural).
- **Generator collapses:** sampling surfaces both readings 1%; explicit elicitation 4% (mini) / 6%
  (gpt-4o); official metric recall 0.23 / all_found 1%.
- **Discovery ≫ realization:** interpretation-first NL recall ~0.75 (judge-inflated; vague 72% solid)
  vs SQL 0.23; two-stage discover→realize lifts all_found 1%→7%. Realization is the bottleneck
  (known-engineering: few-shot + execution repair → AMBROSIA's ~30–65%).
- **Decision sim (Exp 5):** oracle execute/clarify/abstain cuts loss 0.296→0.053 (c=0.1, perfect
  realization), beating both baselines; collapsed posterior captures ~none; at realistic realization
  (r≈0.3) even the oracle abstains.
- **Theorem-1:** naive "PYP discovery = miss prob" is real (Pitman's rule) but exchangeability is
  empirically false → reserve is a lower bound. Robust form borrows conformal/LTT validity (not
  novel); the **efficiency claim FAILED** (PYP-reserve abstention AUROC 0.557 ≈ chance; over-elicitation
  kills precision).

### 3E. Retrieval — the WORKING direction
- **Landscape:** general fusion saturated (per-query adaptive already exists: DAT, MoR); BM25/Dirichlet-
  LM already "Bayesian"; conformal-for-SQL-retrieval near-empty (only RTS 2501.10858); Bayesian +
  multi-DB-routing = confirmed gap.
- **Easy-regime probe (Spider):** cosine wins; learned fusion doesn't beat it.
- **Phase 1 — structured FK-graph posterior (Ising/MRF, exact inference):** beats cosine AND learned
  fusion for table retrieval. Hardened (held-out β, rich features incl. value-match + column-cosine):
  recall@|gold| **MRF 0.805 / 0.822 / 0.787** (≥2/≥3/≥4 tables) vs cosine 0.720/0.673/0.678 vs unary
  fusion 0.782/0.761/0.670. All bootstrap CIs exclude 0; **structure gain grows with join complexity**.
- **Conditional:** gains concentrate on FK-rich schemas (financial +0.20, formula_1 +0.19), regress on
  tiny already-solved ones (toxicology −0.09). Adaptive-β-by-table-count fix FAILED; keep fixed β.
- **Phase 3 (UQ on the posterior) — FAILED:** cosine max-out predicts retrieval completeness better
  (AUROC 0.763) than the posterior (0.700), and the posterior doesn't add.
- **Downstream EX (the validation that matters) — POSITIVE:** prune to top-5 then generate; EX full
  0.500, cosine 0.438, **MRF 0.495**, oracle 0.562. **EX(MRF)−EX(cosine) = +0.057 [+0.021,+0.094]**.
  MRF-pruning ≈ full-schema (prune hard with ~no loss) while cosine-pruning costs 6pp; oracle > full
  (+6pp) ⇒ distractors hurt generation, precise retrieval *exceeds* full schema. Strongest on the
  hardest schema (formula_1 +13.6pp, beats full). Anomaly: financial −3.8pp.

---

## 4. Theory — the unifying pattern

Across **three independent directions** the same structure held:

> **Structured/Bayesian objects help RANKING / recall / generation, but LOSE to simple baselines on
> the UNCERTAINTY / decision layer.**

- Correctness: graph/agreement plateau; a **reasoning verifier** wins the UQ.
- Ambiguity: discovery is tractable, but the PYP-**reserve / divergence** lose for detection; **cosine/
  heuristics** win.
- Retrieval: a **graph posterior** wins ranking (Phase 1) and end-task EX, but **cosine max-out** wins
  the abstention/UQ (Phase 3).

Two more durable findings:
- **Structure helps in proportion to available structure** — the FK-graph prior's gain scales with
  schema FK-density and query hop-count; it's neutral/harmful on tiny schemas.
- **The LLM's output distribution is a biased *proposal*, not a posterior** — it collapses (0.95
  confidence on ambiguous questions), so UQ cannot be read off it; it must come from a separate
  reasoner (verifier) or external signal.
- **Distractors hurt generation** — oracle table-sets beat full-schema by +6pp EX, so *precise*
  retrieval is independently valuable, not just a recall proxy.

---

## 5. Assets

- **Paper 1**: Overleaf (git-bridge synced), TMLR-formatted, ~12pp, compiles clean.
- **Code (`scripts/`):** correctness pipeline; `bnp_probes.py`, `ambiguity_probe.py`, `ambrosia_*`
  (probe/generate/coverage/elicit/interp/realize/rescore/decision_sim/uq_coverage), `bnp_equivclass.py`,
  `bridge_probe.py`, `bayes_subgraph.py`/`_v2.py`, `phase1_validate.py`/`_adaptive.py`,
  `phase3_selective.py`, `retrieval_probe.py`, `downstream_ex.py`. All safe-by-default on API, cached.
- **Data:** BIRD (8 local DBs) + Spider + AMBROSIA (gitignored, re-downloadable; AMBROSIA pw AM8R0S1A);
  embeddings + sample caches gitignored.
- **Docs:** this checkpoint + the four paper2_* / retrieval_exploration write-ups.

---

## 6. Possible directions

**Most promising (the working result):**
1. **Applied retrieval paper — "structured graph-prior table retrieval for text-to-SQL."** A FK-graph
   posterior improves multi-hop table recall (+8–15pp) AND end-task EX (+5.7pp), lets you prune large
   schemas to ~full-schema accuracy, and the gain scales with schema complexity. Frame via **"when does
   structure help retrieval"** (the conditional finding) + the downstream EX. Differentiator vs the
   crowded schema-linking field (CHESS/CRUSH4SQL/RESDSQL/LinkAlign) = the explicit Bayesian subgraph
   posterior + the complexity-scaling characterization + the oracle>full "distractors hurt" result.
2. **Large-schema validation (headline-strengthener):** Spider 2.0-Lite / BEAVER, where every schema is
   rich → the win should be uniform and large (BIRD's ≥4-table tail becomes the norm). BEAVER is
   SQLite-friendlier; Spider 2.0 uses cloud warehouses (real setup cost). This is the main open
   validation and the biggest credibility lever.

**Supporting / smaller:**
3. **Complete the practical system:** structured retrieval (Phase 1) + the simple cosine-max-out
   risk-controlled abstention (the Phase-3 baseline that worked) → an honest retrieve-or-abstain
   schema-linker with an LTT guarantee.
4. **Ship Paper 1 to TMLR** (independent; essentially done — verify the 4 approximate agentic
   citations, optional seed-variance on trained numbers).
5. **Open-world novelty as a side result:** the canon-level PYP power law (d=0.49) — needs a real
   workload to beat the curation-inflation critique; lower priority.

**Honest non-starters (do not revisit):** BNP-for-correctness; ambiguity detection from the generator
posterior; the PYP-reserve as an abstention signal; "better fusion via Bayes" on easy/clean benchmarks.

---

## 7. Overall status

- **One shipped paper** (Paper 1, strong, TMLR-ready).
- **One genuine new positive result** (structured FK-graph retrieval: better table recall *and* better
  end-task SQL accuracy, validated with CIs and downstream EX) — an *applied* contribution, modest but
  real, in a crowded field, pending large-schema validation.
- **The original BNP/UQ methodological contribution did not materialize** in any of three settings; the
  consistent reason (Bayes/structure helps ranking, not UQ) is itself a clear, defensible position.
- **Recommended next move:** large-schema validation of the retrieval result (direction 2), then write
  it up as an applied paper (direction 1); ship Paper 1 in parallel.
