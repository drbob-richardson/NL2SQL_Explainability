# ECIR paper plan — "When Does Structure Help Retrieve Context for LLMs?"

**Target:** ECIR 2027, Reproducibility & Generalizability track (deadline ~mid-Oct 2026; VERIFY CFP).
**Fallback / escalation:** if rejected or if we want the bigger version, strengthen SOTA comparisons for
SIGIR 2027 (~late Jan 2027). Best case: ECIR accept + a *new* result for SIGIR.

## 1. Framing (why it fits the Reproducibility & Generalizability track)
We reproduce and generalize the widely-repeated claim that **graph/structure priors help multi-hop
retrieval** (GraphRAG, HippoRAG's personalized PageRank, schema-linking for text-to-SQL) and subject it
to the discipline the claim usually skips: strong baselines, cross-domain generalization, and an
ablation of *structure vs the specific method*. This is a reproduce-generalize-analyze contribution, not
a SOTA-chase — the track's sweet spot.

## 2. Contributions / Research Questions
- **RQ1 — Does structure beat STRONG baselines, and when?** Not just cosine: a dense bi-encoder (BGE) and
  a cross-encoder reranker. Establish that structure's gain survives strong rerankers (we have a positive
  pilot: +0.067 bridge on top of a cross-encoder on HotpotQA).
- **RQ2 — Structure, or the specific (Bayesian) method?** Reproduce whether the elaborate object (subgraph
  MRF posterior / HippoRAG-style PPR) beats a one-line heuristic (personalized PageRank, shortest-path
  closure). Pilot: PageRank ≈ closure ≈ MRF across 4 datasets (diff <1pt).
- **RQ3 — The connectivity boundary (the generalization).** Structure helps iff evidence is connected:
  hurts single-hop (SQL -0.11; SciFact kNN-graph -0.59), grows with hop-depth (2Wiki +0.13→+0.21) and
  corpus correlation (BEAVER +12–14pp), destructive when the graph is misaligned. A predictive law, not a
  point result.
- **RQ4 — Constructive: topology-routed retrieval.** Route structure vs diversity by predicted query type;
  beats every fixed bias (+4.4pp, ≈oracle) and survives the cross-encoder. The actionable payoff.
- **RQ5 — Downstream.** Structure → higher text-to-SQL execution accuracy (pilot +5.7pp EX).

## 3. Datasets (SQL + RAG, both domains)
- Text-to-SQL schema linking: **BIRD** (full dev), **Spider**; downstream EX on BIRD large-schema DBs.
- Correlated enterprise SQL: **BEAVER**.
- Multi-hop RAG: **HotpotQA** (distractor), **2WikiMultiHopQA**.
- Single-hop controls: **SciFact**, single-table BIRD.

## 4. Baselines (the hardening — this is the main new work)
- Lexical: **BM25**.
- Dense bi-encoder: **BGE-small / e5** (OPEN + reproducible; move OFF closed text-embedding-3-small —
  important for the repro track; BGE infra already built for the BEIR study).
- Reranker (strong point estimate): **ms-marco MiniLM cross-encoder** (already integrated).
- SOTA structured: **HippoRAG-style personalized PageRank over an entity graph** for multi-hop RAG
  (our passage-PPR is HippoRAG-lite — frame explicitly as reproducing/generalizing it); **FK-closure**
  for SQL.
- (Optional, if time) **ColBERT** late interaction.

## 5. Methods under test (all share the unary+graph pipeline)
unary fusion · PageRank diffusion · subgraph MRF posterior · FK/entity closure · topology-routed retriever.
Metrics: recall@|gold|, nDCG@10, downstream EX; paired bootstrap CIs + significance; multiple seeds.

## 6. What we already have vs. what's new
HAVE: full pipeline (MRF/PageRank/closure), topology router, downstream EX, connectivity-boundary results
on 6 datasets, cross-encoder validation on HotpotQA, BGE encoding infra. Scripts: s1a_graphgp, s2_beaver,
s3_sql_*, s3_scifact, s4_*, s5_twowiki, downstream_ex, s_reranker, s_sensitivity, beir_encode.
NEW WORK: (a) swap dense baseline to BGE/e5 uniformly across ALL settings; (b) run every setting at full
scale with the cross-encoder as the strong reranker floor; (c) HippoRAG-style entity-graph PPR baseline
+ explicit positioning; (d) topology routing + ablations on top of strong rerankers; (e) seeds/
significance everywhere; (f) one-command reproducibility package.

## 7. Timeline (July → mid-Oct 2026; ~14 weeks, ample slack)
- W1–2: lock scope + RQ/experiment matrix; unify BGE dense + CE reranker harness across datasets.
- W3–5: re-run all settings at full scale w/ strong baselines → the connectivity-boundary master table + CIs.
- W6–7: HippoRAG-style entity-graph PPR baseline on multi-hop RAG; sharpen RQ2 (structure-not-method).
- W8–9: topology-routing method + ablations (router acc, oracle vs learned, per-type, sensitivity) on CE.
- W10–11: downstream EX at scale; robustness (α, k, embedding model).
- W12–13: write (LNCS ~12–16pp); build repro package (code + cached data + configs + one-command).
- W14: polish, internal review, submit.

## 8. Paper outline (LNCS)
1. Intro — the claim under test, the gap (weak baselines; structure-vs-method), contributions.
2. Related work — GraphRAG/HippoRAG/RAPTOR/G-Retriever, schema linking, dense/late-interaction retrieval,
   IR reproducibility culture (neural-baselines reckoning).
3. Setup — datasets, structured-retrieval pipeline, baselines, metrics, significance protocol.
4. RQ1 structure vs strong baselines · 5. RQ2 structure-not-method · 6. RQ3 connectivity boundary ·
   7. RQ4 topology routing · 8. RQ5 downstream EX.
9. Discussion, limitations, reproducibility.

## 9. Reproducibility package (a strength — lean into it for this track)
Public repo: all scripts, cached features/embeddings, seeds, configs, `make reproduce` for each table,
dataset-license notes, environment file. This is graded favorably in the repro track.

## 10. Relationship to the TAS paper (keep contributions distinct + cross-cite)
TAS = the conceptual audit/perspective + belief-vs-evidence synthesis + statistical lessons + practitioner
guide (cites this paper for structured-retrieval detail). ECIR = the technical deep-dive: the connectivity
law as a validated empirical finding + the topology-routed retriever, against strong baselines. Different
contributions, different audiences, cross-cited. Not salami.

## 11. Open decisions before we start
- Track: Reproducibility&Generalizability (recommended) vs full-paper track — confirm from CFP scope.
- Encoder: BGE-small (fast, CPU/MPS-OK) vs e5 vs a stronger BGE-large (GPU). Default BGE-small for repro.
- HippoRAG comparison depth: our own entity-graph PPR reimplementation (controlled) vs running the
  official HippoRAG (heavier). Recommend controlled reimplementation for a clean apples-to-apples ablation.
