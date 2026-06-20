# Retrieval direction — landscape scan + first probe

Quick scoping of "can a Bayesian engine choose better than cosine/hybrid for retrieval / SQL
schema-DB routing." Verified literature scan (5 sub-agents) + a one-afternoon empirical probe.

## Landscape (verified citations)
- **Probabilistic IR is already "Bayesian retrieval."** BM25 = probabilistic relevance model
  (Robertson–Spärck Jones 1976; Robertson–Zaragoza 2009); Dirichlet-smoothed LM-IR (Zhai–Lafferty
  2001); retrieval-as-Bayes-risk (Lafferty–Zhai 2001); Pólya-urn doc model (Cummins+ 2015,
  arXiv:1502.00804). *Don't reinvent this.*
- **Retriever fusion is SATURATED.** CombSUM/MNZ (1994), RRF (Cormack 2009), convex-combination
  (Bruch 2022, arXiv:2210.11934). **Per-query adaptive fusion already exists and is active**:
  Query-Adaptive Late Fusion (Zheng 2015), DAT (arXiv:2503.23013, 2025), MoR mixture-of-retrievers
  (EMNLP 2025, arXiv:2506.15862). The "better fusion" angle is largely taken.
- **Set-level coverage selection exists**: SETR (ACL 2025, arXiv:2507.06838), DPP multi-answer
  (COLING 2022, arXiv:2211.16029), submodular coverage (Lin–Bilmes 2011). **But Bayesian
  experimental-design / EIG for set selection under budget = NOT FOUND (open framing).**
- **Conformal/calibrated retrieval**: busy on RAG *generation* (C-RAG arXiv:2402.03181, TRAQ NAACL
  2024); thinner on calibrating retriever scores; **near-empty for text-to-SQL schema/table
  retrieval — only RTS/BPP (arXiv:2501.10858) applies conformal schema-linking.** Least crowded,
  best fit for NL2SQL.
- **Triple-intersection** (Bayesian posterior over relevance + calibrated uncertainty + per-query
  multi-retriever fusion) = not found as one framework; genuine but narrow niche.

## Probe — cross-DB table retrieval (Spider-multi: 81 tables / 20 DBs / 355 Qs, mean 1.9 gold/q)
`scripts/retrieval_probe.py` (text-embedding-3-small; cosine vs BM25 vs RRF vs learned fusion).

| method | recall@\|gold\| | recall@5 | allgold@5 | DBroute@1 |
|---|---|---|---|---|
| cosine | **0.778** | 0.964 | 0.924 | **0.946** |
| bm25 | 0.516 | 0.718 | 0.572 | 0.721 |
| RRF | 0.646 | 0.821 | 0.715 | 0.870 |
| fusion | 0.734 | 0.964 | 0.924 | 0.901 |

**Cosine wins; fusion does NOT beat it** (and BM25/RRF hurt). But the corpus is tiny and easy
(cosine recall@5 0.96, DB-routing 0.95) — there's almost no headroom. The hypothesis ("Bayes beats
cosine") can only have room in the **large/messy-schema regime** (Spider 2.0 / enterprise, 100s–1000s
of columns) where dense cosine degrades and structural signals matter. We don't have that data loaded.

## Verdict
Same pattern as the BNP arc: appealing simple idea meets a strong baseline + crowded space. General
"better fusion via Bayes" is crowded and cosine is hard to beat in clean settings. The only genuine
openings: (1) **conformal/guaranteed coverage for text-to-SQL schema retrieval** (near-empty), (2)
**EIG/Bayesian-experimental-design set selection under budget** (open framing) — and both need the
**large-schema setting** to demonstrate value. Decision: pursue only with Spider-2.0-scale data;
otherwise the accessible benchmarks are too easy to show a win.

## Addendum — text-to-SQL schema/DB retrieval landscape (5th scan agent)
- **Large-schema benchmarks exist and are the right testbed:** Spider 2.0 (ICLR 2025, >3000 cols,
  enterprise, frontier ~17–21% EX; arXiv:2411.07763), BEAVER (enterprise logs, 812 tables, names
  "multi-table retrieval" as a subtask; arXiv:2409.02038), BIRD (95 DBs; arXiv:2305.03111).
- **Schema-selection / table-retrieval is heavily worked:** CRUSH4SQL hallucinate-then-retrieve
  (arXiv:2311.01173), RESDSQL schema ranking (2302.05965), CHESS schema selector/pruning
  (2405.16755), MURRE multi-hop table retrieval (2402.10666), ARM (2501.18539), **LinkAlign — the
  first to frame "database retrieval: select the target DB from a large pool" = explicit multi-DB
  routing** (arXiv:2503.18596).
- **Calibrated/UQ table selection is occupied by ONE method:** RTS / Adaptive Abstention (conformal
  prediction on hidden layers for schema linking + abstention; arXiv:2501.10858). Query-level
  calibration exists (sub-clause 2505.23804; node-level 2511.13984; TrustSQL 2403.15879).
- **CONFIRMED GAP:** no *Bayesian* method for table selection/DB routing, and **nothing applying
  calibrated UQ at the multi-database routing layer.** Multi-DB routing itself is emerging, not yet a
  standardized task.

## Refined verdict
The **space** (large-schema SQL retrieval) is crowded with strong methods; the **niche** —
*Bayesian/calibrated UQ + a decision layer (route / abstain / ask) for table-selection AND multi-DB
routing* — is genuinely open (only conformal RTS is adjacent) and directly reuses our Paper-1
LTT/conformal machinery. Catch (from the probe): value only appears in the **large-schema regime**
(Spider 2.0-Lite / BEAVER) where cosine degrades; easy benchmarks (Spider-multi) are saturated by
cosine. Viability gate = stand up Spider 2.0-Lite or BEAVER and re-run the probe there.

## Idea-1 first test — structured (FK-graph) retrieval vs cosine (`scripts/bridge_probe.py`)
BIRD, 639 multi-table questions. Connector = articulation point of the gold FK-subgraph.

- Blind spot is MILD on BIRD: connector gold tables rank only slightly worse than leaves (cosine
  rank 2.90 vs 2.61); connectors are 11% of misses = their 11% base rate (NOT over-represented).
- BUT structure beats cosine at matched budget, and the lift GROWS with join complexity:

| query size | n | cosine@\|gold\| | graph-closure | cosine@matched | structure lift |
|---|---|---|---|---|---|
| ≥2 | 639 | 0.720 | 0.780 | 0.746 | +0.035 |
| ≥3 | 156 | 0.673 | 0.658 | 0.606 | +0.052 |
| ≥4 | 30 | 0.678 | 0.768 | 0.652 | **+0.117** |

**Verdict: weak-positive on BIRD, but the right trend.** First method in the exploration to beat the
strong cosine baseline at matched budget, with lift monotonically increasing in #gold-tables — the
predicted mechanism (more hops → FK graph recovers what cosine misses). BIRD dilutes it (mostly 2–3
table queries); the regime where it dominates is large multi-hop schemas (Spider 2.0 / BEAVER).
This is the seed worth building: replace the Steiner heuristic with a proper Bayesian posterior over
connected subgraphs, and validate where multi-hop is the norm.
