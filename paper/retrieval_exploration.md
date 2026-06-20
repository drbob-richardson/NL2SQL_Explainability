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
