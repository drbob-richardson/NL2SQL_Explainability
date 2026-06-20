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

## Phase-1 method WORKS — Bayesian subgraph posterior (`scripts/bayes_subgraph.py`)
MRF on the FK graph: log P(S) = Σ_{t∈S} a_t + β·(#FK edges in S); a_t = cross-fit learned unary
logit; exact inference (enumerate ≤2^14 subsets), rank by marginal P(t∈S). BIRD, 639 multi-table Qs.

| method | recall@\|gold\| ≥2 | ≥3 | ≥4 |
|---|---|---|---|
| cosine | 0.720 | 0.673 | 0.678 |
| learned unary fusion (β=0) | 0.749 | 0.710 | 0.670 |
| **MRF β=1** | **0.788** | **0.795** | 0.778 |
| MRF β=2 | 0.743 | 0.766 | **0.817** |
| MRF β=4 | 0.669 | 0.708 | 0.768 |

**Result: the structural prior (β>0) beats both learned unary fusion (β=0) and cosine**, and the gain
over unary grows with join complexity (+3.9 → +8.5 → +10.8pp, ≥2→≥4 at β=1; β=2 hits 0.817 on ≥4).
β has an interpretable optimum that rises with complexity. First win (not tie) in the exploration; a
genuine Bayesian posterior, not a heuristic. Caveats: β swept on all data (need held-out β selection
— effect robust enough that tuned β≈1–2); BIRD ≥4 slice is small (n=30). Validation regime = large
multi-hop schemas. Next: held-out β-CV + value-match/column features; then Spider 2.0/BEAVER; then the
calibrated-posterior → conformal coverage + abstain/route layer (reuse Paper 1).

## Phase-1 hardened — held-out β + richer features (`scripts/bayes_subgraph_v2.py`)
Added max question↔column cosine + value-match (LIMIT-500 stored-value scan); β selected on the
held-out fold. BIRD, 639 multi-table Qs.

| method | recall@\|gold\| ≥2 | ≥3 | ≥4 |
|---|---|---|---|
| cosine | 0.720 | 0.673 | 0.678 |
| unary fusion, old feats (β=0) | 0.749 | 0.710 | 0.670 |
| unary fusion, rich feats (β=0) | 0.782 | 0.761 | 0.670 |
| **MRF, held-out β (1, 0.5)** | **0.805** | **0.822** | **0.787** |

Rich features lift unary +3.3/+5.1pp (≥2/≥3); **MRF beats rich unary fusion out-of-sample by
+2.3/+6.1/+11.7pp** (structure gain honest, not β-overfit), net +8.5/+14.9/+10.9pp over cosine, gain
growing with join complexity. **Phase-1 core is hardened and solid.** Caveats: ≥4 slice small (n=30);
modest BIRD schemas — large-schema validation (Phase 2) is the headline test.

Next: **Phase 3** (the methodological novelty, doable on BIRD now) — turn the subgraph-posterior
marginals into a calibrated P(gold ⊆ retrieved) coverage signal, conformal/LTT threshold →
abstain / ask-for-schema / route; show a risk-coverage frontier and that the posterior beats a
cosine-margin abstention baseline. **Phase 2** (parallel) — Spider 2.0-Lite / BEAVER scaling.

## Phase 3 — selective/risk-controlled retrieval: posterior is NOT the abstention signal (`scripts/phase3_selective.py`)
Retriever R = posterior marginal≥0.5 (completeness rate gold⊆R = 0.745). AUROC for predicting a
COMPLETE retrieval:

| completeness signal | AUROC | LTT coverage @risk 0.05 |
|---|---|---|
| **cosine max-out** (−max excluded-table cosine) | **0.763** | **0.22** |
| posterior P(S⊆R) | 0.700 | 0.03 |
| cosine margin | 0.631 | 0.00 |
| all signals combined (xfit) | 0.670 | — |

**Negative result:** a trivial cosine heuristic ("is a high-similarity table excluded?") beats the
Bayesian posterior as an abstention signal, and the posterior does **not add** to it (combined 0.670 <
maxout 0.762). The selective-retrieval *framework* works (LTT gives ~22% coverage at ≤5% incompleteness
risk via cosine max-out), but the **posterior is not the source of the confidence**. Phase-3's
methodological novelty (calibrated subgraph posterior → better selective retrieval) does not hold.

## Consolidated verdict (retrieval direction)
A consistent pattern now holds across the WHOLE project (BNP-correctness, ambiguity, retrieval):
**structured/Bayesian objects help RANKING/recall but lose to simple baselines on the
UNCERTAINTY/decision layer.**
- Paper 1: agreement/structure plateau; a reasoning verifier wins correctness.
- Ambiguity: discovery tractable, but reserve/divergence signals lose for detection.
- Retrieval: **Phase 1 — FK-graph MRF beats cosine/fusion on multi-hop recall (solid, hardened win).
  Phase 3 — the posterior loses to cosine max-out for abstention.**

So the bankable result is **Phase 1: structured graph-prior retrieval improves multi-hop SQL table
recall** (out-of-sample, +8–15pp over cosine, gain scaling with join count). It is an *applied*
text-to-SQL retrieval contribution (crowded field: CHESS/CRUSH4SQL/RESDSQL/LinkAlign), differentiated
by the explicit FK-graph MRF posterior and the complexity-scaling. The Bayesian-UQ/decision novelty
did not materialize. To be a strong paper it needs large-schema validation (Phase 2); to be a *methods*
paper, the UQ angle would need a different signal than the posterior.

## Phase-1 validation — significance + per-DB (`scripts/phase1_validate.py`)
Paired bootstrap 95% CIs (recall@|gold|), all exclude 0:

| | MRF − cosine | MRF − unary fusion |
|---|---|---|
| ≥2 (n=639) | +0.086 [+0.064,+0.107] | +0.024 [+0.007,+0.040] |
| ≥3 (n=156) | +0.155 [+0.120,+0.192] | +0.067 [+0.037,+0.097] |
| ≥4 (n=30) | +0.141 [+0.083,+0.200] | +0.149 [+0.100,+0.200] |

Pure structure effect (MRF−unary) significant and grows with complexity. **Per-DB: the win is
CONDITIONAL on schema richness** — financial +0.203, student_club +0.195, formula_1 +0.187,
superhero +0.125, debit_card +0.044 (rich/large schemas) vs thrombosis −0.029, california_schools
−0.057, toxicology −0.093 (tiny 3–4 table schemas, where coupling over-includes). Clean mechanism:
structure helps ∝ available structure. Refinement: adaptive β gated by schema size/FK-density
(β→0 for tiny schemas). Large-schema regime (all schemas rich) is the natural home.

## Phase-1 adaptive-β refinement — FAILED (`scripts/phase1_adaptive.py`)
Gating β=0 for ≤4-table schemas, β=1.5 else: overall recall 0.781 < fixed β=1 0.806 (adaptive−fixed
−0.025 [−0.038,−0.011]). Per-DB shows the tiny-schema regression is mostly the LEARNED UNARY features
underperforming raw cosine on already-solved schemas (california β=0 0.893 < cosine 0.951), not the
structural prior — so a table-count gate can't fix it. **Conclusion: keep fixed β=1.** The win is
significant and concentrates where retrieval is HARD (financial 0.61→0.81, formula_1 0.47→0.66);
regressions only on tiny schemas where cosine was already ~0.9 (low-stakes). Defensible as-is;
smarter per-schema adaptation not worth it over fixed β.

## Downstream end-to-end — better retrieval → better SQL (`scripts/downstream_ex.py`)
Prune schema to top-K=5 tables under each retriever, generate SQL (gpt-4o-mini), execute. BIRD
schemas ≥8 tables (financial, formula_1, student_club, superhero; 436 Qs). Retrieval recall@5:
cosine 0.849 vs MRF 0.958.

| schema shown | EX |
|---|---|
| full (no prune) | 0.500 |
| cosine top-5 | 0.438 |
| **MRF top-5** | **0.495** |
| oracle (gold tables) | 0.562 |

**EX(MRF) − EX(cosine) = +0.057 [+0.021,+0.094] (significant).** MRF-pruning ≈ full-schema (prune to
5 tables with ~no loss) while cosine-pruning costs −6pp. Oracle > full (+6pp) → distractor tables hurt
generation, so precise retrieval exceeds full-schema and there's headroom above MRF. Per-DB strongest
where hardest: formula_1 +13.6pp (MRF beats full-schema), superhero +8.2, student_club +4.5; ANOMALY:
financial −3.8pp (MRF worse downstream despite higher recall; n=106). **Verdict: the structured
retrieval win translates to end-task accuracy — Phase 1 is a complete applied result.**

## DECISIVE: FK-heuristic vs MRF — is it "just adding bridge tables"? (`scripts/phase1_fkbaseline.py`)
Held-out hyperparams, recall@|gold|:

| method | ≥2 | ≥3 | ≥4 |
|---|---|---|---|
| cosine | 0.720 | 0.673 | 0.678 |
| unary fusion | 0.782 | 0.761 | 0.670 |
| FK-1hop heuristic | 0.782 | 0.763 | 0.678 |
| FK-closure (shortest-path) heuristic | 0.786 | 0.779 | 0.760 |
| MRF (subgraph posterior) | 0.805 | 0.822 | 0.787 |

MRF − FK-closure (bootstrap): ≥2 +0.018 [+0.004,+0.033], ≥3 +0.043 [+0.018,+0.070], ≥4 +0.027
[−0.015,+0.077]. **A shortest-FK-path closure heuristic captures MOST of the structural win**; the
full MRF adds a small *significant* increment on ≥2/≥3 (not ≥4, n=30). Honest implication: the win is
mostly *connectivity* (a cheap heuristic realizes it); the MRF's defense is the small increment + a
principled reason to scale better where hard closure over-includes (large/dense schemas — UNTESTED).

**Reframed contribution:** lead with *connectivity-aware structured retrieval helps multi-hop schema
linking* (heuristic + downstream EX); position the Bayesian subgraph posterior honestly as the
principled, evidence-weighted, distractor-selective version that adds a modest gain and should widen on
large schemas. Theory: bridge/connectivity (#1) = dominant grounded mechanism (heuristic embodies it);
distractor/selectivity (#3) = the MRF's extra. Large-schema validation now tests BOTH scale AND whether
MRF pulls ahead of the heuristic.

## Cosine-correlation coupling in the prior — doesn't add for tables (`scripts/phase1_cosinecoupling.py`)
Extended subset model with a cosine pairwise term (both signs), held-out γ, β=1:

| variant | ≥2 | ≥3 | ≥4 |
|---|---|---|---|
| FK-only MRF | 0.806 | 0.829 | 0.820 |
| FK + cosine-attractive | 0.782 | 0.827 | 0.812 |
| FK + cosine-repulsive | 0.810 | 0.790 | 0.753 |
| cosine-attractive only (no FK) | 0.784 | 0.771 | 0.687 |
| cosine-repulsive only (no FK) | 0.783 | 0.762 | 0.695 |

(1) Attractive/smoothing: no help (slightly hurts) — wrong sign for tables. (2) Repulsive/DPP: HURTS
recall on multi-hop (pushes out co-relevant cosine-similar gold tables, ≥3 0.829→0.790); its distractor
payoff can't show in recall and would be dominated by the recall loss → downstream EX not worth running.
(3) **Cosine-coupling cannot substitute for FK** — cosine-only tops ~0.78/0.77/0.69 ≪ FK 0.806/0.829/
0.820 on multi-hop. **Keeper:** the embedding-similarity graph and the FK graph are *different
structures*; only FK carries the signal (connectors are cosine-*dissimilar*) — justifies the FK prior
and rules out a metadata-free correlation shortcut. Model elaboration on BIRD is now at diminishing
returns; the decisive open question is large-schema validation.

## S1-a: graph-GP / diffusion priors (`scripts/s1a_graphgp.py`)
Personalized PageRank (diffusion from unary over the FK graph) ≈ MRF: 0.812/0.810/0.810 vs MRF
0.806/0.829/0.820 (≥2/≥3/≥4), both ≫ cosine 0.720/0.673/0.678. Graph-GP Laplacian smoothing
UNDERPERFORMS (0.756/0.723/0.728): symmetric smoothing dilutes the unary signal; PageRank's restart
preserves seeds. **Structural win is robust to modeling choice — a 3-line diffusion matches the Ising
MRF (its edge is negligible, only ≥3/≥4). Two simple structural baselines (FK-closure, PageRank) now
match the MRF.** Not all graph priors work (Laplacian over-smooths) — the diffusion form matters.

## S4: multi-hop RAG (HotpotQA distractor) — the structural win GENERALIZES (`scripts/s4_hotpot.py`)
1492 questions (1205 bridge, 287 comparison), 10 passages each, structural prior = title-mention link
graph. recall@2 of the 2 supporting passages:

| method | all | bridge | comparison |
|---|---|---|---|
| cosine | 0.685 | 0.642 | 0.866 |
| unary fusion | 0.749 | 0.718 | 0.876 |
| **PageRank (title graph)** | **0.814** | **0.812** | 0.822 |
| MRF (title graph, β=1) | 0.789 | 0.772 | 0.861 |

MRF−cosine +0.104 [+0.090,+0.118] (all), +0.130 [+0.114,+0.146] (bridge). **The SQL structural win
replicates in text RAG**: graph prior beats cosine +10–13pp, concentrated on BRIDGE (the multi-hop
analog of FK bridges; +17pp), ~tie/slight-hurt on COMPARISON (no bridge to recover) — same conditional
as SQL. PageRank≈MRF again (structure, not the specific Bayes). Scope: distractor *rerank* (10
candidates, clean Wikipedia title-links), not full-corpus first-stage retrieval.

## S4-d (diversity) + S4-e (UQ) on HotpotQA — topology-dependent complementarity (`scripts/s4_diversity_uq.py`)
Redundancy real (mean off-diag passage-sim 0.52). recall@2:

| method | all | bridge | comparison |
|---|---|---|---|
| cosine | 0.685 | 0.642 | 0.866 |
| MMR (λ=0.7) | 0.684 | 0.624 | **0.939** |
| DPP (k=2) | 0.664 | 0.600 | 0.934 |
| PageRank (structure) | 0.703 | **0.719** | 0.639 |

**Orthogonality critique CONFIRMED & refined:** diversity helps under redundancy — but only for
COMPARISON (+7pp; independent/contrastive evidence) — and HURTS bridge; structure is the mirror
(best bridge, worst comparison). **The right inductive bias is query-topology-dependent: structure
for bridge (connected evidence), diversity for comparison (independent evidence).** Oracle
topology-routing ≈ 0.76 recall@2, beating every fixed method (best single 0.703) → an adaptive
method to build (query-type is easily predicted).

S4-e UQ: predicting "both gold in top-2" — cosine margin 0.744 > softmax-pair posterior 0.692 >
cosine max-out 0.625. **Family-5 (UQ) keeps losing in RAG too** (simple cosine signal beats the
posterior) — lesson #2 replicates across domains.

## S4 adaptive topology-routed retriever — a POSITIVE method result (`scripts/s4_adaptive.py`)
Predict query topology (bridge vs comparison) from lexical cues (cross-fit logistic, acc 0.936), route:
bridge→PageRank(structure), comparison→MMR(diversity). recall@2: cosine 0.685, PageRank-all 0.703,
MMR-all 0.684, **adaptive 0.748**, oracle-routed 0.761. **ADAPTIVE − best-fixed +0.044 [+0.036,+0.053]**
(significant). The topology complementarity is EXPLOITABLE with a cheap classifier → match the
inductive bias to the query's evidence topology beats any single fixed bias. A constructive method
(not just an audit finding); generalizes the conditional "structure helps when structure exists."
