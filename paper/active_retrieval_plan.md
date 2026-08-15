# Bayesian active retrieval under budget — the genuinely-Bayesian IR paper (SIGIR-shot)

**One-line thesis:** When relevance judgments are expensive (LLM calls) and budget-limited, retrieval
becomes Bayesian experimental design; and the corpus STRUCTURE that does not help ranking becomes
load-bearing as the COVARIANCE that makes each judgment efficient. A graph/similarity-kernel GP plus a
hierarchical cross-query prior beats plain-GP active learning (BAGEL) and fixed-budget reranking,
especially at low budgets and where relevance is clustered/connected.

## Why it's distinct from BAGEL (2604.17906)
- BAGEL: query-specific GP, LLM-score-seeded, active selection. Kernel ~ embedding RBF.
- OURS: (1) STRUCTURE-AS-COVARIANCE — kernel from the corpus graph / similarity, propagating judgments to
  connected candidates; the same object that HURT ranking (diffusion buries the answer) HELPS as a
  covariance ("similar docs have correlated relevance"). (2) HIERARCHICAL/BNP CROSS-QUERY PRIOR — cold-start
  borrows strength across queries (our validated few-shot-pooling win), a proper prior mean + calibration
  that BAGEL lacks. (3) Connectivity-dependence PREDICTION: the gain concentrates where relevance is
  clustered/multi-hop, tying to our connectivity-boundary program.

## The unification (why this paper is "us")
- Structure doesn't help ranking directly (our audit) -> but as a GP covariance it IS load-bearing:
  epistemic value (which judgments inform which candidates), not relevance-prediction. Deep, non-obvious.
- Hierarchical pooling (validated few-shot win) -> the cross-query prior for cold-start.
- Bayes wins because it's the DECISION / experimental-design layer, where audit + literature agree it earns
  its keep (BAGEL is the one credible pro-Bayes result in the field sweep).

## Experimental design
Setting: per query, first-stage top-N candidate pool (N=100); each candidate has cheap features (dense,
BM25, cross-encoder) and can be JUDGED (expensive) to reveal relevance. Budget B judgments/query.
- Judge model: oracle (gold) for the controlled study (cheap, clean); real LLM-as-judge validation on a
  subset (uses tokens/GPU) for external validity.
- Prior mean: cross-fit calibrated cheap-relevance (logistic on cosine / a hierarchical reranker score).
- GP: kernel over candidate embeddings/graph, K_ij = exp(-(1-cos_ij)/l); posterior conditioning on judged.
- Acquisition: UCB (mean + beta*sd) among unjudged; batch variant for parallel LLM calls (GPU/token runs).
Baselines: no-judge (rank by prior) | passive top-B judge | uncoupled GP (diagonal kernel = no propagation)
| plain-GP UCB (BAGEL-lite) | OURS graph/similarity-GP + hierarchical prior | oracle-active (ceiling).
Metric: nDCG@10 (and recall) vs BUDGET B in {0,5,10,20,40}; per-domain; paired significance.
Benchmarks: BEIR (17 domains, have features/pools) spanning clustered vs single-gold relevance; multi-hop
RAG (HotpotQA, 2Wiki) for the connected-corpus regime; optionally TravelDest for direct BAGEL comparison.

## Pre-registered predictions (falsifiable)
1. Coupled-kernel GP-UCB > passive-top-B and > uncoupled GP at LOW budget (CI excludes 0).
2. The coupled-kernel advantage GROWS with relevance clustering / corpus connectivity (nfcorpus/scidocs/
   multi-hop >> arguana/scifact single-gold) -- the connectivity boundary reappearing in ACQUISITION.
3. Hierarchical cross-query prior beats a per-query flat prior at low budget (cold-start shrinkage).
4. OURS >= plain-GP (BAGEL-lite) at matched budget, with the gap from structure-covariance + hierarchical prior.

## Feasibility gate (run FIRST, on cached data, oracle judge)
Does coupled-kernel GP active retrieval beat passive/uncoupled under budget, more so in clustered-relevance
domains? -> scripts/active_pilot.py. If yes, commit to big runs (real LLM judge, more benchmarks, batch
acquisition, hierarchical prior, BAGEL comparison). If no, diagnose kernel/prior before scaling.

## Big-run phase (uses GPUs / LLM tokens)
- Real LLM-as-relevance-judge across benchmarks (token budget).
- Larger pools (N=100-500), batch acquisition (parallel judgments).
- Hierarchical/BNP cross-query prior (DP over query regimes / GP hyperparameters).
- Head-to-head vs BAGEL and strong fixed-budget rerankers; budget-efficiency curves.
- Reproducibility package.

## Venue
SIGIR 2027 (full; ~late Jan 2027) if the story lands; else SIGIR-AP / CIKM / ECIR. Genuinely Bayesian,
showcases GP/BNP expertise, on the one hill where Bayes wins in retrieval.

## FEASIBILITY GATE RESULT (2 pilots)
Pilot 1 (BEIR single-hop, cosine kernel) — FAILED: coupled GP-UCB -0.32 vs passive. Diagnosis: (a) ranking
bug (judged non-rel floated instead of sank), (b) wrong regime — when dense prior already tops the gold,
"verify top-B" is near-optimal and exploration only hurts; cosine kernel explores away from the good cluster.
Pilot 2 (HotpotQA multi-hop, GRAPH kernel, bug fixed) — GREEN: on BRIDGE questions graph-GP beats passive
verify-top AND cosine-GP at low budget (recall@2 B=1: 0.742 vs 0.671 passive / 0.680 cosine-GP; B=2:
graph-GP - passive = +0.071 [+0.059,+0.083]). Neutral on COMPARISON (independent evidence, no bridge). The
GRAPH kernel specifically wins (graph-GP > cosine-GP > passive) => distinct from BAGEL; connectivity
dichotomy reappears in ACQUISITION. "Structure as covariance not prior" CONFIRMED. Caveats: 10-cand pool,
oracle judge, one dataset -> big runs fix these. Scripts: active_pilot.py (pilot 1), active_pilot2.py (pilot 2).
GO for big runs.

## SCALED GATE (2 datasets, cache, oracle judge) — GREEN, REPLICATED  [scripts/graphrag_active_scale.py]
recall@k vs budget B; graph-GP (ours, GMRF kernel inv(I+lambda L)) vs passive verify-top-B vs cosine-GP (BAGEL-lite):
- HotpotQA CHAINED (bridge): graph-GP - passive **+0.071 [+0.059,+0.083]** @B=2; - cosine-GP +0.046 @B=2.
- 2Wiki CHAINED (compositional/inference/bridge_comparison, 1168q): graph-GP - passive **+0.106 [+0.095,+0.118]** @B=2;
  - cosine-GP +0.077 @B=2 (STRONGER than HotpotQA -> richer multi-hop helps more).
- INDEPENDENT (comparison), BOTH datasets: graph-GP - passive ~0 (CI includes 0) -> the connectivity boundary,
  now razor-sharp and symmetric across two datasets.
The GRAPH kernel specifically beats the embedding kernel (graph-GP > cosine-GP > passive) => distinct from BAGEL;
the win is the STRUCTURE, not "any GP." Core contribution validated beyond the pilot.
NEXT (big runs): larger pools (N=100 full-wiki retrieval, not the 10-passage distractor set), real LLM-as-judge on
a subset, hierarchical cross-query prior ablation, DOWNSTREAM multi-hop QA (retrieval win -> answer accuracy),
BAGEL head-to-head, MuSiQue as a 3rd dataset, and a continuous connectivity-boundary curve (gain vs bridge-buriedness).

## CONNECTIVITY-BOUNDARY CURVE + PRIOR ABLATION (cache)  [scripts/graphrag_active_analysis.py]
(1) The chained/independent dichotomy is a continuous LAW. graph-GP - passive recall@k @B=2 on chained questions
(2373, both datasets pooled), by cosine rank of the hardest gold x golds-connected:
  gold in top-2: ~0 (found already). rank 3: slightly - (passive verify-top-2 already surfaces it). rank 4:
  **+0.272 [+0.238,+0.305] connected** vs +0.071 not. rank 5+: **+0.190 [+0.174,+0.206] connected** vs +0.020 not.
  => gain concentrates sharply where a gold is BURIED (rank>=4, beyond verify-top reach) AND CONNECTED (a bridge to
  propagate along); connected is 4-10x not-connected at every buried bin. The mechanism made law, self-explaining
  (rank-3 golds are reached by passive verify-top-2, so no gain there).
(2) Prior ablation: graph-GP with the cross-query-pooled calibrated prior (0.658->0.903 over B=0..4) vs a flat
  base-rate prior (0.224->0.574) => the calibrated prior mean and the graph covariance are COMPLEMENTARY (propagation
  alone is far weaker). CAVEAT: does NOT isolate the HIERARCHICAL (pooled-cross-query vs per-query) value, since
  pooled-calibrated and per-query-raw-cosine share a B=0 ranking -> needs the few-shot cold-start setup (big-run item).

## DOWNSTREAM MULTI-HOP QA -- the retrieval win becomes an ANSWER win  [scripts/graphrag_downstream_qa.py]
Feed the budget-B top-k retrieved passages to gpt-4o-mini, score EM/F1 vs gold (300 chained Qs, both datasets,
answers cached data/graphrag_qa_answers.json, actual cost $0.05). graph-GP - passive, paired bootstrap 95% CI:
  B=0 (shared prior baseline, identical retrieval): +0.000 all metrics.
  B=1: EM +0.040 [+0.007,+0.073], F1 +0.052 [+0.020,+0.084]  (recall +0.076).
  B=2: EM +0.050 [+0.013,+0.087], F1 +0.059 [+0.024,+0.096]  (recall +0.083).
  B=3: EM +0.047 [+0.013,+0.080], F1 +0.053 [+0.023,+0.085]  (recall +0.067).
Every CI at B>=1 excludes 0. The EM/F1 gain TRACKS the recall gain (answer gain ~= 2/3-3/4 x recall gain) =>
clean causal chain retrieval->answer. graph-GP > cosine-GP on answers too (B=1 EM 0.433 vs 0.387); BAGEL-lite
does NOT convert (cosine-GP ~ passive at B=1). End-task headline CONFIRMED **UNDER THE ORACLE JUDGE ONLY** --
does NOT survive an off-the-shelf real judge (see RED stress test next).

## REAL LLM-JUDGE STRESS TEST -- RED: the oracle win does NOT survive an off-the-shelf judge  [scripts/graphrag_llm_judge.py, graphrag_judge_fix.py]
Replaced the oracle judge in the active loop with gpt-4o-mini relevance verdicts (yes/no per candidate,
cached data/graphrag_judge_labels.json). Judge quality vs gold: precision 0.643, **recall 0.350**, acc 0.809,
says-yes 0.123 vs gold 0.227 -- CONSERVATIVE and BRIDGE-BLIND (asked 'does it help answer', it says no to
intermediate/bridge passages, exactly what multi-hop needs). Result, two layers:
(1) HARD-pin design (current) COLLAPSES: graph-GP - passive recall -0.043 / EM -0.057 / F1 -0.067 @B=3 (all
  significant NEGATIVE). Reckless trust in a noisy judge sinks gold; graph-GP amplifies via propagation.
  Passive recall itself craters 0.657 -> 0.37.
(2) SOFT/Bayesian fix (judge label as noisy evidence: rank by GP posterior mean, obs-noise sn2 ~ judge
  reliability) removes the collapse -- graph-GP recovers to ~0.62-0.65 recall. BUT fair robustified compare
  (sn2=1.0, under the real judge): graph-cosine +0.017..+0.025 (CI INCLUDES 0, n.s.); graph-prior -0.037..-0.007
  (acting on this judge is NO BETTER than ignoring it and using the cosine prior 0.657).
VERDICT: the oracle end-task win is an ORACLE artifact; a realistic off-the-shelf judge is too weak/bridge-blind
to realize it. Silver linings (real but modest): the Bayesian soft design is robust where naive hard-pin self-
destructs; structure damps judge noise. Genuine finding either way: off-the-shelf answer-relevance judges are
bridge-blind -> naive LLM-judged active retrieval fails multi-hop.
DECISIVE NEXT TEST: a HOP-AWARE judge (graded 0-2 / 'relevant to any step incl. supplying a linking entity' /
gpt-4o, ~$0.15). If higher judge recall restores an honest margin, the story lives ('needs a hop-aware judge');
if not, the paper is ABOUT the failure mode. (Also: 10-passage/strong-prior regime gives active retrieval little
room; N=100 full-wiki has more -- not leaned on to rescue.)

## HOP-AWARE JUDGE -- RED #2: fixing the judge does NOT rescue the structure win  [scripts/graphrag_judge_hopaware.py]
Graded 0-2 hop-aware prompt (explicitly credits linking/intermediate passages), gpt-4o-mini, soft sn2=1.0,
graded label g -> soft target g/2. It FIXED bridge-blindness: judge recall on gold 0.350 -> **0.751** (g>=1,
precision 0.534; g==2 recall 0.494). BUT the structure win still did not return -- fair margins under the
hop-aware judge: graph-cosine recall +0.013/-0.003/+0.000 (CIs include 0), graph-cosine F1 -0.030/-0.021/-0.020
(B=1 significantly NEGATIVE), graph-prior recall ~0 (paying to judge+retrieve via the graph ~= just using the
cosine prior 0.657). graph beats HARD passive (+0.05..+0.09) only because hard-passive self-destructs.
DEEPER FINDING (now across TWO judge designs): the graph-kernel advantage over the embedding kernel is an
ORACLE artifact -- under realistic label noise the discrete-connectivity signal washes out and graph ~= cosine
~= prior. Judge quality is NOT the bottleneck anymore (0.75 recall is fine) so gpt-4o escalation won't help
(doesn't address the failure). WHAT SURVIVES: (i) the oracle result as a clean controlled UPPER BOUND (real,
oracle-only); (ii) the Bayesian soft design as NECESSARY to avoid self-destruction; (iii) judge findings
(off-the-shelf judges bridge-blind; graded hop-aware prompt fixes recall 0.35->0.75). The 'structure beats
BAGEL under budget' HEADLINE does not survive a real judge. Last untested positive shot: N=100 full-wiki
(weak prior -> structure has room). Else reframe to the honest 'when does structure help active retrieval'
characterization (oracle upper bound + why it collapses under real judges).

## N=100 ORACLE DIAGNOSTIC (cache, $0)  [scripts/graphrag_n100.py]
Corpus = all encoded dev passages (Hotpot 14549, 2Wiki 9062); retrieve top-100 per chained question (>=2 golds
in pool: kept 150/167 Hotpot, 150/188 2Wiki). PRIOR recall@k over the top-100 pool = 0.632 -- NOT much weaker
than N=10 (0.66): the >=2-golds-in-pool filter re-selects cosine-findable golds, so the hoped 'weak prior'
regime only partially materialized. Oracle margins at N=100: graph-cosine +0.060[+0.033,+0.087] @B=1, +0.039
@B=2, +0.016 @B=3 (n.s.); graph-passive +0.06..+0.07. => structure signal PRESENT + significant but NOT amplified
vs N=10 (+0.05..+0.08). Since the N=10 real-judge washout erased a similar-sized oracle margin, expectation for a
real-judge N=100 run is GUARDED -- though the deeper burial (bridge at rank 20-90, where cosine propagation can't
reach but a title-mention edge can) is a mechanism the margin-size may understate. Decision surfaced to Robert
(2026-08-15): spend ~$1 on the definitive real hop-aware-judge N=100 run, or reframe now.
