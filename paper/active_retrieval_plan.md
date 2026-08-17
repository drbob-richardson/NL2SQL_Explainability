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

## N=100 REAL HOP-AWARE JUDGE -- MODEST REVIVAL: structure separates in the deep-burial regime  [scripts/graphrag_n100_judge.py]
Judged the full top-100 pool (hop-aware graded, gpt-4o-mini, 22k calls, $1.02; judge recall on gold 0.821),
soft sn2=1.0 (same as the N=10 fair test -- not tuned). POOLED (n=240): graph-cosine +0.032[+0.008,+0.056] @B=1,
+0.042[+0.015,+0.069] @B=2 (significant); graph-prior +0.037/+0.045/+0.036 @B=1/2/3 (significant). graph-GP is
the ONLY method that beats the cosine prior (cosine-GP ~= prior; passive self-destructs under the hard pin).
Advantage is LOW-BUDGET (B=1-2), decays by B=3-4 (convergence). PER-DATASET (n=120, under-powered): 2Wiki shows
graph-cosine (+0.040/+0.042 sig), Hotpot shows graph-prior (+0.050/+0.067 sig); the OTHER margin is positive but
n.s. on each. All 8 point estimates positive across both datasets/comparisons -- consistent in sign, modest in
size (~+0.03-0.04). KEY FINDING = a REGIME BOUNDARY: N=10 (shallow bridge) washes out under a real judge, N=100
(deep burial, rank 20-90) separates -- structure earns its keep exactly when the bridge is buried beyond the
embedding kernel's reach. Contingent on: hop-aware judge (bridge recall) + deep burial + Bayesian soft design.
Modest, honest revival -- NOT a knockout. NEXT to solidify: more questions (tighten per-dataset CIs) + downstream
QA at N=100 (recall->answer payoff). Total GraphRAG spend ~$1.4.
REVISED STORY: not 'structure always wins' (false) nor 'structure never survives a real judge' (also false) --
it's 'structure-as-covariance helps budget-limited active retrieval PRECISELY in the deep-multi-hop regime, with
a hop-aware judge and a noise-aware design; it washes out when the bridge is shallow or the judge is bridge-blind
or the design hard-trusts the judge.' A bounded, mechanistic, defensible contribution.

## N=100 DOWNSTREAM QA (payoff) -- recall win does NOT yet convert to a significant ANSWER win  [scripts/graphrag_n100_qa.py]
Same top-100 pools + cached hop-aware labels; soft retrieval; gpt-4o-mini reader; $0.02 (357 new answers).
POOLED n=240 EM/F1 margins: graph-cosine EM +0.008/+0.013/-0.004, F1 +0.018/+0.014/-0.009 (all CIs include 0);
graph-prior EM +0.004/+0.008/+0.008, F1 +0.014/+0.020/+0.018 (all n.s. but consistently +). The recall win
(+0.03-0.04) is too small to produce a significant end-task gain at n=240 -- unlike N=10 where recall +0.076-0.083
converted to F1 +0.05-0.06. Directionally positive, statistically INCONCLUSIVE. => the modest revival is a
RECALL-level result; the END-TASK payoff is NOT established at this scale. Strongest defensible claim right now =
the recall result + the regime boundary (N=10 washout vs N=100 separation) + the mechanistic story (bridge-
blindness, noise-washout, deep-burial, Bayesian-soft necessity). KEY LEVER to resolve it = more questions (bigger
n) -> tightens both the per-dataset recall CIs AND the QA CIs; tells us if the end-task gain is real-but-small or null.
GraphRAG spend to date ~$1.45.

## FIRM-UP (n=600: 300/dataset, real hop-aware judge, +$1.62)  [graphrag_n100_judge.py --subset 300 --n 8000]
RECALL solidified per-dataset. graph-cosine (structure vs embedding kernel) now SIGNIFICANT ON BOTH datasets @B=1:
Hotpot +0.039[+0.018,+0.061], 2Wiki +0.039[+0.017,+0.060]; 2Wiki also @B=2 +0.033[+0.010,+0.057] (Hotpot @B=2
+0.022 borderline). Pooled n=600: graph-cosine +0.039[+0.024,+0.054] @B=1, +0.027[+0.009,+0.045] @B=2. => the
graph kernel beats the embedding kernel (BAGEL-lite) ROBUSTLY at low budget under a real judge, on both datasets --
the core defensible claim. graph-prior (act vs ignore) is PRIOR-DEPENDENT: significant on Hotpot (weak prior 0.609)
but n.s. on 2Wiki (strong prior 0.684) -- acting-on-the-judge helps most when the prior is weak; the graph-vs-cosine
advantage is the consistent one. Oracle-at-scale (n=2003) graph-cosine +0.080[+0.069,+0.090] @B=1 (ceiling firm).
END-TASK (QA) firm-up n=600 [graphrag_n100_qa.py --subset 300]: STILL NOT significant. Best case F1 graph-cosine
+0.020[-0.000,+0.040] @B=1 (CI touches 0); graph-prior F1 +0.015/+0.019 (n.s.); EM all n.s. Doubling n did not
rescue it -- the +0.04 recall win is too small to reliably move gpt-4o-mini answers.

## CHAIN-COMPLETION REANALYSIS + EVOI RACE (both $0, cached labels)  [graphrag_chain_completion.py, graphrag_evoi.py]
(A) Reanalyze n=600 with SET-COMPLETION utilities: graph-cosine @B=1 chain-completion +0.068[+0.040,+0.097]
pooled (Hotpot +0.070, 2Wiki +0.067), bridge-found +0.070 -- ~1.8x the average-recall margin (+0.039), sig on
BOTH datasets. Average recall was diluting the intervention. Retrieval-vs-reasoning: answer-in-context (oracle
reader) +0.040[+0.015,+0.067] sig, but gpt-4o-mini QA n.s. -> chain more often complete, reader doesn't exploit
it. Reachability ceiling 0.75. => CHAIN COMPLETION is the headline metric; QA null is a decomposition not a failure.
(B) EVOI-vs-UCB race REFUTED as specified. 2x2 {cosine,graph}x{UCB,EVOI} chain-completion @B=1 pooled:
cosine-UCB 0.323 | cosine-EVOI 0.358 | **graph-UCB 0.392 (best)** | graph-EVOI 0.355. EVOI HELPS cosine (+0.035)
but HURTS graph (-0.037[-0.068,-0.007] sig; Hotpot -0.060 sig; 2Wiki -0.013 n.s.); under EVOI graph-vs-cosine ~0.
=> the graph kernel's advantage lives in UCB's EXPLORATION (judge confident anchor -> propagate along edge to the
buried bridge); myopic omitted-mass EVOI exploits directly, over-trusts the 0.27-precision judge, forgoes the
propagation, neutralizes the kernel. REFUTES 'acquisition is the bottleneck'; kernel x acquisition are ENTANGLED,
graph NEEDS UCB. graph-UCB stays the method. Cleaner mechanistic finding than a win would have been.
PIVOT: bank graph-UCB + the kernel x acquisition interaction; paper spine = chain-completion headline +
retrieval-vs-reasoning decomposition + this mechanism + regime boundary. Amplifiers (lambda_q mixture, MuSiQue/
N=500, real BAGEL) over more acquisition engineering. (Only lower-odds acquisition variant left: propagation-aware,
cosine-prior-based completion target.)

## ORACLE-LAMBDA CEILING + an accidental KERNEL-NORMALIZATION win  [scripts/graphrag_lambda_ceiling.py]
Built the mixture kernel K_q=(1-lam)*Ehat+lam*Ghat with BOTH kernels normalized to unit diagonal (correlation form).
(1) BIG, FREE win: normalizing the GMRF graph kernel to correlation form strictly improves graph-UCB chain-completion:
+0.053[+0.028,+0.078] @B=1, +0.093 @B=2, +0.115 @B=3 over the RAW kernel. => graph-cosine margin DOUBLES from +0.068
(raw) to **+0.122[+0.088,+0.155] @B=1, +0.150 @B=2** (normalized). Mechanism (amplifies the EVOI finding): the raw
GMRF gives high-degree HUBS lower prior variance so UCB under-explores them; the hubs are the anchors whose judgment
propagates to bridges. Equalize variance -> UCB explores hubs -> propagation fires -> chains complete. ADOPT the
normalized kernel as the method; re-run the key recall/completion/N=100 comparisons to update the headline upward.
(2) Oracle-lambda headroom over the (normalized) graph kernel: +0.030[+0.015,+0.045] @B=1, +0.045 @B=3 -- modest but
significant. lam* is PREDICTABLE: graph-strictly-best queries have golds_connected 0.93 / bridge_reachable 0.93 /
deepest-gold-rank 23 vs 0.71/0.74/13 for mix/cosine-best -> a learned lambda_q keyed on connectivity+burial is
feasible (the structural-leverage regime). SEQUENCE: adopt normalized kernel (re-run headline, $0) -> build learned
lambda_q (modest extra headroom) -> then amplifiers (MuSiQue/N=500, real BAGEL, judge independent Qs for the full
lambda_q adaptivity story).

## NORMALIZED-KERNEL END-TASK RESULT -- the answer win LANDS  [scripts/graphrag_n100_normalized.py]
Re-ran N=100 downstream QA with normalized kernels ($0.01, 131 new answers). The doubled retrieval effect CARRIED
to the end-task. POOLED n=600 F1 graph-cosine +0.040[+0.018,+0.063] @B=1 (raw was +0.020 n.s. -> now SIGNIFICANT),
+0.041 @B=2, +0.039 @B=3; EM +0.033/+0.035/+0.035 (all sig); graph-prior F1 +0.035/+0.052/+0.064 (sig). recall
graph-GP 0.69/0.71/0.72, completion 0.45/0.49/0.51 (vs cosine 0.63.. / 0.32..). PER-DATASET: Hotpot F1 +0.048/
+0.058/+0.059 (all SIG), EM +0.037/+0.053/+0.053 (sig); 2Wiki F1 +0.032[-0.002,+0.067] BORDERLINE, +0.024, +0.019
-- 2Wiki retrieval/completion IS sig (+0.09-0.12) but the reader converts less (compositional answers, stronger
prior); the retrieval-vs-reasoning gap is dataset-dependent. => 'GraphRAG (graph-covariance active retrieval)
beats BAGEL-lite on ANSWERS' is now REAL pooled + Hotpot, borderline 2Wiki. Kernel normalization flipped the
investigation from 'recall win, end-task null' to 'recall + completion + answer win'. NEXT: learned lambda_q
(modest extra headroom); firm up 2Wiki / add MuSiQue; real BAGEL; write-up.

## MuSiQue HOP-SCALING TEST -- prediction NOT confirmed; MuSiQue much harder  [scripts/musique_run.py]
Hop-aware judge (48.3k calls, ~$2.1) + normalized kernels, per-hop 200/200/83 (2/3/4-hop, require-all chain in
top-100). MuSiQue is a MUCH harder retrieval problem than Hotpot/2Wiki: prior recall@k 0.48/0.45/0.33, chain-
completion 0.15/0.02/0.00 -- completing a full k-hop chain in top-k FLOORS OUT for k=3,4 (all-or-nothing metric
mis-specified for long chains). Result (graph-GP vs cosine-GP, normalized):
  2-hop (n=200): completion +0.050[+0.015,+0.085] @B=2 (sig); F1 +0.005/+0.009 (n.s.).
  3-hop (n=200): completion ~0 (floored); **F1 +0.064[+0.026,+0.103] @B=1 (SIG)** -- graph surfaces partial chains.
  4-hop (n=83): recall FLOORS and DROPS with budget (0.33->0.27, judging HURTS); F1 -0.025 (n.s.); completion 0.
=> The 'margin RISES monotonically 2->3->4 hop' prediction FAILED: F1 non-monotonic (~0 / +0.064 sig / -0.025),
completion floors, 4-hop is a failure regime. The graph advantage does NOT cleanly generalize to MuSiQue and
does NOT scale with hop count as the mechanism predicted. Bright spot: the 3-hop F1 win is real. Net: MuSiQue
TEMPERS the CS/ML story back toward 'bounded to Hotpot/2Wiki-style shallow multi-hop,' not a scaling law.
### AUTOPSY (done, $0) -- the negative is GRAPH CONSTRUCTION, NOT mechanism  [musique_diagnose.py, musique_entity_graph.py]
- MuSiQue graph density HIGHER than Hotpot (0.035 vs 0.004) but GOLD-connectivity much LOWER (title: 0.35 vs 0.76)
  -- edges in the wrong place (built against title-shortcuts). Judge also noisier (recall 0.57 vs 0.89, secondary).
- Entity-overlap graph reconnects golds (3-hop 0.23->0.86) but too DENSE (0.17) -> diffuse propagation -> no gain.
- Sparsity sweep (df cutoff 0.30->0.03, min-shared 1/2): NO surface config restores a significant margin (2hop ~+0.02,
  3hop ~0 everywhere). MuSiQue's distractors were chosen to share entities with golds -> can't sparsify gold-edges
  without killing them (adversarial entanglement of surface co-occurrence and reasoning).
- **ORACLE gold-clique graph (golds connected, density 0.0005): rec-margin @B=2 +0.068[+0.035,+0.100] (2hop),
  +0.087[+0.050,+0.125] (3hop)** -- LARGE, significant, INCLUDING 3-hop where surface gave ~0.
=> THE MECHANISM WORKS ON MuSiQue given the right (sparse + gold-connected) graph; the binding constraint is GRAPH
CONSTRUCTION. DEEPENED THESIS: structure-as-covariance value = f(graph-chain ALIGNMENT); gold-connectivity is the
measurable alignment metric; oracle ceiling +0.07-0.09; surface graphs align for standard multi-hop, fail for
adversarial (MuSiQue). This is a mechanism + a LAW about when it applies -- stronger than a clean win.
FREE-GRAPH PROBES (musique_implgraph.py, $0): question-entity graph FAILS (gold-conn ~0.05 -- bridges are not
question-named entities, confirming the hop is latent); prior-gated-entity halves density but 3-hop margin still
~0 (MuSiQue distractors are high-cosine by design, so a relevance gate can't separate them). => CONFIRMED: no
free (cosine+entity) graph recovers MuSiQue's chain; the gap to the oracle (+0.09) is entirely the inability to
exclude distractors without a true relevance/logical signal.
NEXT (motivated, costly, uncertain): an LLM-inferred LOGICAL graph to approximate the oracle-clique WITHOUT gold.
Cheapest principled version = DECOMPOSITION graph: LLM decomposes each question into ordered single-hop sub-qs
(~483 calls ~$0.05), cosine-retrieve top-few per sub-q, connect co-/consecutively-retrieved passages -> a sparse
chain graph. How close to the +0.07-0.09 ceiling can it get? Also: fairer chain-recall metric.

## FIRMED-UP BOTTOM LINE (GraphRAG investigation, total spend ~$5.3)
DEFENSIBLE: a bounded, structure-specific RECALL result -- graph-kernel GP-UCB active retrieval beats the
embedding kernel (BAGEL-lite) AND passive at low budget in the deep-multi-hop regime under a real hop-aware
judge, SIGNIFICANT ON BOTH datasets (graph-cosine +0.039 @B=1 each). Plus (i) a clean REGIME BOUNDARY (washes
out at N=10 / shallow bridges / bridge-blind judge / hard-pin design), (ii) a full MECHANISTIC story (structure
as covariance; bridge-blindness of off-the-shelf judges + the graded hop-aware fix; noise-washout; deep-burial;
Bayesian-soft necessity), (iii) an oracle UPPER BOUND. NOT SUPPORTED: an end-task/answer-accuracy headline (QA
n.s. even at n=600). => this is an IR/recall + characterization contribution, NOT a 'GraphRAG beats X on QA'
paper. Framing: SIGIR IR-track / short, or a component of the RSS Discussion Paper's 'where Bayesian structure
earns its keep in retrieval' thesis. Remaining if pursued standalone: BAGEL head-to-head (real, not lite),
MuSiQue as a 3rd dataset, nDCG alongside recall, hierarchical-prior cold-start.
