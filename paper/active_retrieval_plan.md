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
REFEREE FOLLOW-UP (graphrag_ccvoi.py, $0): the earlier EVOI minimized omitted-MASS (sum p_j); the GENUINE
one-step VOI for chain completion minimizes P(fail)=1-prod(1-p_j). Tested both vs UCB on the NORMALIZED graph
kernel: graph-ccVOI (true completion-VOI) trails graph-UCB by -0.078/-0.102/-0.113 chain-completion @B=1/2/3
(sig), ~tying the omitted-mass surrogate. => NOT 'a bad surrogate backfires' -- even the correct one-step
Bayes-risk-reduction loses, because the value of graph exploration is MULTI-STAGE (judge anchor now, propagation
pays off at the next decision). Worth theory (one-step VOI vs optimal sequential/POMDP policy). Strengthens the
acquisition claim; folded into paperA writeup.

## LEARNED GOLD-FREE lambda_q GATE (graphrag_lambda_learn.py, $0) -- honest NEGATIVE on chained-only data
5-fold CV; gold-free features (top-k cosine, burial gap, cos spread, density, anchor degree, budget) -> ridge
predicts per-query graph advantage -> gate lambda_q. OUT-OF-SAMPLE chain completion: learned lambda_q TIES
fixed-graph (-0.005 @B=1/2/3, CI ~[-.012,0]); captured ~none of the oracle headroom (itself small + budget-
inconsistent: oracle-graph +0.042 @B=2 but -0.037 @B=1). WHY: on chained-only data the graph helps on ~86% of
queries, so 'always graph' is near-optimal and there is almost nothing to route. Features DO carry sensible
signal (topk_cos NEGATIVELY predicts graph advantage -- a confident prior => graph adds less). => the lambda_q
payoff needs a MIXED query distribution including INDEPENDENT (comparison) questions where the graph HURTS
(lambda->0); those are not judged yet (small API). HONEST CLAIM: gold-free features predict graph advantage,
but chained-only data lacks the graph-unfavorable queries to demonstrate routing value; do NOT claim a
learned-lambda_q win until the mixed distribution is run.

## NON-MYOPIC + EXPLORATION ACQUISITION (graphrag_lookahead.py, $0) -- UCB uniquely best; value = FOCUSED exploration
Review #3's top A move: is the UCB>1-step-VOI gap about DELAYED value (non-myopia)? Raced on the normalized graph
kernel (chain completion), vs UCB @B=1/2/3: 1-step VOI -0.080/-0.103/-0.113; **2-STEP lookahead -0.078/-0.108/
-0.143 (NO horizon help -- REFUTES 'delayed value')**; pure INFO-GAIN (max variance reduction) -0.087/-0.103/
-0.108 (not pure information either); pure EXPLORATION (max-var) -0.048/-0.070/-0.092 (CLOSEST but still below).
=> the value is NOT horizon (2-step fails), NOT pure info (infogain fails), NOT pure exploration (maxvar closest).
It's UCB's mean+sd BALANCE: the mean term FOCUSES exploration on high-prior nodes = the anchors whose judgment
PROPAGATES through the graph to buried bridges; maxvar wanders to graph-peripheral nodes (no propagation);
VOI/infogain chase the decision boundary / raw uncertainty. CLEAN CHARACTERIZATION: across VOI (1&2-step),
info-gain, and pure exploration, none beats UCB -- structural information is realized by FOCUSED optimistic
exploration, not decision-theoretic acquisition at any horizon. Corrected paperA (the old 'multi-stage value'
intuition was wrong). Still-owed analytic centerpiece = the TOY THEOREM (anchor in top-k, buried bridge, distractor:
a regime where 1-step VOI gives ~0 value to judging the anchor but its observation raises later P(retrieve bridge)).

## HETEROPHILIC CHAIN-IDENTIFICATION SIM (paperB_identify_sim.py, $0) -- mechanism CONFIRMED, but a real tension
3-state role-HMM (irrelevant/bridge/direct), heterophilic transition (low diagonal); recover emission by
Baum-Welch (CHAIN, uses dependence) vs mixture-EM (I.I.D., dependence removed). Min-perm emission error, best-of-4:
  SEPARATED emissions: chain-EM 0.07-0.09 vs i.i.d. 0.70 -- CLEAN ~10x gap => the 'heterophilic DEPENDENCE (not
    replication) identifies the channel' mechanism is CONFIRMED; a single grade-2-heavy anchor NAMES the roles.
  BRIDGE-BLIND emissions (the ACTUAL phenomenon): chain-EM ~0.20 (noisy, ~ i.i.d.) -- WEAKLY identified, because
    bridge-blindness MEANS the bridge emission overlaps the irrelevant emission (both mass on grade 0) = poor
    emission separation, exactly the condition the theorem needs. LEN 5->15 did NOT fix it (separation, not chain
    length, is binding -- flips the earlier 'asymptotic in chain length' guess).
=> DEEP TENSION: bridge-blindness is partly SELF-OBSCURING -- the more the judge conflates bridge with irrelevant,
the harder to identify the bias FROM GRADES. LIKELY RESOLUTION (refines the theorem): the bridge ROLE is
identified by GRAPH POSITION (connected to a direct/anchor), not by grades; and the DELIVERABLE (relevance
correction Pr(r|g,A)) may work via the graph/Potts prior + anchors even with a fuzzy emission Pi. SPLIT THE
THEOREM: (a) role-field identification (graph/transition-driven, plausibly OK) vs (b) emission/bias-MAGNITUDE
identification (weak in the bridge-blind regime). NEXT make-or-break for B: simulate the FULL model (roles on a
graph + Potts prior + bridge-blind emission + anchors) and test whether the posterior correctly FLIPS bridge
grades to relevant -- the actual deliverable -- even when Pi is only weakly identified. If yes, B works (graph
carries the correction); if no, B's central claim is in real trouble.

## FULL-MODEL RELEVANCE-CORRECTION SIM (paperB_correction_sim.py, $0) -- B STANDS, on a STRONGER thesis
Roles on a graph (anchored relevant cluster + ANCHORLESS distractor clusters) + relevance-Potts prior + bridge-
blind emission + low-prior bridges; Gibbs -> posterior Pr(r|g,A). Sweep coupling theta:
  theta=0 (NO graph): bridge recall 0.48->0.14 (WORSE than raw judge), AUC 0.69 -- correction FAILS without graph.
  theta=1: bridge 0.48->0.64, distractor FP 0.049, direct recall 0.96, AUC 0.965 -- WORKS (flips bridges, FP low).
  theta=2: bridge 0.85, FP 0.078 (mild over-smoothing); theta=3 collapses (over-coupled, dir recall 0.35).
  ROBUSTNESS: a MISSPECIFIED 'rough' emission (doesn't know the true bridge-blindness) STILL works -- theta=1
    bridge 0.50/AUC 0.974, theta=2 bridge 0.71/AUC 0.921 ~ true-emission => the deliverable does NOT need a
    cleanly-identified emission.
=> RESOLVES the tension. The GRAPH carries the correction, ROBUSTLY, and is NECESSARY (theta=0 fails). The weakly-
identified emission (prior sim) is irrelevant to the deliverable. SPLIT-THE-THEOREM VALIDATED: (a) relevance-
correction identification = graph-driven, works; (b) emission-magnitude = weak but doesn't matter. **B's thesis
SHARPENS (stronger + more distinctive): 'when a biased oracle's errors mimic true negatives (bridge-blindness),
the measurements are structurally SILENT about the bias, so the correction MUST come from inter-item dependence --
structure is NECESSARY, not merely helpful.'** CAVEATS (sim with favorable structure; theta sweet-spot ~1-2;
reliable anchors). NEXT: validate on REAL data (cached Hotpot/MuSiQue judge labels + graphs + gold) -- do graph-
corrected posteriors beat the raw judge on held-out gold on actual corpora? + principled theta selection (CV on
gold / prior over theta). B is back on track, on a better thesis than the original.

## REAL-DATA CORRECTION VALIDATION (paperB_realdata_correction.py + _2.py, $0) -- the sim does NOT survive real data
On real Hotpot/2Wiki pools (real hop-aware judge grades, real title graph, real gold) the graph correction is
MARGINAL. Symmetric-Potts Pr(r|g,A): AUC 0.954 ranking but the GRAPH (theta>0) HURTS AUC (0.954->0.81); the gain
is from the PRIOR (theta=0). Diagnosis: gold neighbourhoods are BALANCED (0.75 gold vs 0.72 distractor) but SPARSE
(~1.5 degree); only 35% of judge-MISSED bridges have a confident grade-2 neighbour = a graph reachability ceiling.
Directed anchor-diffusion fix: AUC(missed-gold vs distractor) prior 0.875 / diffusion 0.600 / prior+diff 0.882
(+0.007); AUC(all gold) raw-grade 0.876 / prior 0.941 / prior+diff 0.968 (+0.027 over prior). => the CALIBRATED
RETRIEVER PRIOR does the relevance-recovery (missed bridges have higher cosine than distractors); the graph adds
only a small increment. HONEST: the biased judge CAN be corrected (AUC 0.876->0.968) but the correction is
DOMINATED BY THE PRIOR, not the structural/graph component. **Paper B's distinctive 'structure is NECESSARY to
correct the judge' claim (from the favorable sim) does NOT hold on real data -- the practical correction is
'trust the calibrated retriever over the raw judge grade,' which needs neither the measurement model nor the
graph.** The sim assumed dense anchored clusters; real title graphs are sparse + low-reachability.
IMPLICATION for B: it SHRINKS from a structural-identification STATS theorem toward an EMPIRICAL finding
(bridge-blindness is real + differential + a prior-based correction beats the raw judge) -- ACL-Findings-tier, not
a JASA theorem. To salvage the stats version: find a NO-PRIOR setting (items without a good retriever embedding,
where structure is the ONLY correction signal), OR a denser/better graph that raises anchor-reachability, OR
reframe B entirely around the (real, robust) bridge-blindness characterization + the honest 'prior beats the
biased judge' correction. This is a genuine strategic inflection for the two-paper plan.
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
DECOMPOSITION graph (musique_decomp_graph.py, $0.01): LLM decomposes Q into sub-qs (mean 3.55), assign passages
to hops by cosine, connect top-2 across hops. Achieves BOTH target properties -- gold-conn 0.66/0.78 (3/4-hop, up
from 0.23/0.53 title) AND density 0.0022 (near-oracle sparse) -- BUT margin STILL ~0 (2hop -0.005, 3hop +0.008).
WHY: the decomposition (hops) is good, but MATCHING sub-qs to passages relies on COSINE, which MuSiQue defeats ->
top-2 per sub-q pulls in distractors -> ~80% of edges wrong -> propagation doesn't discriminate.

## FINAL CHARACTERIZATION (MuSiQue investigation complete)
The bottleneck is chain-IDENTIFICATION via surface similarity, which MuSiQue adversarially defeats. Cheap signals
(cosine / entity / decomposition+cosine-match) ALL fail; only the oracle (knows golds) or expensive deep per-passage
reading (= the judge budget) recovers the chain. The MECHANISM IS SOUND (oracle +0.07-0.09, both 2&3-hop); cheap
APPLICABILITY is bounded by a measurable CORPUS property: **structure-as-covariance helps cheaply IFF the corpus's
surface structure aligns with the reasoning chain** (Hotpot/2Wiki yes -- titles mention each other; MuSiQue no --
distractors made surface-similar to golds by design). This is the paper's BOUNDARY LAW + an honest limit, and it
makes MuSiQue a FEATURE (the boundary case that reveals the alignment requirement + the oracle ceiling), not a
failed experiment. (Possible expensive future refinement: cross-encoder or LLM passage-scoring for the sub-q match
-- but that abandons the cheap-structural-prior premise; likely still partly defeated by MuSiQue's design.)

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

## MIXED-DISTRIBUTION ROUTING + COMPARISON-QUESTION CONTROL (Paper A firm-up #1)  [scripts/graphrag_lambda_mixed.py, graphrag_judge_comparison.py]
Judged the INDEPENDENT (comparison) N=100 pools with the same hop-aware judge (30k calls, $1.26; judge recall on
gold 0.841) so the alignment law and lambda_q routing can run on a REAL mixed distribution (300 chained + 300
comparison, both datasets), not chained-only.
(A) ALIGNMENT LAW ON A MIXED DISTRIBUTION -- firmed. graph-cosine recall@k:
   ORACLE judge: CHAINED +0.045..+0.059 (sig); COMPARISON -0.008..+0.017 (n.s., slightly neg @B=2/3). Sharp boundary.
   REAL judge:   CHAINED +0.033[+.007,+.061]@B=1 (sig), +0.020@B=2 (n.s.); COMPARISON +0.012@B=1 / +0.025@B=2(sig)/-0.004@B=3.
   => the graph helps chained and is NEUTRAL-to-slightly-helpful on comparison under a real judge (never hurts).
      Mechanism visible: comparison Qs have a STRONG prior (passive recall 0.79-0.83, both entities directly
      findable, no bridge to bury), so there is nothing to propagate. This is the both-sides alignment-law result.
(B) ROUTING -- honest split. Reframed the lambda_q decision from KERNEL (cosine<->graph, ~null: graph is neutral
   not harmful on comparison so 'always graph' is near-optimal) to the meaningful EXPLORATION decision (spend
   budget with the graph-GP vs trust the prior/passive):
   ORACLE judge: learned gold-free gate BEATS BOTH fixed policies @B=2: +0.013[+.003,+.024] vs always-graph,
     +0.027[+.015,+.040] vs passive (both sig). Routes graph on 0.66 chained vs 0.53 comparison.
   REAL judge:   learned gate TIES always-graph (-0.004@B=2 n.s.); ties passive. BUT the ORACLE gate has +0.040
     headroom (0.729 vs 0.689 @B=2) -- the routing SIGNAL exists, it just is not capturable gold-free under judge
     noise on this data. Same oracle-win / real-judge-washout pattern as the core GraphRAG arc.
   => HONEST takeaway: under a realistic judge the graph is a SAFE neutral-to-helpful default across query types,
      so adaptive gating is NOT needed for deployment (always-graph is near-optimal); the adaptive routing is an
      ORACLE-only mechanism (headroom real, gold-free predictor insufficient). Robustness positive, not a routing win.
NET for the paper: the alignment law is now demonstrated on a real MIXED distribution (comparison control) under a
real judge -- a genuine firm-up of Sec 'When does structure help'. The lambda_q 'learns when to use structure'
contribution stays PROPOSED/oracle-only; the deployable message is 'graph is a safe default, no gate needed'.
Total extra spend: $1.26.

## ALIGNMENT-LAW THEOREM (Paper A firm-up #2, $0)  [paper/writeup/paperA_alignment_theorem.tex, scripts/paperA_alignment_sim.py]
Turned the measured 'structure helps iff graph-chain alignment' into a PROVED result (elevates A from mechanism to
mechanism+law). Model: GP with semantic mean + correlation-form kernel; canonical buried-bridge instance (gold =
{findable anchor, buried bridge}); SBM graph (within-chain edge prob p, off-chain q; alignment = p-q).
  LEMMA (exact surfacing): judging the UCB-first anchor a, the correlation-form posterior mean is
    mu_i = m_i + K_ia/(1+sigma^2)*(y_a - m_a), so the bridge b surfaces into top-k IFF the KERNEL DIFFERENTIAL
    K_ba - max_d K_da > tau = (1+sigma^2)(max_d m_d - m_b)/(y_a - m_a) [burial threshold].
  THEOREM (alignment law): under the SBM, (i) E[differential] is strictly increasing in p-q, leading term
    beta(p-q), zero at p=q (one-hop kernel exact; Katz monotone same-sign); (ii) the embedding kernel has
    differential <=0 for a buried bridge (never surfaces it), so the graph gain Delta(p,q) is 0 at p=q, POSITIVE
    IFF p>q, monotone increasing in p-q; (iii) q->0,p->1 = oracle-clique ceiling, p~q = MuSiQue boundary.
  VERIFIED (paperA_alignment_sim.py, N=30, Katz kernel, B=1): kernel differential -0.001->0.113 and graph
    advantage over embedding +0.007->+0.115 as p-q:0->0.95, monotone, ~0 at p=q; embedding flat at chance 0.500.
  Two remarks bind the paper together: (a) UCB>VOI -- the anchor is judged first BECAUSE UCB weights the prior
    mean; its value is the DOWNSTREAM surfacing (non-myopic), not its own label -> grounds the acquisition result;
    (b) gold-connectivity is the measurable estimate of p-q, explaining why one scalar predicts the gain + why the
    oracle ceiling and MuSiQue failure are the two ends of one axis + why the effect is bounded (saturates).
  => the alignment-law SECTION now has a theorem, the strongest single lift for an AISTATS submission (theory-
     rewarding venue); no experiments/tokens needed. paperA_alignment_theorem.tex compiles (2pp).

## LLM HOP-ASSIGNMENT GRAPH ON MuSiQue (Paper A firm-up #3, the swing)  [scripts/musique_hopassign_graph.py]
Theorem prescription: MuSiQue fails because cheap graphs aren't chain-assortative; build a better graph by
replacing the cosine sub-q->passage matching (that MuSiQue defeats) with the LLM's OWN hop-assignment (ask which
sub-question each pool passage answers, or none). 483 questions x 100 passages judged ($1.65 incl pilot; assign).
FULL-SET result (n=200/200/83 at 2/3/4-hop), rec-margin@B2 graph-cosine w/ 95% CI:
   cosine-decomp:  2h -0.005 / 3h +0.008[-.02,+.03] / 4h +0.021[-.02,+.06]   (null everywhere)
   LLM hop-assign: 2h -0.003 / 3h +0.035[+.01,+.06] / 4h +0.021[-.01,+.06]   (3-hop SIG)
   ORACLE clique:  2h +0.068 / 3h +0.087[+.05,+.13] / 4h +0.148[+.08,+.22]   (ceiling)
=> HONEST: a SIGNIFICANT but MODEST +0.035 recovery at 3-hop (deep-multi-hop regime), ~4x the cosine graph
   (null) and ~40% of the oracle 3-hop headroom; NULL at 2-hop (strong prior, shallow) and 4-hop (n=83, noisy).
   PILOT OVERSTATED (+0.107 at n=25 -> +0.035 at n=200): the discipline of scaling caught a small-n false signal.
   WRINKLE: on the full set the LLM graph's GLOBAL p-q (0.216) is LOWER than cosine-decomp (0.272) yet it wins at
   3-hop -> it helps via BETTER-PLACED edges (connecting the buried golds specifically), not higher global
   assortativity; the simple p-q proxy is muddier than the theorem's bridge-anchor-specific alignment. Assignments
   are cached -> alternative graph constructions (adjacent-hop-only, confidence-thresholded) are $0 to try.
PAPER TAKEAWAY: MuSiQue is PARTIALLY rescuable -- inferring structure from the judge's own hop-reasoning recovers a
significant fraction of the oracle gain at 3-hop where every cheap surface graph is null. A proof-of-concept for
the alignment-law prescription + the adaptive-structure-learning direction (the bridge to Paper B), not a knockout.

## WHY THE ROUTING GATE FAILS -- negative-result investigation (Paper A, $0)  [scripts/paperA_negative_analysis.py]
Dissected why the deployable lambda_q / exploration gate ties always-graph under the real judge. Clean answer:
  (1) SNR WALL: the per-query graph advantage (graph-cosine recall@2) is a SMALL MEAN effect swamped by LARGE
      per-query variance -- pooled mean +0.023, sd 0.198 => SNR mean/sd = 0.12. recall@k is discrete/coarse;
      whether the buried bridge surfaces on a given query is idiosyncratic (exact cosine ranks, boundary
      distractors, realized edges). The average is a stable small +, the per-query realization is ~+-0.2 noise.
  (2) NEARLY UNPREDICTABLE, EVEN UNDER ORACLE: gold-free ridge R^2 of the advantage = 0.034 (6 generic feats) and
      only 0.052 with THEORY-MOTIVATED feats (graph-Laplacian prior roughness, anchor->buried reachability,
      edge prior-gap 'bridging potential'); corr(adv, reachability) -0.02, corr(adv, bridging-pot) -0.03. So it is
      NOT a feature-engineering failure -- the per-query advantage is not a learnable function of cheap features.
  (3) MY MECHANISTIC FIXES REFUTED (honest): H1 judge-error amplification is NOT the mechanism (corr(adv_real,
      anchor reliability) = -0.17, opposite sign; graph helps MORE with a mislabeled anchor, not less); H3
      confidence-gated propagation (only grade==2 anchors propagate) HURTS -0.030[-0.044,-0.017] -- the grade-1
      'related but not clearly needed' labels carry useful propagation signal, so the soft graded design is right.
PRINCIPLED TAKEAWAY (the paper's answer to 'why the negative'): the structural gain is a DISTRIBUTIONAL property --
a robust small AVERAGE win in the deep-multi-hop/chained regime (the alignment law) -- NOT a per-query-predictable
signal. Per-query adaptive routing is infeasible (SNR ~0.12, R^2 ~0.05); the graph is a SAFE default (neutral-to-
helpful, never sig. hurts under a real judge); so 'ALWAYS USE THE GRAPH' is the correct, simple deployment and the
adaptive-lambda_q machinery is unnecessary. The DATA CHARACTERISTIC that predicts the gain is the REGIME (hop-depth
/ chain structure = the alignment law), not any per-query feature. This converts the routing negative into a
principled design recommendation + closes the 'adaptivity' open item honestly.

## KERNEL-NORMALIZATION CORRECTION + NORMALIZED HEADLINE (Paper A, overnight)  [paperA_metrics.py, paperA_routing_normalized.py]
CAUGHT during the write-up: all this session's analysis (mixed alignment #1, negative-result, metrics) used the
RAW kernel kern_graph=(I+lam L)^{-1}, but the paper's METHOD is the NORMALIZED correlation-form kernel
(unit-diagonal) -- so every session number UNDERSTATED the effect by ~2x (that IS the normalization finding).
Recomputed with normalized kernels (the correct method):
  HEADLINE (chained N=100 real judge, correlation-form): recall +0.058[.040,.076] / nDCG@10 +0.030[.020,.041] /
    completion +0.122[.088,.155] @B=1; +0.071/.032/.150 @B=2; +0.063/.033/.127 @B=3 -- SIGNIFICANT AT EVERY
    BUDGET (raw converged to null by B=3). Normalization ablation: completion +0.068 raw -> +0.122 norm (1.8x).
  ALIGNMENT LAW SHARPER with normalized kernels: graph HELPS chained (+0.058 sig) and does NOT help comparison
    (metrics -0.026 sig / routing-recompute ~0; sign is prior-calibration-sensitive -> reported conservatively as
    'neutral to slightly negative, <=0'). Cleaner both-sides boundary than the raw-kernel neutral.
  ROUTING NEGATIVE HOLDS under normalized kernels too: learned gold-free gate only TIES always-graph
    (+0.003[-0.004,+0.009] n.s.); oracle-regime routing +0.000; oracle PER-QUERY +0.027[.018,.036] headroom exists
    but is NOT gold-free-reachable. So the SNR/predictability wall + the 'per-query routing infeasible' conclusion
    are ROBUST to the (correct) kernel; the negative is real, not a raw-kernel artifact. Note: because normalized
    graph slightly penalizes comparison, 'always-graph' is scoped to the multi-hop regime it targets (not a
    universal safe default), but per-query gating still can't capture the small penalty.
DRAFT (paperA_submission.tex) updated with all correct normalized numbers; fig_alignment.pdf regenerated. This
correction STRENGTHENS the paper (bigger, all-budget-significant headline; sharper alignment law) while the
routing negative survives -- exactly the honesty we want before submission.

## REVIEW ROUND: THEOREM REPAIR + figure + honesty fixes (Paper A)  [paperA_alignment_sim.py, paperA_assortativity.py]
Reviewer caught Theorem 1 is FALSE as written: E[K_ba - max_d K_da] = beta[p-1+(1-q)^|D|] is NOT zero at p=q
(the max-over-distractors adds a penalty), so 'positive iff p>q' is wrong. Adopted the reviewer's cleaner correct
theorem (verified paperA_alignment_sim.py, |D|=10, q=.05):
  ONE-HOP EXACT: bridge surfaces iff A_ba=1 AND A_da=0 forall d => P(surface)=p(1-q)^|D| (empirical matches).
  ALIGNMENT EXCESS vs density-matched unaligned graph (chain edge also at q): Delta=(p-q)(1-q)^|D| -- ZERO at p=q,
    >0 iff p>q, LINEAR in p-q (matches). The RAW surfacing is q(1-q)^|D|>0 at p=q (graph helps by luck); the
    EXCESS is what vanishes -- the statistically correct claim, and explains the small nonzero synthetic gain.
  ACTUAL GMRF kernel (not the proxy corr(I+beta A)): (I+lam L)^{-1}=I-lam L+O(lam^2) => K_ij=lam A_ij+O(lam^2),
    corr-normalization preserves it => E[K_ba-K_da]=lam(p-q)+O(lam^2). Rigorous first-order alignment for the REAL
    kernel (verified, slope ~lam). Dropped the bogus 'diffusion paths preserve sign/monotonicity' hand-wave.
Also: (a) FIGURE 1 rebuilt -- panel A = corrected excess-vanishes-at-p=q; panel B = empirical DOSE-RESPONSE, real
graph-cos gain vs empirical ASSORTATIVITY p_hat-q_hat (not gold-connectivity, which is p_hat only) across
datasets: Hotpot-ch (p-q=0.748, +0.053), 2Wiki-ch (0.588, +0.016), comparison (~0, ~0/neg), MuSiQue cosine
(0.272,+0.008)/hop-assign(0.216,+0.035)/oracle(1.0,+0.087). Rough positive dose-response; hop-assign slightly
exceeds its global assortativity (better-PLACED edges -- honest wrinkle). (b) UCB story SOFTENED+HONEST: mechanism
(delayed value undervalued by 1-step VOI) explains 1-step; but 2-step lookahead ALSO loses => 'why optimism beats
short-horizon lookahead is an empirical finding we do not fully explain' (open); NOT a theorem corollary. (c)
MuSiQue BUDGET-FAIRNESS made explicit: hop-assignment uses EXTRA calls outside budget B -> labeled a
structure-CONSTRUCTION experiment (not budget-fair competitor); the fair version = elicit (grade, role) in the
SAME judge call (the adaptive sequel). (d) Wording: 'safe (never sig hurts)' -> 'did not sig hurt in the regimes
tested'; 'always use the graph' -> 'global graph policy better supported than per-query routing in the multi-hop
regime'; gold-connectivity -> assortativity p_hat-q_hat. Draft now 7pp, compiles. Theorem soundness = fixed.
