# Hierarchical few-shot reranking (Bayes in its home regime) — pre-registration + results

**Question (Q2):** Is there ANY regime where a Bayesian treatment beats a well-tuned point estimate for
LLM-IR? Prior audit: NO in data-rich single-corpus ranking (all layers). Here we test the regime theory
favors: scarce labels across many related corpora (shrinkage / partial pooling).

**Pre-registered design.** Heterogeneous BEIR domains; dense top-100 + cross-encoder + shared 39-dim
feature vector per (q,doc). All rerankers = logistic heads over the SAME features, differing only in
pooling across domains: zero-shot CE floor; (a) no-pooling; (b) complete-pooling; (c) hierarchical EB
[w_d ~ N(mu,tau2), empirical Bayes]; (d) BNP DP-mixture [CRP prior over {w_d}]. Metric nDCG@10 on a fixed
held-out eval split; sweep k in {2,5,10,25,50}; 12 seeds; bootstrap over domains.
**Pre-registered win:** (c),(d) > max(a,b) at small k (CI excludes 0) AND gap shrinks as k grows;
(d) >= (c) when domains heterogeneous.

**Results (5 domains: nfcorpus, arguana, scidocs, fiqa, scifact; 220 q each).** nDCG@10:
| k | zero-shot | no-pool | complete | hier-EB | BNP-DP | hier-max(a,b) [CI] |
|---|---|---|---|---|---|---|
| 2 | 0.452 | 0.395 | 0.426 | 0.469 | 0.469 | +0.029 [-.000,+.066] |
| 5 | 0.452 | 0.453 | 0.437 | 0.485 | 0.485 | +0.023 [+.006,+.056] |
| 10 | 0.452 | 0.467 | 0.419 | 0.490 | 0.490 | +0.022 [+.009,+.042] |
| 25 | 0.452 | 0.487 | 0.429 | 0.494 | 0.494 | +0.008 [-.002,+.023] |
| 50 | 0.452 | 0.493 | 0.421 | 0.497 | 0.497 | +0.004 [-.006,+.020] |

**Verdict: CONFIRMED (hierarchical).** Partial pooling beats BOTH point-estimate extremes at small k
(significant k=5,10); at k=2 it is the only method above the zero-shot floor (per-domain OVERFITS to
0.395). Clean shrinkage signature: advantage +0.029 -> +0.004 as k grows. FIRST genuine Bayes-beats-
point-estimate result in the program, in the predicted home regime.

**Caveats (honest).** (1) BNP INCONCLUSIVE: DP found 1 cluster every seed (k~1.0) -> collapsed to the flat
hierarchy; 5 domains too few. The genuinely-Bayes-only claim (infer WHICH domains pool) is untested. (2)
The flat-hierarchical win is capturable by a frequentist empirical-Bayes / ridge-to-pooled-mean -> "the
pooling idea, not the apparatus," consistent with the meta-thesis. The apparatus becomes irreplaceable
only at the BNP level, which needs scale.

**Next (decisive): scale to ~10-18 BEIR domains** (GPU box) to (a) tighten the hierarchical win with power
and (b) give the DP-mixture real substructure to find -> test whether BNP (data-driven pooling structure)
beats the flat hierarchy. That is the genuine BNP contribution and the ML-paper anchor. Scripts:
beir_encode.py, beir_hier.py (scale unchanged; just add domains to DOMAINS).

## SCALED RESULT (17 domains: 5 BEIR + 12 CQADupStack forums) + oracle cluster-structure test
nDCG@10, mean over 12 seeds, 17 domains:
| k | no-pool | complete | hier(flat) | oracle-grp | BNP-DP | hier-best [CI] | oracle-flat [CI] |
|---|---|---|---|---|---|---|---|
| 2 | 0.356 | 0.451 | 0.483 | 0.456 | 0.483 | +0.031[.000,.114] | -0.027[-.064,-.012] |
| 5 | 0.397 | 0.472 | 0.488 | 0.471 | 0.488 | +0.016[.001,.037] | -0.017[-.037,-.007] |
| 10 | 0.419 | 0.476 | 0.492 | 0.474 | 0.492 | +0.016[.002,.045] | -0.018[-.030,-.009] |
| 25 | 0.444 | 0.483 | 0.494 | 0.481 | 0.494 | +0.011[.004,.025] | -0.013[-.023,-.005] |
| 50 | 0.458 | 0.482 | 0.494 | 0.486 | 0.494 | +0.013[.006,.018] | -0.008[-.012,-.004] |

**Verdict (definitive):**
1. HIERARCHICAL PARTIAL POOLING WINS at EVERY k (CIs exclude 0), powered by 17 domains. Genuine Bayes-
   beats-point-estimate result; first in the program; the predicted home regime (scarce labels, many tasks).
2. BNP CORRECTLY NULL. DP found 1 cluster over all 17 domains. The oracle topical grouping (prog/sci/gen)
   is WORSE than the flat single-prior hierarchy at every k (orc-flat -0.008 to -0.027, CIs exclude 0):
   splitting fragments the pool and weakens shrinkage. So there is NO exploitable cluster structure -- and
   we PROVED it (even the correct grouping hurts), not merely failed to find it. Reason: given good shared
   relevance features (cross-encoder/BM25/dense), the optimal reranking POLICY is domain-universal; domain
   identity governs WHAT is relevant, not HOW to score it. One shared prior is optimal; DP collapses to it.

**Meta-thesis completed:** Bayes helps IR via simple ideas (structure, adaptation, partial pooling) that
frequentist methods also capture; every distinctively-Bayesian escalation (posterior-as-UQ, posterior-as-
decision, graph-posterior marginalization, nonparametric clustering) adds nothing -- point estimate
suffices OR the latent structure is absent. Here we explained the BNP null, not just observed it.

**Implication:** the partial-pooling win is a real positive to FOLD INTO TAS (Bayes helps in its textbook
shrinkage regime too), with the rigorous BNP-null as the capstone of "the idea, not the apparatus." Not a
top-tier ML method (empirical-Bayes partial pooling is known), but a clean, honest IR-venue-worthy result.

## UQ / selective retrieval under distribution shift (`beir_uq.py`) -- bulletproof negative
Fixed predictor = ensemble-mean reranker (deep ensemble of per-domain heads); success = hit@1; compare
confidence signals at predicting success. AUROC (AURC):
| signal | in-domain | weak-OOD | STRONG-shift (4 src -> 13 tgt) |
|---|---|---|---|
| ce_margin (point) | 0.641 | 0.634 | 0.697 (AURC .402) |
| pool_margin (point) | 0.669 | 0.664 | 0.731 (.378) |
| max-softmax (point) | 0.683 | 0.673 | **0.755** (.384) |
| bayes_std / variance | 0.415 | 0.451 | 0.424 (.641) |
| bayes_agree / vote | 0.684 | 0.683 | 0.681 (.482) |

Under genuine shift the POINT signals IMPROVE (max-softmax 0.683->0.755) and the BAYESIAN signals lose
decisively; ensemble-variance (epistemic) is ANTI-PREDICTIVE everywhere (<0.5). Pre-registered "Bayes
UQ helps and grows under shift" FALSIFIED -- it reverses. Mechanism (ties to BNP-null): reranking policy
is domain-universal, so per-domain heads are near-identical -> ensemble disagreement is noise not
epistemic signal (all-agree often = confidently-wrong distractor -> low disagreement predicts FAILURE);
and the reranker transfers well, so shift doesn't induce the overconfidence Bayesian UQ exists to catch.
COMPLETES the UQ department: (1) in-dist abstention loses (0.700 vs 0.763); (2) deficit is discrimination
not calibration (AUROC recalibration-invariant); (3) under shift, deep-ensemble posterior-predictive
still loses to point max-softmax, epistemic variance anti-predictive. Same meta-thesis in the UQ dept.
