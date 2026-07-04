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
