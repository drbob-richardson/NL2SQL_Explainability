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
