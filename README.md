# NL2SQL Explainability & Uncertainty

**Where does calibrated uncertainty live in text-to-SQL — and which signals actually predict
when a generated query is wrong?**

This repo treats an SQL query as a *typed graph* (tables = nodes, joins = edges, columns =
attributes) and asks, rigorously, where Bayesian structure and semantic signals give *calibrated,
transferable* uncertainty — and, just as importantly, where they do not. Every claim here carries
an execution-grounded number with confidence intervals; negative results are reported as plainly as
positive ones.

## The one-paragraph story

The project began as "Bayesian-nonparametric (BNP) priors over query graphs for NL2SQL uncertainty."
Rigorous ablation reshaped that into a clearer, honest map:

- **Schema-linking uncertainty is real, calibrated, and transfers across schemas.** Per-node
  Bayesian posteriors over the query graph — driven by the database's data dictionary through a
  class-conditional *likelihood-ratio* update — are strong at the table (per-question AUROC 0.86)
  and join-edge (0.85) nodes, moderate at the SELECT-column node (0.74), and hold on *unseen*
  schemas (leave-one-DB-out 0.71–0.86) where a structural frequency prior collapses to chance
  (0.50). The same posterior gives **open-world detection** — "is the right table even in the lake?"
  — at AUROC 0.78.
- **Schema-linking does *not* predict query correctness**; nor do sampling self-consistency or
  execution self-consistency (all plateau at AUROC ≈ 0.62 on hard multi-table BIRD).
- **Correctness UQ needs logic-aware signals.** An LLM *verifier* (0.72; an independent stronger
  judge 0.77) and white-box *log-probabilities* (0.67) break the ceiling, combine to 0.76, and
  enable risk-controlled abstention the baselines structurally cannot offer.
- **The genuinely-BNP win that survives** is the Pitman–Yor *discovery probability* for open-world
  failure (modest on BIRD, ~0.65; it was inflated to 0.84 by the saturated single-table regime).

The full synthesis — theory, every result, and the candidate paper framings — is in
[`paper/master.md`](paper/master.md).

## Two papers in progress

- **Paper 1 — correctness / selective prediction** ("what predicts correctness in text-to-SQL"):
  the verifier + logprob ceiling-break, the abstention frontier, and the rigorous negatives.
- **Paper 2 — [`paper/paper2_schema_linking.md`](paper/paper2_schema_linking.md)**: calibrated,
  transferable Bayesian schema-linking uncertainty + open-world detection for data lakes.

## Repository layout

```
src/bnp_nl2sql/        library: query_graph, pyp (Pitman–Yor), posterior (Model A),
                       fit (empirical Bayes + calibration), calibrate (conformal), execeval,
                       uq_baselines
tests/                 unit tests (40+; query graph, PYP, posterior, fit, execeval, baselines)
scripts/               every experiment (see below); each is safe-by-default for API spend
paper/                 master.md (synthesis), paper2_schema_linking.md, theory.md, methods.md,
                       lit_review.md, manuscript.md, schema_linking_uq.md, tex/ (LaTeX), figures/
server_experiments/    self-contained GPU package (trained verifier + reranking); see its README
data/                  small JSON caches of generations/results (the reproducible record);
                       large files — embeddings.json, SQLite DBs — are gitignored (regenerable)
```

### Key experiment scripts
- Schema-linking posteriors: `bird_table_posterior.py`, `bird_column_posterior.py`,
  `bird_join_posterior.py`; open-world: `bird_openworld.py`; discovery: `bird_discovery.py`.
- Generation + execution + correctness UQ: `bird_generate.py`, `bird_verify.py`,
  `bird_correctness_uq.py`, `bird_exec_uq.py`, `bird_graph_uq.py`, `bird_abstention.py`.
- Structural-BNP line (Spider): `spider_benchmark.py`, `ablate_mechanism.py`,
  `discovery_detection.py`, `logprob_experiment.py`, and others.
- Trained-verifier probe: `verifier_probe.py`.

## Data notes (what's gitignored and how to regenerate)

To keep the repo light, three things are **not** committed:
- `data/embeddings.json` (~300 MB OpenAI embedding cache) — rebuilt on demand by the scripts.
- SQLite databases (`data/spider_db/`, `data/bird/db/`) — Spider DBs from `premai-io/spider`;
  BIRD DBs from the official `dev.zip` (`bird-bench.github.io`).
- The DataCamp PDF and any third-party material.

The small JSON caches of model generations and computed signals *are* committed, so most analyses
re-run without spending on the API.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pytest tests/ -q          # or run individual test files
```

Experiments that call the OpenAI API are **safe-by-default**: they print a cost estimate and make
no calls without `--run`, with a hard `--max-calls` cap and on-disk caching. Set `OPENAI_API_KEY`
in the environment (never commit it).

## Status & honesty

Total OpenAI spend to date for all results here ≈ **\$2.65**. The discipline throughout has been to
validate each hypothesis with a cheap experiment before scaling — which repeatedly caught artifacts
(a saturation-driven AURC win, cross-schema non-transfer, a null correctness signal) before they
became claims. See `paper/master.md` §9–§11 for the consolidated limitations and the candidate
directions (including the trained-verifier and verification-guided-generation bets the
`server_experiments/` package is built to test).
