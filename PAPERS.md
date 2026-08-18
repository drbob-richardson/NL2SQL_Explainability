# Papers index

Five papers live in this repo. This index maps each to its venue, status, and files so nothing gets
lost across the `paper/`, `paper/writeup/`, `paper/tex/`, `paper-overleaf*/`, and `scripts/` locations.
(Last updated 2026-08-18.)

| # | Short name | Venue | Status | Deadline |
|---|---|---|---|---|
| 1 | **Text2SQL UQ** — selective prediction / SQL-correctness verifiers | **TMLR** | In revision (3 reviews: 2 addressable + 1 positive) | rolling |
| 2 | **Bayes Schema Subgraph** — hierarchical autologistic schema-subset selection | **JASA A&CS** | Submitted (reframed after Bayesian Analysis reject; AoAS backup) | — |
| 3 | **How can Bayes help retrieval** — broad thesis-driven synthesis | **JRSS (invited Discussion/Read Paper)** | Drafting; cites #2, #4 rather than reproducing | **20 Nov 2026** |
| 4 | **GraphRAG "Paper A"** — structure-as-covariance active retrieval | **AISTATS** | Draft firmed (BAGEL head-to-head, alignment law); review items addressed | — |
| 5 | **"Paper B"** — structural de-aliasing under differential misclassification | **JASA T&M** | Theory near-complete (rate + field minimax + singular local regime) | — |

## Where each paper's files are

### 1. Text2SQL UQ  →  TMLR
- **Paper:** `paper/tex/paper1_correctness.tex` (local) · `paper-overleaf/` (Overleaf sync clone)
- **Plan / revision:** `paper/paper1_correctness_uq.md` · `paper-overleaf/REVISION_LOG.md`
- **Scripts:** `scripts/bird_*.py`, `scripts/paper1_*.py`, `scripts/verifier_probe.py`; `server_experiments/exp*.py`
- **Open (GPU):** Table-4 CIs (reviewer #8) — prepped in `server_experiments/RUN_TABLE4_CIS.md`; handoff `server_experiments/NOTE_FOR_SERVER_CLAUDE.md`
- **Thesis:** what predicts SQL execution-correctness is *verification* (reasoning judges), not black-box statistical UQ; independent-provider judges ensemble to AUROC 0.82; trained verifiers overfit schemas and don't transfer.

### 2. Bayes Schema Subgraph  →  JASA A&CS
- **Paper:** `paper-overleaf-subgraph/` (Overleaf clone — do not move)
- **Plans:** `paper/bayes_subgraph_stats_plan.md`, `paper/schema_linking_uq.md`, `paper/paper2_*.md`
- **Scripts:** `scripts/bayes_subgraph_*.py`, `scripts/bayes_schema_*.py`, `scripts/ambrosia_*.py`
- **Thesis:** hierarchical autologistic schema-subset selection + asymmetric-cost decision rule (containment set, coverage guarantee); scales to 97-table BEAVER.

### 3. How can Bayes help retrieval  →  JRSS Discussion (Read) Paper  *(due 20 Nov 2026)*
- **Paper:** `paper/tex/tas_bayes_ir.tex` (+ `tas_refs.bib`) · `paper-overleaf-tas/` (Overleaf clone — do not move)
- **Plan:** `paper/tas_bayes_ir.md`
- **Note:** reframe from "retrieval audit" to a thesis-driven Read Paper; **cite, don't reproduce** #2 (BEAVER) and #4. May wait until #4/#5 firm up.

### 4. GraphRAG "Paper A"  →  AISTATS
- **Paper:** `paper/writeup/paperA_submission.tex` (+ `fig_alignment.pdf`, `fig_bagel.pdf`); theorem note `paper/writeup/paperA_alignment_theorem.tex`
- **Plan / running log:** `paper/active_retrieval_plan.md`
- **Scripts:** `scripts/graphrag_*.py`, `scripts/paperA_*.py`, `scripts/musique_*.py`
- **Parked (GPU):** N=500 larger-pool BAGEL runs (fairness bulletproofing) — not urgent
- **Thesis:** the corpus graph is useless for ranking but load-bearing as the GP *covariance* (semantics sets the mean, structure sets the dependence); beats a faithful BAGEL at matched budget, with a budget-invariant covariance gap.

### 5. "Paper B"  →  JASA T&M
- **Paper (submission):** `paper/writeup/paperB_JASA_skeleton.tex` + proofs `paper/writeup/paperB_JASA_proofs.tex`; working doc `paper/writeup/paperB2_structural_dealiasing.tex`
- **Plans:** `paper/paper2_bnp_decision_exploration.md`, `paper/paper2_theorem1_sketch.md`
- **Scripts:** `scripts/paperB_*.py` (rate / branch / field / unknown / lan / subtree / boundary / lowerbound sims)
- **Thesis:** when a biased oracle's errors mimic true negatives, measurements are structurally silent about the bias, so the correction must come from inter-item dependence; near-aliasing minimax rate + field minimax + singular n^{-1/4} local regime.

## Repo layout notes
- `paper-overleaf/`, `paper-overleaf-subgraph/`, `paper-overleaf-tas/` are **separate Overleaf git-bridge clones** (own `.git`). Edit/commit inside them; don't relocate.
- `scripts/` (~170 files) share `data/` and cross-import; classify by the prefixes above.
- `server_experiments/` = GPU jobs for #1 (trained-verifier experiments).
- Legacy / unclassified: `paper/tex/paper.tex`, `paper/tex/recovery_theorem.tex` (verify before reusing).
