# Papers index

Five papers live in this repo, now organized into per-paper folders under `paper/`. This index maps each
to its venue, status, and files. (Last updated 2026-08-18.)

| # | Short name | Venue | Status | Deadline | Folder |
|---|---|---|---|---|---|
| 1 | **Text2SQL UQ** — selective prediction / SQL-correctness verifiers | **TMLR** | In revision (3 reviews: 2 addressable + 1 positive) | rolling | `paper/1-text2sql-uq/` |
| 2 | **Bayes Schema Subgraph** — hierarchical autologistic schema-subset selection | **JASA A&CS** | Submitted (reframed after Bayesian Analysis reject; AoAS backup) | — | `paper/2-subgraph/` |
| 3 | **How can Bayes help retrieval** — broad thesis-driven synthesis | **JRSS (invited Discussion/Read Paper)** | Drafting; cites #2, #4 rather than reproducing | **20 Nov 2026** | `paper/3-retrieval-jrss/` |
| 4 | **GraphRAG "Paper A"** — structure-as-covariance active retrieval | **AISTATS** | Draft firmed (BAGEL head-to-head, alignment law); review items addressed | — | `paper/4-graphrag-A/` |
| 5 | **"Paper B"** — structural de-aliasing under differential misclassification | **JASA T&M** | Theory near-complete (rate + field minimax + singular local regime) | — | `paper/5-dealiasing-B/` |

## Where each paper's files are

### 1. Text2SQL UQ  →  TMLR — `paper/1-text2sql-uq/`
- **Paper:** `paper1_correctness.tex` (+ `references.bib`) — *authoritative submission copy is the Overleaf clone `paper-overleaf/`*; `paper.tex` is a legacy alt draft (shares `references.bib`)
- **Plan:** `paper1_correctness_uq.md` · revision log `paper-overleaf/REVISION_LOG.md`
- **Scripts:** `scripts/bird_*.py`, `scripts/paper1_*.py`, `scripts/verifier_probe.py`; `server_experiments/exp*.py`
- **Open (GPU):** Table-4 CIs (reviewer #8) — `server_experiments/RUN_TABLE4_CIS.md`; handoff `server_experiments/NOTE_FOR_SERVER_CLAUDE.md`
- **Thesis:** what predicts SQL correctness is *verification* (reasoning judges), not black-box statistical UQ; independent-provider judges ensemble to AUROC 0.82; trained verifiers overfit schemas and don't transfer.

### 2. Bayes Schema Subgraph  →  JASA A&CS — `paper/2-subgraph/`
- **Paper:** Overleaf clone `paper-overleaf-subgraph/` (do not move); local `recovery_theorem.tex` (graph-coupling recovery theorem)
- **Plans:** `bayes_subgraph_stats_plan.md`, `schema_linking_uq.md`, `paper2_schema_linking.md`, `paper2_bnp_decision_exploration.md`, `paper2_options_and_roadmap.md`, `paper2_theorem1_sketch.md`
- **Scripts:** `scripts/bayes_subgraph_*.py`, `scripts/bayes_schema_*.py`, `scripts/ambrosia_*.py`
- **Thesis:** hierarchical autologistic schema-subset selection + asymmetric-cost decision rule (containment set, coverage guarantee); scales to 97-table BEAVER.

### 3. How can Bayes help retrieval  →  JRSS Discussion (Read) Paper — `paper/3-retrieval-jrss/`  *(due 20 Nov 2026)*
- **Paper:** `tas_bayes_ir.tex` (+ `tas_refs.bib`) — *authoritative copy is the Overleaf clone `paper-overleaf-tas/`*
- **Plans:** `tas_bayes_ir.md`, `research_program.md`, `retrieval_exploration.md`, `hier_fewshot.md`
- **Note:** reframe from "retrieval audit" to a thesis-driven Read Paper; **cite, don't reproduce** #2 (BEAVER) and #4. May wait until #4/#5 firm up.

### 4. GraphRAG "Paper A"  →  AISTATS — `paper/4-graphrag-A/`
- **Paper:** `paperA_submission.tex` (+ `fig_alignment.pdf`, `fig_bagel.pdf`, data `assort_points.json`, `bagel_results.json`); theorem note `paperA_alignment_theorem.tex`; older draft `paperA_active_graph_retrieval.tex`
- **Plan / running log:** `active_retrieval_plan.md`; related `ecir_plan.md`
- **Scripts:** `scripts/graphrag_*.py`, `scripts/paperA_*.py`, `scripts/musique_*.py` — the `paperA_fig_*.py`/`paperA_bagel.py`/`paperA_assortativity.py` write figs+data into this folder
- **Parked (GPU):** N=500 larger-pool BAGEL runs (fairness bulletproofing) — not urgent
- **Thesis:** the corpus graph is useless for ranking but load-bearing as the GP *covariance* (semantics sets the mean, structure sets the dependence); beats a faithful BAGEL at matched budget, with a budget-invariant covariance gap.

### 5. "Paper B"  →  JASA T&M — `paper/5-dealiasing-B/`
- **Paper (submission):** `paperB_JASA_skeleton.tex` + proofs `paperB_JASA_proofs.tex`; working doc `paperB2_structural_dealiasing.tex`; measurement-error framing `paperB_llm_oracle_measurement_error.tex`
- **Plans:** in `paper/2-subgraph/` some `paper2_*` overlap; B-specific exploration lives with the drafts here
- **Scripts:** `scripts/paperB_*.py` (rate / branch / field / unknown / lan / subtree / boundary / lowerbound sims)
- **Thesis:** when a biased oracle's errors mimic true negatives, measurements are structurally silent about the bias, so the correction must come from inter-item dependence; near-aliasing minimax rate + field minimax + singular n^{-1/4} local regime.

## Repo layout notes
- `paper/_shared/` — cross-cutting docs (`CHECKPOINT.md`, `lit_review.md`); `paper/_archive/` — superseded early "BNP-over-query-graph" drafts + the A/B-split rationale.
- `paper/figures/` — **shared** figure outputs (paper1_*, schema_growth, phase_transition_*, reliability_*); `paper1_correctness.tex` references these via `../figures/`.
- (Removed) the old `paper/tex/` and `paper/writeup/` dirs — sources moved into the per-paper folders; only build artifacts remained, so they were deleted.
- `paper-overleaf/`, `paper-overleaf-subgraph/`, `paper-overleaf-tas/` are **separate Overleaf git-bridge clones** (own `.git`) — the authoritative submission copies for #1/#2/#3. Edit/commit inside them; don't relocate (`sync-tas.sh` points at `paper-overleaf-tas/`).
- `archive/` (repo root) — unused reference/export material (course PDF, paper zips); safe to ignore.
- `scripts/` (~170 files) and `data/` (~3 GB) are **shared infrastructure** — the scripts cross-import and read `data/` via one relative path, so they are intentionally NOT split per-project (splitting would silently break pipelines). `server_experiments/` = GPU jobs for #1; `src/bnp_nl2sql` + `tests/` = the package.
- Compile a draft from inside its folder, e.g. `cd paper/4-graphrag-A && pdflatex paperA_submission.tex`.
