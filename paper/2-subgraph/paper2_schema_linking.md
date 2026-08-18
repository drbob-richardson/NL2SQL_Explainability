# Calibrated Bayesian Schema-Linking Uncertainty for Text-to-SQL over Data Lakes

*Paper 2 draft — 2026-06-14. Scope: uncertainty over the query graph's schema-linking decisions
(which tables, joins, and returned columns a question implies), and open-world detection of
whether the needed table is present at all. Query-correctness UQ is the subject of a companion
paper and is explicitly out of scope here.*

---

## Abstract

Deploying text-to-SQL over a **data lake** — many databases, unseen schemas, no per-schema
training — makes *schema linking* (mapping a question to the tables, joins, and columns it needs)
the central uncertainty: the system must know **which** schema elements a question implies and
**whether the right table is present at all**. We cast the SQL query as a typed graph and place a
**Bayesian posterior on each graph element**, driven by the semantic match between the question
and the database's **data dictionary** through a principled class-conditional **likelihood-ratio**
update (not a softmax). The construction is naturally Bayesian-nonparametric: tables are a
species-sampling draw, the projected (SELECT) column set is a feature allocation (Indian-Buffet /
Beta–Bernoulli), and joins are an edge posterior over the foreign-key graph.

On BIRD (1,534 questions, 11 databases with human-written column descriptions) the posteriors are
**calibrated and strong at the table (per-question AUROC 0.86) and join-edge (0.85) nodes, and
moderate at the SELECT-column node (0.74)**. The central property for the data-lake setting is
**cross-schema transfer**: under leave-one-database-out the semantic posterior holds (0.71–0.86),
while a structural frequency prior over the same elements **collapses to chance (0.50)**. As a
deployable consequence we obtain **open-world detection** — flagging when the correct table is
absent from the lake — at **AUROC 0.78**, a capability that question-only or generation-based
signals cannot provide. We also give two honest negatives that scope the contribution: schema
elements are chosen for structural reasons embeddings cannot see (so column recall is limited and
schema *pruning* is unsafe), and schema-linking confidence does **not** predict query
*correctness*. The result is a principled, calibrated, transferable uncertainty layer for the
retrieval side of text-to-SQL.

---

## 1. Introduction

Text-to-SQL is increasingly deployed not against a single, well-known database but against a
**data lake**: hundreds of tables across many schemas, frequently changing, with no opportunity to
train per schema. In this regime the first and most consequential uncertainty is not "is my SQL
syntactically right?" but **"did I even point at the right part of the lake?"** — *schema
linking*. A system that cannot quantify schema-linking uncertainty cannot abstain when the
question refers to data that is not present, cannot ask for help when it is unsure which of two
similar tables is meant, and cannot expose calibrated confidence to a downstream risk process.

We take the view that an SQL query is a **typed graph** — tables are nodes, joins are edges, and
the projected and filtered columns are node attributes — and we ask whether **Bayesian structure
over that graph** yields calibrated, transferable uncertainty about the schema-linking decisions.
Our signal is semantic: the match between the question and the database's **data dictionary**
(human-readable column names, descriptions, and value notes). The contribution is to turn that
match into *principled probabilities* and to characterize, rigorously, where it works.

### Contributions

1. **A per-node Bayesian construction over the query graph** (Section 3): species-sampling for
   tables, a covariate-tilted Indian-Buffet / Beta–Bernoulli process for the projected column
   set, and an edge posterior for joins — each updated from data-dictionary embeddings by a
   class-conditional **likelihood-ratio** rule that yields *calibrated* inclusion probabilities,
   with "logistic-on-cosine" recovered as a special case.
2. **Calibrated, transferable schema-linking uncertainty** on BIRD (Section 5): strong at tables
   (per-q AUROC 0.86) and joins (0.85), moderate at columns (0.74), well-calibrated (ECE
   0.02–0.04), and — crucially for data lakes — **transferring to unseen schemas** where a
   structural prior collapses to chance.
3. **Open-world detection** (Section 6): the retrieval posterior detects when the correct table is
   **absent from the lake** at AUROC 0.78 — a calibrated abstention signal for unanswerable
   questions.
4. **Two scoping negatives** (Section 7): schema-element relevance does not predict query
   *correctness*, and aggressive schema *pruning* is unsafe because many gold elements are
   structural (join keys, filters) with no semantic signature. We state these plainly; they
   delimit what schema-linking UQ is for.

---

## 2. Related work

**Schema linking for text-to-SQL.** Schema linking is a recognized, hard sub-problem; encoder-side
graph models (RAT-SQL, LGESQL) inject schema structure into the representation, and retrieval
methods select relevant tables/columns to fit context limits. These produce point selections; we
produce a *calibrated posterior* over the selection and an *open-world* signal.

**Uncertainty quantification.** Self-consistency / sampling, verbalized confidence, semantic
entropy (Farquhar et al., *Nature* 2024), conformal / selective generation, sub-clause Platt
scaling, and RTS conformal abstention (the nearest baseline) all act on the *generated query* and
flatten its structure. None places a Bayesian posterior over the query graph's schema-linking
elements, and none provides open-world "is the table present?" detection.

**Bayesian nonparametrics.** Pitman–Yor / Dirichlet processes and species sampling; the
Indian-Buffet / Beta–Bernoulli process for feature allocation. We use these as priors over the
*output* graph elements of a generative decoder, tilted by a semantic likelihood — to our
knowledge a new application.

---

## 3. Method

### 3.1 The query graph and the schema-linking decisions

A query `q` over schema `S` induces a typed graph `G(q)`: a set of **tables** `T(q)`, a set of
**join edges** `E(q)` over the foreign-key graph, and a set of **projected columns** `C(q)` (the
SELECT list) together with filter/group columns. Schema linking is the problem of inferring
`T(q), E(q), C(q)` from the question. We model each as a set of binary inclusion decisions and
place a Bayesian posterior on each decision.

### 3.2 The class-conditional likelihood-ratio update

For a candidate element `c` (a table, an edge, or a column) with latent inclusion `z_c ∈ {0,1}`:

- **Prior** `P(z_c = 1) = π`, the base rate that an element is used (estimated from training golds;
  on BIRD the column base rate is `π ≈ 0.113`).
- **Observation** the cosine similarity `s_c = cos(emb(question), emb(dict_c))` between the
  question and the element's data-dictionary text, modelled **class-conditionally**:
  `s_c | z_c = 1 ~ f_1` and `s_c | z_c = 0 ~ f_0`, with `f_1, f_0` Gaussian (QDA; unequal
  variances permitted) fit on training golds.
- **Posterior**

  ```
  P(z_c = 1 | s_c) =  π · f_1(s_c) / [ π · f_1(s_c) + (1 − π) · f_0(s_c) ].
  ```

The embedding enters as a genuine **likelihood ratio** `f_1(s_c)/f_0(s_c)` — a Bayes factor — so
the update is a proper posterior, calibratable and base-rate aware, rather than an arbitrary
softmax over similarities. Equal-variance Gaussians reduce the rule to a logistic link in `s_c`;
thus "logistic-on-cosine" is a *derived* special case.

### 3.3 The per-node BNP construction

Each graph component is a different combinatorial object with a matching nonparametric prior, all
tilted by the same likelihood:

- **Tables — species sampling.** Which entities are present is a draw over a (potentially open) set
  of types; the posterior ranks candidate tables and supports an open-world "new species" event
  (Section 6).
- **Projected columns — feature allocation (IBP / Beta–Bernoulli).** The SELECT list is a
  *subset* of columns; the canonical BNP prior over subsets is the Indian-Buffet / Beta–Bernoulli
  process, here made **covariate-tilted** by the embedding likelihood ratio.
- **Joins — edge posterior.** An edge over the foreign-key graph is included iff *both* endpoint
  tables are relevant; the edge score aggregates the two endpoint posteriors (mean / product beats
  the bottleneck minimum).

### 3.4 Element representation

The dictionary text for a column is `table.column: description` — keeping the `table.column`
identifier sharp and appending the human description; we show (Section 5.3) that verbose
representations dilute the signal. Tables are represented as `table: col, col, …`; questions may be
augmented with the dataset's external **evidence** when available.

### 3.5 Open-world detection

In a lake the correct table may be absent. Under species sampling this is a "the truth is a new
species" event. Operationally, we score the retrieval posterior's **confidence** over the
candidate lake — maximum cosine, softmax peakedness, top-1/top-2 margin, and entropy — and use low
confidence as the open-world flag: if the best available table is only a weak match, the needed
table is probably not present.

---

## 4. Experimental setup

- **Benchmark.** BIRD dev: 1,534 questions over 11 databases, each shipping a **data dictionary**
  (`original_column_name`, human `column_name`, `column_description`, `value_description`) and a
  foreign-key graph; gold-column resolution against the dictionary is 99.6%.
- **Embeddings.** OpenAI `text-embedding-3-small`, cached.
- **Two regimes.** *Parity split* (in-distribution: train on even-indexed questions, test on odd)
  and *leave-one-database-out (LODO)* (fit on 10 databases, test on the held-out one — the
  data-lake reality).
- **Metrics.** Per-question AUROC (are a question's used elements ranked above its unused ones?),
  recall@k (fraction of gold elements in the top-k), and expected calibration error (ECE) of the
  posterior; for open-world, AUROC for detecting a removed gold table.
- **Cost.** Embeddings only (≈ \$0.01); all reported analyses are otherwise compute-only.

---

## 5. Results: calibrated, transferable schema linking

### 5.1 The per-node map (parity / LODO)

| Query-graph node | per-q AUROC | recall@k | ECE |
|---|---|---|---|
| **Table** (which tables) | 0.864 / 0.858 | 0.749 / 0.739 | 0.018 / 0.113 |
| **Join edge** (which FK joins) | 0.861 / 0.852 | 0.617 / 0.631 | 0.019 / 0.033 |
| **SELECT column** (what to return) | 0.740 / 0.746 | 0.309 / 0.315 | 0.007 / 0.037 |

Tables and joins are strong and calibrated; an edge inherits the table node's strength because it
requires both endpoints relevant. The SELECT-column node *ranks* a question's columns well (0.74)
but cannot pin the exact set (recall@k 0.31): columns chosen for structural reasons — join keys,
identifiers, filters — carry no semantic match. Restricting the target to *returned* columns
raises the signal relative to all referenced columns (per-q 0.67 → 0.74).

### 5.2 Cross-schema transfer (the data-lake property)

For the edge node we compare the semantic posterior with a **structural** prior — the marginal
foreign-key edge-usage frequency:

| edge signal | AUROC (parity) | AUROC (LODO, unseen schema) |
|---|---|---|
| structural FK-usage frequency | 0.741 | **0.500 (chance)** |
| semantic posterior | 0.727 | **0.708** |
| combined | 0.815 | 0.708 |

In-distribution the two are comparable and complementary (combined 0.82); **out of schema the
structural prior collapses to chance** while the semantic posterior holds (per-q AUROC 0.85,
recall@k 0.63). Edge identities and frequencies do not transfer to a new database; meaning does.
This is the core argument for the semantic construction: in a data lake every schema is effectively
unseen.

### 5.3 Representation matters

SELECT-column per-q AUROC by element representation:

| representation | parity | LODO |
|---|---|---|
| `table.col` (bare) | 0.728 | 0.728 |
| `table.col (human name)` | 0.737 | 0.737 |
| **`table.col: description`** | **0.740** | **0.746** |
| verbose `Table X, column Y (name): desc. Values: …` | 0.706 | (worst) |

The description adds signal only when the identifier stays sharp; boilerplate and value-dumps
dilute the discriminative token under mean-pooled embeddings. Question-side evidence also helps.

### 5.4 Calibration

The class-conditional posterior is well-calibrated in-distribution (ECE 0.007–0.019) and degrades
gracefully out of schema (0.03–0.04 for edges/columns; the table node's ECE rises to 0.11 under
LODO, indicating the per-class similarity scales shift across schemas — a recalibration target).

## 6. Open-world detection: is the right table in the lake?

We simulate the open-world case by removing each question's gold table(s) from the candidate set
and asking whether retrieval confidence detects the absence. The candidate set is either the
question's own database (within-DB) or the full cross-database lake (75 tables across 11
databases; cross-DB top-1 retrieval accuracy with the gold present is 0.847).

| confidence signal | within-DB | cross-DB lake |
|---|---|---|
| maximum cosine | 0.779 | 0.770 |
| softmax peakedness (top-prob) | 0.599 | 0.750 |
| top-1/top-2 margin | 0.621 | 0.642 |
| entropy | 0.550 | **0.780** |

In the data-lake setting, multiple confidence signals detect a **missing** table at AUROC ≈ 0.78.
This is a calibrated abstention signal for unanswerable questions, and it is the capability that
generation-side signals cannot provide: judging a generated query presupposes the right table was
retrieved. (Removing the gold — often the top match — mechanically lowers maximum similarity; this
*is* the legitimate signal — a weak best-match implies the target is likely absent — and the
AUROC of 0.78, below 1.0, reflects genuine distributional overlap rather than circularity.)

### 6.1 Discovery over sampled structures (supporting)

A complementary, structure-side open-world signal is the Pitman–Yor **discovery probability**
`(θ + dK)/(θ + N)` over `K` distinct structures among `N` samples, weighted by a base measure over
gold structures. On BIRD it detects "no sampled query is execution-correct" at AUROC 0.68
in-distribution / 0.64 LODO and general incorrectness at 0.71 / 0.68 — modestly above
disagreement baselines (0.58 / 0.62) and partially transferring across schemas. It is a real but
secondary signal; the table-level open-world detector (Section 6) is stronger and more directly
deployable.

## 7. Scoping negatives (what schema-linking UQ is *not* for)

**It does not predict query correctness.** Composing the per-node posteriors over an LLM's
*generated* query to predict execution correctness yields no reliable signal (combined lift over
self-consistency +0.007, 95% CI [−0.012, +0.025] at n=800). The relevance of *chosen* elements
even anti-correlates with correctness: an LLM selects plausible, on-question elements even when its
query's logic is wrong. Query-correctness UQ requires logic-aware signals (verification, white-box
log-probabilities) and is the subject of a companion paper.

**Aggressive schema pruning is unsafe.** A tempting use is to prune the schema fed to the LLM. But
retaining gold elements at a small budget fails, especially for columns: keeping the top 50% of
columns by posterior retains *all* gold columns for only 21% of questions (tables: 75%). Many gold
columns are structural (join keys, filters) with no semantic signature, so the posterior cannot
safely drop columns. Table-level pruning is only safe when timid (keep 75% → 93.5% of questions
retain all gold tables, ≈25% token saving). We report this plainly: the calibrated posterior is a
good *uncertainty* signal, not a safe *aggressive-pruning* signal.

## 8. Limitations

- A single embedding model; a cross-encoder or larger encoder may lift the moderate column node.
- Edge positives are defined as "both endpoints appear in the gold query," a proxy for an actual
  join.
- Class-conditional Gaussians are the simplest density choice; KDE/Beta densities are unexplored.
- The table node's LODO calibration drifts (ECE 0.11), motivating cross-schema recalibration.
- BIRD's documented dictionaries are richer than many real lakes; the representation finding
  (Section 5.3) is the practical lever where documentation is sparse.

## 9. Conclusion

Casting the SQL query as a typed graph and placing a calibrated Bayesian posterior on each
schema-linking decision yields uncertainty that is **strong at the table and join nodes, moderate
at the projected-column node, well-calibrated, and — uniquely valuable for data lakes —
transferable to unseen schemas** where structural priors fail. The same posterior gives an
**open-world detector** (AUROC 0.78) for whether the needed table is present at all. We are candid
about the boundary: schema-linking uncertainty is a retrieval-side signal, not a predictor of
query correctness, and not a license for aggressive pruning. Within that boundary it is a
principled, deployable uncertainty layer for text-to-SQL over data lakes.

---

## Appendix: reproducibility

BIRD dictionaries in `data/bird/desc/`; embeddings cached in `data/embeddings.json`. Scripts:
`bird_lib.py` (dictionary loader, gold/SELECT column extraction); `bird_table_posterior.py`,
`bird_column_posterior.py`, `bird_join_posterior.py` (per-node posteriors, Sections 5.1–5.3);
`bird_openworld.py` (Section 6); `bird_discovery.py` (Section 6.1); `bird_graph_uq.py`,
`bird_prune_feasibility.py` (Section 7). Class-conditional update in `gauss_posterior`
(`bird_column_posterior.py`); empirical-Bayes PY fit in `src/bnp_nl2sql/fit.py`.
