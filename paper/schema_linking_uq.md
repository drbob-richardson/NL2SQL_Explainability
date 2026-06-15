# Calibrated Bayesian Schema-Linking Uncertainty over the Query Graph

*Working write-up — semantic / data-dictionary direction. Updated 2026-06-14.*

## Abstract

An SQL query is a graph: tables are nodes, joins are edges, and projected/filtered columns
are node attributes. We ask whether **Bayesian posteriors over these graph elements**, driven
by the semantic similarity between a natural-language question and a database's **data
dictionary**, yield *calibrated uncertainty* for text-to-SQL — and, critically, *where* on the
graph that uncertainty is informative. We give a principled construction (a class-conditional
likelihood-ratio update, not a softmax) that turns embedding cosine similarities into calibrated
per-node inclusion posteriors, with a natural Bayesian-nonparametric reading (species-sampling
for tables, feature-allocation/IBP for the projected column set, an edge posterior for joins).

On BIRD (1,534 questions, 11 databases with human-written column descriptions) the per-node
posteriors are **strong and calibrated at the table node (per-question AUROC 0.86) and the
join-edge node (0.85), moderate at the SELECT-column node (0.74), and weak for filter/join
columns.** The central positive result is **cross-schema transfer**: the semantic posterior
holds on databases never seen during fitting (leave-one-DB-out AUROC 0.79–0.85), whereas a
structural frequency prior over the same elements **collapses to chance (0.50)** out of schema —
exactly the regime a data lake imposes. The central honest negative: composing the per-node
posteriors over an LLM's *generated* query does **not** reliably predict execution correctness
(+0.046 AUROC over self-consistency, 95% CI [−0.017, +0.108]), because the LLM tends to select
plausible, relevant schema elements even when the query's *logic* is wrong. The defensible
contribution is therefore **calibrated, transferable schema-linking uncertainty for data lakes**,
not full-query correctness UQ.

---

## 1. Motivation

Uncertainty quantification for LLM-generated structured outputs is hard: a single decoded string
hides where the model is unsure. Text-to-SQL is an ideal testbed because the output is an
executable object with explicit structure. Our original hypothesis was that **Bayesian
nonparametric priors on the query graph** could provide structure-native UQ. A first line of work
(§7) placed Pitman–Yor priors over sampled query *structures*; rigorous ablation showed its one
durable, genuinely-Bayesian payoff was **open-world discovery detection**, while its ranking
advantage reduced to a structural-membership test that **did not transfer across databases**.

That non-transfer motivated this direction. Instead of priors over observed *structures*, we put
posteriors over the *graph elements* (tables, edges, columns) and drive them with the **semantics
of the schema** — the database's data dictionary — so that the signal is anchored in meaning that
transfers to unseen schemas. We start at the simplest node ("did we pick the right table?") and
build the graph outward.

## 2. A principled cosine→probability update

Softmax-over-cosine is not principled: cosine is not a likelihood and the temperature is an
arbitrary knob. We instead use a generative latent-variable model, per schema element `c`.

- **Latent variable** `z_c ∈ {0,1}`: is element `c` part of the gold query?
- **Prior** `P(z_c = 1) = π` — the base rate that an element is used (small; a query touches few
  of many candidates). Estimated from training golds (BIRD column base rate `π ≈ 0.113`).
- **Observation** `s_c = cos(emb(question), emb(dict_c))` modelled *class-conditionally*:
  `s_c | z_c=1 ~ f_1`, `s_c | z_c=0 ~ f_0` (Gaussian; QDA, so unequal variances allowed).
  `f_1, f_0` are fit on training data — the similarities of elements that *are* vs *are not* in
  the gold query.
- **Posterior (the update):**

  ```
  P(z_c = 1 | s_c) =  π · f_1(s_c)
                     ----------------------------------
                     π · f_1(s_c) + (1−π) · f_0(s_c)
  ```

The embedding enters as a genuine **likelihood ratio** `f_1(s_c)/f_0(s_c)` (a Bayes factor) that
tilts the prior to a calibrated posterior. If `f_0,f_1` are equal-variance Gaussians this reduces
*exactly* to a logistic link in `s_c` — so "logistic on cosine" is recovered as a special case,
but here it is derived, calibratable, and carries the base rate.

**BNP reading.** Each clause of the query graph is a different combinatorial object with a
matching nonparametric prior, all tilted by the same embedding likelihood:
- **tables** — a categorical / species-sampling draw (which entities are present);
- **the SELECT set** — a *feature allocation*, whose canonical BNP prior is the Indian Buffet /
  Beta–Bernoulli process; the embedding makes it a covariate-tilted IBP;
- **joins** — an edge prior over the foreign-key graph.

## 3. Data and protocol

- **Benchmark:** BIRD dev — 1,534 questions over 11 databases, each shipping a **data dictionary**
  (`original_column_name`, human `column_name`, `column_description`, `value_description`) and a
  foreign-key graph. Gold-column resolution against the dictionary is 99.6%.
- **Embeddings:** OpenAI `text-embedding-3-small`, cached (total embedding spend ≈ \$0.01).
- **Evaluation:** two regimes —
  - *parity split* (in-distribution: train on even-indexed questions, evaluate on odd);
  - *leave-one-DB-out (LODO)* (transfer: fit on 10 databases, evaluate on the held-out one).
- **Metrics:** per-question AUROC ("among this question's candidates, are the used ones ranked
  above the unused?"), recall@k (fraction of gold elements in the top-k), and ECE (calibration of
  the posterior).

## 4. Per-node results (BIRD)

Best representation per node (column: `table.col: description`; edge: mean of endpoint
similarities). Parity / LODO:

| Query-graph node | per-q AUROC (parity / LODO) | recall@k (parity / LODO) | ECE (parity / LODO) |
|---|---|---|---|
| **Table** (which tables) | 0.864 / 0.858 | 0.749 / 0.739 | 0.018 / 0.113 |
| **Join edge** (which FK joins) | 0.861 / 0.852 | 0.617 / 0.631 | 0.019 / 0.033 |
| **SELECT column** (what to return) | 0.740 / 0.746 | 0.309 / 0.315 | 0.007 / 0.037 |
| filter / join columns | weak (≈ all-refs 0.67) | — | — |

Reading:
- **Table and edge nodes are strong and well-calibrated.** A join edge is needed iff *both* its
  endpoint tables are relevant, so the edge node inherits the table node's semantic strength
  (using both endpoints — mean/product — beats the bottleneck `min`).
- **The SELECT-column node is moderate** — it *ranks* a question's columns (per-q 0.74) but cannot
  precisely pin the exact set (recall@k 0.31). Columns chosen for *structural* reasons (join keys,
  IDs, filters) carry no semantic match to the question and are effectively unpredictable from
  similarity; restricting the target to the *returned* columns is what lifts the signal (all-refs
  per-q 0.67 → SELECT-only 0.74).

## 5. The headline: cross-schema transfer

For the edge node we compared the semantic posterior to a **structural** prior — the marginal
foreign-key edge-usage frequency:

| edge signal | AUROC (parity) | AUROC (LODO, unseen schema) |
|---|---|---|
| structural FK-usage frequency | 0.741 | **0.500 (chance)** |
| semantic posterior | 0.727 | **0.708** |
| combined | 0.815 | 0.708 (structural adds nothing out of schema) |

In-distribution, structure and semantics are comparable and complementary (combined 0.82). **Out
of schema, the structural prior collapses to chance** — edge identities and frequencies do not
transfer to a new database — while **the semantic posterior holds (0.71, per-q 0.85, recall@k
0.63).** This is the same cross-DB non-transfer that limited the earlier structural BNP work
(§7), and it is the central argument for the semantic construction: in a **data lake**, every
schema is effectively unseen, so only the meaning-anchored posterior is usable.

## 6. Representation matters (a practical finding)

Naively embedding a verbose data-dictionary record *hurts*. SELECT-column node, per-q AUROC:

| column representation | parity | LODO |
|---|---|---|
| `table.col` (bare name) | 0.728 | 0.728 |
| `table.col (human name)` | 0.737 | 0.737 |
| **`table.col: description`** | **0.740** | **0.746** |
| verbose `Table X, column Y (name): desc. Values: …` | 0.706 | (worst) |

The column **description genuinely adds signal**, but only when the `table.column` anchor is kept
sharp and the description appended cleanly; boilerplate prefixes and value-dumps **dilute** the
discriminative token under mean-pooled embeddings. Question-side **evidence** (BIRD's external
knowledge) also helps. Net: metadata helps *if formatted to preserve the identifier*.

## 7. Relation to the structural BNP line (prior work in this project)

Before the semantic direction we built a Pitman–Yor "Bayesian self-consistency" model over
sampled query structures (single-table Spider, n=544; full library, 36+ tests; LaTeX draft in
`paper/tex/`). Honest findings after adversarial ablation:
- Its selective-prediction *ranking* advantage (AURC) was largely a **tie-interpolation artifact**
  under 83% saturation; on the tie-robust metric it was ≈ a binary structural-membership test.
- That membership signal **did not transfer across databases** (≈1.8% structure overlap).
- Its one durable, uniquely-Bayesian payoff was **open-world discovery detection** — the
  probability that the correct query uses a never-sampled structure flagged guaranteed-wrong
  ("gold-unseen") cases at **AUROC ≈ 0.84 out-of-sample**, where every disagreement-based UQ
  signal was at chance. Discovery enables correct *abstention*, not correction.

The semantic direction is the response to the transfer failure: anchor the signal in schema
*meaning*, which §5 shows does transfer.

## 8. The honest negative: graph confidence does not predict query correctness

We tested the real downstream payoff. We generated SQL with gpt-4o-mini (BIRD slice, 200
questions across 7 databases, K=8 samples, schema + evidence in the prompt, executed against the
real DBs; modal execution accuracy 0.445 — a genuinely hard regime). For each generated query we
composed the per-node posteriors (table / edge / SELECT-column relevance of the *generated*
elements, plus table-retrieval peakedness) into a graph-level confidence, with class-conditional
models fit on the **non-slice** questions (no leakage).

| signal | AUROC for execution correctness |
|---|---|
| self-consistency (sample agreement) | 0.654 |
| table-retrieval peakedness | 0.585 |
| relevance of chosen tables | 0.415 (anti-correlated) |
| relevance of chosen edges | 0.394 (anti-correlated) |
| relevance of chosen SELECT columns | 0.566 |
| **self-consistency + graph UQ (cross-fit logistic)** | **0.693** vs 0.646 |
| bootstrap delta (full − base) | **+0.046, 95% CI [−0.017, +0.108]**, P(>0)=0.92 |

The composed signal does **not** reliably beat self-consistency: the lift is small and its CI
crosses zero, and the relevance of *chosen* tables/edges even **anti-correlates** with
correctness. The mechanism is clear and was independently seen on Spider: **the LLM selects
plausible, on-question schema elements even when its query is wrong**; correctness is dominated by
*logic* — aggregation, filter conditions, value formatting, nesting — that schema-element
relevance cannot see. Schema-element selection is largely satisfied for both correct and incorrect
queries, so it does not discriminate them.

## 9. What this is, and is not

- **It is:** a principled, calibrated, **cross-schema-transferable** uncertainty model for
  **schema linking** — which tables, joins, and returned columns a question implies — with a
  coherent per-node BNP construction and an honest map of where embedding signal lives on the
  query graph (strong: tables, joins; moderate: returned columns; weak: structural columns).
- **It is not:** a predictor of whether a generated query is *correct*. That gap is real and
  recurs across both the structural and semantic directions: UQ here works for well-posed
  *sub-problems* (open-world discovery; schema linking), not for general query-logic correctness.

## 10. Limitations

- Single embedding model (`text-embedding-3-small`); a stronger encoder or a cross-encoder
  reranker might lift the column node, untested.
- Correctness study is one generator (gpt-4o-mini) at n=200; the composition delta is
  under-powered (CI crosses zero) — more data would sharpen it but the effect size is small
  regardless.
- Edge positives are defined as "both endpoints appear in the gold query," a proxy for an actual
  join.
- Class-conditional Gaussians are the simplest choice; KDE/Beta densities are unexplored.

## 11. Contribution and future work

**Contribution.** Calibrated Bayesian schema-linking uncertainty over the query graph from
semantic metadata, with per-node combinatorial priors tilted by an embedding likelihood ratio,
that transfers to unseen schemas where structural priors fail — the data-lake regime.

**Future work, in priority order.**
1. *The logic-error gap (the real unsolved problem this work exposes).* Predicting query
   correctness needs **execution-grounded** signals (run candidates, test result-set properties /
   typed sanity checks) or **white-box** signals (sequence log-probabilities — which on Spider
   already separated confident-wrong queries at AUROC 0.79 where sampling UQ was blind). This is
   orthogonal to schema linking and is where correctness UQ likely lives.
2. *Selective schema linking.* Use the calibrated table/edge posteriors to abstain or to prune the
   schema fed to the LLM (smaller, higher-precision context) — a concrete, testable downstream use
   of the transfer property.
3. *Stronger column representations* (cross-encoder rerank; value-aware embeddings) to lift the
   weakest node.
4. *Actuarial / enterprise validation,* where rich, documented data dictionaries are the norm and
   the metadata signal should be strongest.

---

### Reproducibility

All scripts under `scripts/`, embeddings cached in `data/embeddings.json`, BIRD data dictionaries
in `data/bird/desc/`, generated samples in `data/bird_samples.json`. Total OpenAI spend for this
direction ≈ \$0.06 (embeddings + the 200-question generation slice).

- `bird_lib.py` — data-dictionary loader; gold-column / SELECT-column extraction.
- `bird_table_posterior.py`, `bird_column_posterior.py`, `bird_join_posterior.py` — per-node
  posteriors (§4–6).
- `table_selection.py` — original table-node / data-lake retrieval experiment.
- `bird_generate.py` — safe-by-default sampler + executor (§8).
- `bird_graph_uq.py` — graph-level composition + bootstrap (§8).
