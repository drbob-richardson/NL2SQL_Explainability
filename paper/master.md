# Bayesian Uncertainty over the Query Graph for Text-to-SQL
### A master manuscript: theory, methods, and a complete (good-and-bad) empirical account

*Working synthesis — 2026-06-14. Consolidates the structural-BNP line, the semantic
schema-linking line, and the correctness-UQ line into one document for direction-setting.*

---

## Abstract

An SQL query is a typed graph — tables are nodes, joins are edges, projected and filtered
columns are node attributes — produced by a generative process (an LLM) we do not control. We
ask whether **Bayesian, and specifically Bayesian-nonparametric (BNP), structure over that graph
yields calibrated uncertainty** for text-to-SQL, and we map, rigorously and honestly, *where* on
the graph and *with which signals* useful uncertainty actually exists.

We develop (i) a formal query-graph space and a Pitman–Yor (PY) generative model over sampled
query structures, whose discovery probability `(θ+dK)/(θ+N)` gives an **open-world** signal no
frequency method can express; (ii) a per-node BNP construction — species-sampling for tables, a
covariate-tilted Indian-Buffet/Beta–Bernoulli process for the projected column set, and an edge
posterior for joins — each updated from the database's **data dictionary** by a principled
class-conditional **likelihood-ratio** rule (not a softmax); and (iii) a selective-prediction
layer with distribution-free risk control.

Empirically, across Spider (single-table, n=544) and BIRD (11 databases, n=1,534; with a
200/800-question generation study), we find a clear and repeatable picture:
- **The structural PY model's ranking advantage was largely a metric artifact** (tie
  interpolation under 83% saturation); its one durable, uniquely-Bayesian win is **open-world
  discovery detection (AUROC 0.84 out-of-sample)**.
- **Semantic schema-linking posteriors are strong and calibrated at the table (per-q AUROC 0.86)
  and join-edge (0.85) nodes, moderate at the SELECT-column node (0.74)**, and — the key positive
  — **transfer to unseen schemas (LODO 0.71–0.86) where structural priors collapse to chance
  (0.50)**.
- **No structural, sampling, or schema-linking signal predicts query *correctness***; on hard
  BIRD all such signals plateau at AUROC ≈ 0.62.
- **Correctness UQ is tractable only with signals that are not blind to logic**: white-box
  **log-probabilities (0.67)** and an **LLM verifier (0.72; an independent stronger judge 0.77)**,
  which combine to AUROC 0.76 and enable risk-controlled abstention the self-consistency baseline
  structurally cannot offer.

We give the full theory, all numbers, all negatives, and a frank discussion of where Bayesian
nonparametrics is load-bearing versus where the strongest result is standard selective
prediction.

---

## 1. Introduction

### 1.1 The idea

Uncertainty quantification (UQ) for LLM outputs is hard, and hardest for *structured* outputs,
where a single decoded string hides where the model is unsure. Text-to-SQL is an ideal testbed:
the output is an executable object with explicit, typed structure. The originating hypothesis of
this project — from a Bayesian-nonparametrics standpoint — was that an SQL query *is a graph*, and
that **BNP priors over that graph** could provide structure-native, calibrated UQ: a *posterior
over query graphs* rather than a flattened confidence on a token string.

### 1.2 What we set out to deliver

(a) public datasets to test on; (b) theory; (c) methods; (d) code; (e) a publication. This
document is the synthesis of (a)–(d) and the substrate for (e).

### 1.3 The arc (and why the paper has three parts)

The work moved through three lines, each motivated by the honest failure of the last:

1. **Structural BNP (Part A).** A PY "Bayesian self-consistency" model over sampled query
   *structures*. Outcome: a uniquely-Bayesian open-world signal that works, wrapped around a
   ranking advantage that turned out to be a metric artifact and **did not transfer across
   databases**.
2. **Semantic schema-linking (Part B).** The transfer failure motivated anchoring the signal in
   schema *meaning* (the data dictionary), with per-node BNP posteriors. Outcome: calibrated,
   **transferable** uncertainty about *which* schema elements a question implies — but it does not
   predict whether the generated query is *correct*.
3. **Correctness UQ (Part C).** The correctness gap motivated testing signals that can reason
   about query logic. Outcome: verification + white-box confidence break the correctness ceiling
   and enable abstention.

### 1.4 Contributions

1. A formal query-graph space and a PY generative model with an explicit **open-world discovery
   probability**, validated as the one genuinely-Bayesian capability that detects guaranteed-wrong
   outputs where all disagreement signals are at chance.
2. A **per-node BNP construction** (species-sampling / IBP / edge posterior) with a principled
   **class-conditional likelihood-ratio** update from data-dictionary embeddings, shown to be
   calibrated and **cross-schema transferable**.
3. A rigorous **map of where UQ signal lives** on the query graph, including strong negatives.
4. A **correctness-UQ** result: verification + log-probabilities break the black-box ceiling and
   support distribution-free selective prediction; a deployable abstention frontier.

---

## 2. Background and related work

**Text-to-SQL.** LLM-dominated; evaluated by *execution accuracy* (result-set equality, not
string match). Benchmarks: Spider, BIRD (adds human column descriptions + "evidence"), Spider 2.0
(hard, ~21% solved). Schema linking (mapping a question to tables/columns) is a recognized,
hard sub-problem.

**UQ for SQL / structured outputs.** Self-consistency / sampling; verbalized confidence;
semantic entropy (Farquhar et al., *Nature* 2024); conformal/selective generation; sub-clause
Platt scaling (EMNLP 2025); RTS conformal abstention (SIGMOD 2025, the nearest baseline). All
flatten the output; none places a Bayesian posterior over the query graph. (Verified literature
gap; we do **not** claim "first calibration for SQL," which is false.)

**Graph representations of SQL.** RAT-SQL, LGESQL — graph structure on the *encoder* side. No
prior work puts a posterior over the *output* query graph.

**Bayesian nonparametrics.** Pitman–Yor / Dirichlet processes and species sampling; adaptor
grammars (Johnson et al., NIPS 2006); Indian Buffet / Beta–Bernoulli processes for feature
allocation. Closest neighbours place BNP priors over latent structure, not over the discrete
output structures of a generative decoder.

---

## 3. Theory

### 3.1 The query graph

A query `q` over schema `S` is a typed graph `G(q)`: nodes are table instances and the columns /
aggregates they expose; edges are joins (FK paths) and the operator dependencies
(WHERE/GROUP BY/HAVING/ORDER BY). We use two canonical serializations via `sqlglot`:
- **canonical_key** — column-concrete structure (binding-level);
- **skeleton_key** — column-abstracted shape (structure-level: columns→placeholder,
  functions→AGG/SCALAR).

This factorization separates **structural** uncertainty (what shape) from **binding**
uncertainty (which columns/values).

### 3.2 The space of queries Q(S)

Restricting (initially) to a single-table fragment with bounded predicate depth/arity, the set of
valid queries `Q(S)` is **countable and finite** (Prop. 1), and enumerable: for a 9-column airbnb
schema, `|Q(S)| ≈ 4.1M` under tight bounds (≈85B looser). Constraints Φ encode executability
(type compatibility, GROUP BY consistency, HAVING-needs-grouping, ORDER scope). Adaptor grammars
guarantee only *context-free* validity; SQL executability is *context-sensitive* — handled by a
typed/attribute grammar or per-step masking. This is the formal object a prior lives on.

### 3.3 Pitman–Yor over structures, and the discovery probability

Given `N` LLM samples falling into `K` distinct structures, a PY restaurant with discount `d` and
concentration `θ` gives the predictive mass on a seen structure and, crucially, the
**discovery probability**

```
P(next structure is new) = (θ + d·K) / (θ + N).
```

Interpreted at test time, this is **P(the correct query is a structure never sampled)** — an
*open-world* abstention signal. Frequency self-consistency cannot express it (it only sees the
structures that appeared). Empirical-Bayes fits `(d, θ)` by maximizing the PY EPPF over training
partitions; the gold-structure distribution serves as the base measure `H`.

### 3.4 Per-node BNP construction (the semantic model)

Rather than a prior over whole observed structures, place a posterior over each **graph element**,
driven by the question's semantic match to the data dictionary. For element `c` with latent
inclusion `z_c ∈ {0,1}` and prior base rate `P(z_c=1)=π`:

- **Tables** — a categorical / **species-sampling** draw (which entities are present).
- **The SELECT set** — a **feature allocation**, whose canonical BNP prior is the
  **Indian Buffet / Beta–Bernoulli process**; the embedding makes it a *covariate-tilted* IBP.
- **Joins** — an **edge posterior** over the foreign-key graph; an edge is "on" iff both endpoint
  tables are relevant.

### 3.5 The principled cosine→probability update

Softmax-over-cosine is not principled (cosine is not a likelihood; temperature is arbitrary). We
model the embedding similarity `s_c = cos(emb(question), emb(dict_c))` **class-conditionally**:
`s_c | z_c=1 ~ f_1`, `s_c | z_c=0 ~ f_0` (Gaussian/QDA), fit on training golds. Then

```
P(z_c = 1 | s_c) =  π·f_1(s_c) / [ π·f_1(s_c) + (1−π)·f_0(s_c) ].
```

The embedding enters as a genuine **likelihood ratio** `f_1/f_0` (a Bayes factor) that tilts the
prior to a calibrated posterior. Equal-variance Gaussians recover a logistic link in `s_c` as a
special case — so "logistic on cosine" is *derived*, calibratable, and carries the base rate.

### 3.6 Selective prediction and conformal risk control

Given a confidence score `s` per question, answer iff `s ≥ τ`, else abstain. We calibrate `τ`
for a target selective risk `α` with distribution-free certificates: Learn-Then-Test (LTT) with
exact-binomial p-values over a nested grid, and a Bonferroni-over-grid variant (δ=0.1). We report
both the PAC-certified frontier and the empirical risk–coverage frontier (AURC, risk-coverage
curve).

---

## 4. Experimental setup

- **Benchmarks.** Spider dev single-table fragment (n=544, 20 DBs, golds from `xlangai/spider`,
  DBs from `premai-io/spider`); BIRD dev (n=1,534, 11 DBs with data dictionaries + evidence,
  official `dev.zip`). Earlier airbnb cheat-sheet fragment (n=81) for development.
- **Generation.** OpenAI `gpt-4o-mini` (and `gpt-4o`, `gpt-3.5-turbo` for sweeps), K temperature
  samples per question, schema (+ evidence for BIRD) in the prompt; executed against real SQLite
  DBs for ground-truth correctness.
- **Embeddings.** `text-embedding-3-small`, cached.
- **Verifiers.** `gpt-4o-mini` and `gpt-4o` as LLM judges; P(correct) read from the YES/NO
  first-token logprobs.
- **Metrics.** Execution accuracy; (tie-robust) AUROC for correctness/inclusion; AURC; ECE;
  per-question AUROC and recall@k (schema linking); risk–coverage with PAC certificates.
- **Discipline.** All sampling is *safe-by-default* (dry-run cost estimate, no calls without
  `--run`, hard `--max-calls`, on-disk caching). **Total OpenAI spend to date ≈ \$2.65 of \$17.**

---

## 5. Results — Part A: Structural BNP (Spider, single-table)

**Accuracy.** Execution accuracy 0.805 (`gpt-4o-mini`), 0.825 (`gpt-4o`); model strength barely
moves single-table accuracy (near ceiling).

**The ranking advantage was a metric artifact (the central negative).** Naïvely, the PY model's
selective-prediction AURC looked dominant (0.094 vs baseline 0.172). But **83% of single-table
questions are unanimous (K=1)**; AURC is interpolation-sensitive under that saturation. On the
honest, **tie-robust AUROC**, the gap shrinks to PY 0.703 vs self-consistency 0.609 — and
`H(MAP)` alone (a binary "is the MAP structure in the training set?" test) already gives 0.691.
So the "BNP machinery" largely reduced to a **structural-membership prior**, not the PY/discount
dynamics.

**It does not transfer across databases.** The membership signal relies on structural overlap
between train and test golds; cross-database structure overlap is ≈ **1.8%**, so `H` is near-vacuous
on a new schema. This is the failure that motivated Part B.

**The one durable, uniquely-Bayesian win: open-world discovery.** Define a question as
*gold-unseen* if the correct query was never sampled (answering is then guaranteed wrong);
212/544 are gold-unseen, 172 of them unanimous-K=1 (the "confident-wrong floor"). The
**discovery probability detects gold-unseen at AUROC 0.868 in-sample / 0.839 out-of-sample**,
versus **chance** for every disagreement signal (1−top_prob 0.514, n_distinct 0.518, semantic
0.557). Mechanism: unanimous-but-wrong queries use rare structures (low `H` → high discovery);
unanimous-correct use common structures (high `H` → low discovery). Discovery also predicts
general incorrectness at 0.710 vs ≈0.61 for baselines. It enables correct *abstention*, not
correction.

**Selective prediction.** Split-conformal: the PY model answers 51% of questions at 7.9%
held-out risk; the **self-consistency baseline achieves 0% coverage at any threshold** ≤10% risk,
because its top-confidence bucket (unanimous samples) already exceeds 10% error. Bayesian
de-saturation is what enables risk-controlled abstention. Distribution-free PAC certificates
(LTT/Bonferroni) remain conservative at δ=0.1 (the confident-wrong floor caps the top-decile
precision).

**White-box logprobs (preview of Part C).** Among unanimous K=1 questions, sequence
log-probability separates correct from wrong at **AUROC 0.79**, where sampling UQ is blind;
logprob-alone AURC (0.074) even edges the PY model (0.097) on this data. The first hint that
correctness UQ needs white-box / reasoning signals.

---

## 6. Results — Part B: Semantic schema-linking (BIRD)

Setup: π (column base rate) ≈ 0.113; gold-column resolution against the dictionary 99.6%.
Two regimes: parity split (in-distribution) and leave-one-DB-out (LODO, transfer). Best
representation per node (column: `table.col: description`; edge: mean endpoint similarity).

### 6.1 The per-node map (parity / LODO)

| Query-graph node | per-q AUROC | recall@k | ECE |
|---|---|---|---|
| **Table** (which tables) | 0.864 / 0.858 | 0.749 / 0.739 | 0.018 / 0.113 |
| **Join edge** (which FK joins) | 0.861 / 0.852 | 0.617 / 0.631 | 0.019 / 0.033 |
| **SELECT column** (what to return) | 0.740 / 0.746 | 0.309 / 0.315 | 0.007 / 0.037 |
| filter / join columns | ≈ 0.67 (all-refs) | — | — |

Tables and edges are strong and calibrated; an edge inherits the table node's strength (it needs
*both* endpoints relevant, so mean/product beats the bottleneck min). The SELECT-column node
*ranks* well (0.74) but cannot pin the exact set (recall@k 0.31); columns chosen for structural
reasons (join keys, IDs, filters) carry no semantic match and are unpredictable — restricting the
target to *returned* columns lifts the signal (all-refs 0.67 → SELECT 0.74).

### 6.2 The headline positive: cross-schema transfer

For the edge node, semantic posterior vs a structural FK-usage-frequency prior:

| edge signal | AUROC (parity) | AUROC (LODO, unseen schema) |
|---|---|---|
| structural FK-usage frequency | 0.741 | **0.500 (chance)** |
| semantic posterior | 0.727 | **0.708** |
| combined | 0.815 | 0.708 |

In-distribution they're comparable and complementary; **out of schema the structural prior
collapses to chance while the semantic posterior holds** (per-q 0.85, recall@k 0.63). This is the
same non-transfer that limited Part A — and the central argument for the semantic construction: in
a **data lake**, every schema is effectively unseen, so only the meaning-anchored posterior is
usable.

### 6.3 Representation matters (a practical finding)

SELECT-column per-q AUROC: bare `table.col` 0.728 → **`table.col: description` 0.740/0.746** →
verbose `Table X, column Y (name): desc. Values: …` **0.706 (worst)**. The description adds signal
only when the `table.column` anchor stays sharp; boilerplate and value-dumps dilute the
discriminative token under mean-pooled embeddings. Question-side **evidence** also helps.

### 6.4 The negative: schema linking does not predict correctness

Composing the per-node posteriors over an LLM's *generated* query to predict execution
correctness gives essentially nothing: at n=200 a +0.046 AUROC hint, which **vanishes at n=800
(+0.007, 95% CI [−0.012, +0.025])**. The relevance of *chosen* tables/edges even anti-correlates
with correctness — the LLM selects plausible, on-question elements *even when its query is wrong*.
Correctness lives in the logic, not the schema selection.

---

## 7. Results — Part C: Correctness UQ (BIRD, n=800, modal accuracy 0.451)

### 7.1 The black-box ceiling, confirmed

| signal | AUROC for correctness |
|---|---|
| string self-consistency | 0.616 |
| execution self-consistency | 0.613 |
| schema-linking graph confidence | 0.553 |

All black-box statistical signals plateau at ≈0.62. "Does it execute / return rows" is chance
(wrong queries run fine). The ceiling is a property of the problem, not of small samples.

### 7.2 Breaking the ceiling: verification + white-box confidence

| signal | AUROC alone | combined with self-consistency (Δ, 95% CI) |
|---|---|---|
| logprob (white-box) | 0.669 | +0.054 [+0.027, +0.081] |
| LLM verifier (gpt-4o-mini) | 0.724 | +0.095 [+0.060, +0.131] |
| LLM verifier (gpt-4o, independent) | **0.770** | — |
| verifier + logprob | — | +0.109 [+0.071, +0.147] |
| all signals | **0.763** | +0.117 [+0.077, +0.156] |

All positive CIs exclude zero (P(Δ>0)=1.00 at n=800). The signals that work are exactly those
**not blind to query logic**.

### 7.3 The same-model-bias caveat — checked and alleviated

The verifier could be suspect if a model just agrees with itself. But an **independent, stronger**
judge (`gpt-4o` verifying `gpt-4o-mini`'s SQL) scores **0.770 > 0.724**, i.e. *better*, not worse,
and the two judges correlate only **0.49** — they capture different things and combine. The
verifier signal is not a self-agreement artifact.

### 7.4 Selective prediction / abstention (empirical risk–coverage)

| target risk | self-consistency | all signals |
|---|---|---|
| 0.20 | — (no valid subset) | **9% cov @ 14% risk** |
| 0.30 | — | 27% cov @ 24% risk |
| 0.40 | — | 58% cov @ 38% risk |

Self-consistency cannot form a low-risk subset at all; the combined score can. The
**distribution-free PAC certificate is conservative** (mostly abstain-all; only "all signals at
α=0.4" certifies 16% cov / 22% risk) because **base accuracy is only 45%** — tight guarantees need
a stronger generator (higher base accuracy) or more calibration data, not a better signal.

---

## 8. Synthesis: a map of where UQ signal lives

| Question | Best signal | Strength | Transfers? |
|---|---|---|---|
| Which **tables**? | semantic posterior | strong (0.86) | ✅ |
| Which **joins**? | semantic posterior | strong (0.85) | ✅ |
| Which **SELECT columns**? | semantic posterior | moderate (0.74) | ✅ |
| Is the correct query **unseen** (open-world)? | PY discovery prob | strong (0.84 oos) | ✅ (within benchmark) |
| Is the generated query **correct**? | verifier + logprob | moderate (0.76) | needs cross-model check ✓ |
| Is the generated query correct (black-box only)? | self-consistency | weak (0.62 ceiling) | — |

Two regimes have real, calibrated, transferable signal — **schema linking** (which elements) and
**open-world discovery** (is the truth even reachable). General **correctness** UQ requires
logic-aware signals (verification, logprobs); pure sampling/structure cannot do it.

---

## 9. Honest limitations and consolidated negatives

- **The PY ranking advantage was a saturation/tie artifact;** the durable BNP win is discovery,
  not AURC, and not post-hoc calibration (Platt scaling equalizes all methods' ECE).
- **Structural priors do not transfer across schemas** (≈1.8% overlap; LODO → chance).
- **Schema-element relevance does not predict query correctness** (null at scale).
- **Strict PAC certificates are conservative** under low base accuracy (45% on BIRD).
- **Single embedding model**; a cross-encoder/`embedding-3-large` is untested for the weak column
  node.
- **The verifier is within one provider** (gpt-4o-mini/gpt-4o); a truly cross-*family* judge
  (e.g., Claude) is untested (no Anthropic credits).
- **Generators are gpt-4o-mini/gpt-4o**; no frontier or fine-tuned generator; Spider 2.0 untested.
- The fundamental limit stands: **confident-wrong unanimous samples are invisible to all sampling
  UQ** — only logprobs/verification/discovery touch them.

---

## 10. The BNP question (frank)

Where is Bayesian nonparametrics *load-bearing*?

- **Genuinely BNP and genuinely working:** the **open-world discovery probability** (PY EPPF) —
  the one signal that detects guaranteed-wrong outputs at 0.84 where everything else is at chance.
  This is the clearest case where the nonparametric machinery does something no baseline can.
- **BNP as principled framing (novel, moderately working):** the **per-node construction**
  (species-sampling tables, **IBP/Beta–Bernoulli** SELECT set, edge posterior) with the
  likelihood-ratio update. The math is clean and the calibration/transfer results are real; the
  column node is only moderate.
- **Not BNP (but the strongest practical result):** the **correctness UQ** (verifier + logprob →
  abstention) is standard selective prediction. It is the most deployable result and the least
  Bayesian-nonparametric.

So the project's *strongest empirical result* and its *most distinctive theoretical contribution*
are not the same thing. That tension is the central decision below.

---

## 11. Candidate paper framings (for direction-setting)

**(A) "Selective prediction for text-to-SQL: what predicts correctness."**
Lead with the Part C result (verifier+logprob break the ceiling; abstention frontier; rigorous
negatives for sampling/structure/schema-linking). BNP appears as one section (discovery).
*Pros:* strongest, cleanest empirics; high acceptance odds. *Cons:* least BNP; the discovery and
schema-linking results become supporting acts. *Venue:* an ML/NLP empirical track.

**(B) Two papers.**
(B1) *BNP schema-linking UQ over the query graph* — Parts A(discovery)+B; the species-sampling/IBP
construction, transfer, calibration. (B2) *Selective prediction for text-to-SQL* — Part C.
*Pros:* each is coherent and well-scoped; B1 is the BNP paper you want. *Cons:* B1's correctness
relevance is indirect; more total work.

**(C) One integrated paper: "Bayesian uncertainty over the query graph."** *(recommended)*
Theory = the query graph + PY discovery + per-node BNP construction (your BNP core). Empirics in
three layers: discovery (open-world), schema-linking (transferable, calibrated), correctness
(verifier+logprob → abstention). The narrative is exactly the map in §8: a principled account of
*where* Bayesian structure helps and where logic-aware signals are required.
*Pros:* keeps BNP as the principled spine while leading the practical claims with what works;
the honest "map" is itself a contribution; matches the project's actual story. *Cons:* broader
scope; must be careful not to overclaim BNP for the correctness layer.

**(D) BNP-purist: discovery + per-node posteriors only.** Drop Part C. *Pros:* maximally BNP,
tight. *Cons:* leaves the strongest result on the table; correctness UQ is the part reviewers most
want.

A note on fit with your trajectory (BNP → AI/actuarial): framing **(C)** keeps the BNP identity
and foregrounds **risk-controlled abstention**, which is the actuarial-relevant capability.

---

## 12. Future work

1. **Tighten the certificate** with a stronger generator (higher base accuracy makes PAC certs
   fire at useful coverage) and/or more calibration data.
2. **Cross-family verifier** (Claude or another provider) — the one remaining robustness check on
   the strongest result.
3. **Stronger column representations** (cross-encoder rerank; value-aware embeddings) for the weak
   node.
4. **Selective schema linking** — use the calibrated, transferable table/edge posteriors to prune
   the schema fed to the LLM (smaller, higher-precision context), a concrete downstream use.
5. **The trained predictor (the big bet)** — fine-tune a model to emit calibrated query-graph
   posteriors end-to-end, rather than wrapping a frozen LLM.
6. **Harder benchmarks** (Spider 2.0) where UQ and abstention matter most.
7. **Actuarial / enterprise validation**, where rich documented data dictionaries are the norm and
   the schema-linking signal should be strongest.

---

## 13. Conclusion

Treating an SQL query as a graph and asking *where Bayesian structure helps* yields a clear,
honest map. Bayesian nonparametrics earns its keep in two specific places — the **open-world
discovery probability** (a capability no frequency method has) and the **per-node, transferable,
calibrated schema-linking posteriors**. It does **not** by itself solve query-correctness UQ; that
requires logic-aware signals (verification and white-box log-probabilities), which we show break
the black-box ceiling and enable risk-controlled abstention. The result is both a principled
Bayesian framework over the query graph and a candid account of its reach.

---

## Appendix: reproducibility and cost

All code under `scripts/`; library under `src/bnp_nl2sql/` (40+ tests). Embeddings cached in
`data/embeddings.json`; BIRD dictionaries in `data/bird/desc/`; generations in
`data/bird_samples.json` (with logprobs); verifier scores in `data/bird_verify*.json`; cached
signals in `data/bird_signals.json`. Total OpenAI spend ≈ **\$2.65 of \$17**.

| component | script | spend |
|---|---|---|
| Spider single-table (544, mini + gpt-4o) | `spider_benchmark.py`, `model_sweep.py` | ~\$1.6 |
| structural ablations / discovery / logprobs | `ablate_mechanism.py`, `discovery_detection.py`, `logprob_experiment.py` | ~\$0.1 |
| BIRD per-node posteriors | `bird_table_posterior.py`, `bird_column_posterior.py`, `bird_join_posterior.py` | ~\$0.01 (embeddings) |
| BIRD generation (800, logprobs) | `bird_generate.py` | ~\$0.19 |
| LLM verifier (mini + gpt-4o) | `bird_verify.py` | ~\$0.58 |
| correctness / abstention analysis | `bird_correctness_uq.py`, `bird_abstention.py`, `bird_graph_uq.py`, `bird_exec_uq.py` | \$0 |

Prior write-ups: `paper/schema_linking_uq.md` (Part B detail), `paper/tex/paper.tex` (Part A
LaTeX draft), `paper/theory.md`, `paper/methods.md`, `paper/lit_review.md`.
