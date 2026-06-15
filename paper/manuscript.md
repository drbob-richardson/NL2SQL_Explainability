# Pitman–Yor Bayesian Self-Consistency: Calibrated, Open-World Uncertainty for Text-to-SQL

*Working draft — 2026-06-13. Single-author methods paper. Every number below is reproducible
from the repository scripts named in §6.6; the supporting library has 43 passing unit tests.*

---

## Abstract

Text-to-SQL (NL2SQL) systems report execution accuracy but rarely a **calibrated confidence**
on the query they are about to run against a database. Existing black-box uncertainty
methods operate on the flat token sequence (sequence probability, semantic entropy) or on the
empirical frequency of the most common sampled query (self-consistency). We recast
uncertainty quantification (UQ) as **Bayesian inference over the space of query structures**:
each sampled query is parsed into a typed graph, canonicalized, and treated as a draw from a
latent per-question distribution on which we place a **Pitman–Yor (PY) process** prior. The
posterior-predictive mass on the most likely structure is a Bayesian form of self-consistency
that (i) **de-saturates** the frequency estimate — eight-of-eight agreement is not certainty —
and (ii) carries a closed-form **discovery probability**, the posterior mass on structures the
model never sampled, an open-world signal that frequency methods cannot express.

We scope the study to a single-table SQL fragment, for which the executable query space is
exactly characterizable, and evaluate with **execution accuracy** on two benchmarks: a
31-question cheat-sheet set extended to 81 templated questions, and **all 544 single-table dev
queries of Spider** across 20 real databases. Against a full baseline suite — structural and
semantic self-consistency, predictive and semantic entropy — our confidence achieves the best
risk–coverage tradeoff at scale (AURC 0.094 vs 0.17 for the strongest baseline on Spider; 0.006
vs 0.048 on the easy set). It is the only method that supports risk-controlled abstention where
self-consistency cannot, and a Bonferroni conformal procedure yields a valid distribution-free
certificate (selective risk ≤ 0.20 at 51% coverage). We are deliberate about the limits: a
naive skeleton-level variant *fails* on hard data; the advantage over semantic-execution
self-consistency is **distribution- and scale-dependent** (it shrinks on a narrow 60-question
slice); model strength barely changes the picture because the fragment is near the accuracy
ceiling; and confidently-wrong, self-agreeing errors are invisible to *all* sampling-based UQ.

---

## 1. Introduction

A text-to-SQL model maps a natural-language question and a database schema to an SQL query.
Modern systems are LLM-based and strong on classic benchmarks but far from solved on hard ones
(Spider 2.0: ~21% for a leading agent). In deployment — analytics copilots, and risk-sensitive
domains such as actuarial data pipelines — the operationally critical question is not only *can
the model answer* but *can we tell when to trust it*, so we can abstain or route to a human.

Yet a calibrated confidence over a generated query is an open problem. The recognized methods
flatten the structured output:

- **Self-consistency / `top_prob`**: sample $K$ queries, confidence = fraction equal to the
  modal query. Simple, strong, model-agnostic, but **saturates** (it returns 1.0 whenever the
  $K$ samples agree, even when they agree on a *wrong* query).
- **Sequence probability / verbalized confidence**: the model's own token likelihood, or a
  self-reported number. Poorly calibrated for structured outputs.
- **Semantic / structural entropy**: entropy over the sampled outputs, optionally clustered by
  meaning. For code/SQL the meaning-aware version is strong.

Separately, SQL has a long tradition of **graph/AST representations**, but only on the
*encoder* side (RAT-SQL, LGESQL): no prior work places a probability measure on the *output*
query structure. We close that gap. The contribution is to make the object of uncertainty the
**query structure itself**, equip it with a Bayesian nonparametric prior, and read uncertainty
off the posterior.

**Contributions.**
1. A typed **query-graph** representation with two canonical fingerprints (full structure and
   column-abstracted skeleton), and an exact characterization of the executable query space for
   a single-table fragment (§3, Prop. 1).
2. **Pitman–Yor Bayesian self-consistency** (§4): a PY posterior over full query structures,
   yielding a de-saturated confidence and a closed-form **discovery probability**.
3. A **structure-localized** uncertainty readout separating *which query shape* from *which
   columns/aggregates* (§4.4).
4. A careful **empirical study with execution accuracy** (§6–7): a full baseline comparison, a
   negative result on structural granularity, valid conformal certificates, a cross-model
   sweep, and an explicit account of what sampling-based UQ fundamentally cannot do.

---

## 2. Background and related work

**Text-to-SQL.** Benchmarks: Spider (cross-domain, schema-rich), BIRD (dirty values, external
knowledge), Spider 2.0 (enterprise). Standard metric: **execution accuracy** — the predicted
query is correct iff it returns the same result set as a gold query on the database. We adopt
it (§6.3) because string/structure matching wrongly penalizes valid paraphrases.

**UQ for LLMs and code.** Self-consistency / sampling, verbalized confidence, conformal
abstention (RTS, over schema-linking branch points), semantic entropy (cluster by meaning),
and AST/structural entropy for code. All operate on token sequences, sub-clauses, or post-hoc
scalars; none place a generative Bayesian posterior over the output structure.

**Bayesian nonparametrics over structures.** A survey of candidate priors (DP/Pitman–Yor urns,
nested CRP, tree-stick-breaking, adaptor grammars, feature-allocation/IBP, constrained-DAG
posteriors) identified the **Pitman–Yor process / adaptor grammar** family as the right fit:
its support is exactly the grammar-valid structures, and its predictive rule carries explicit
open-world mass. We use a PY *urn* over observed structures directly (the species-sampling
view), reserving the full adaptor grammar for the multi-table extension (§8).

---

## 3. Problem formulation

### 3.1 Query graphs and canonicalization

We parse each SQL string (via `sqlglot`) into a typed directed graph $g(q)=(V,E)$:

- **Node types**: `query` (a SELECT scope), `table`, `column`, `literal`, `function`
  (aggregate/scalar), `operator` (comparison/logical/arithmetic), `clause` (WHERE, GROUP BY,
  HAVING, ORDER BY, LIMIT), `join`, `set_op`.
- **Edge roles**: `from`, `select`, `clause`, `cond`, `key`, `arg`, `operand`, etc.

A canonicalization $\kappa$ quotients out semantically irrelevant surface variation: (i) table
**alias names**, (ii) **commutative reordering** of `AND`/`OR`/`=`, (iii) **literal values**
(kept as typed placeholders `num`/`str`). From $\kappa(g(q))$ we derive two hashable
fingerprints:

- **Canonical key** $c(q)$ — the full structure, *column- and function-concrete* (e.g.
  `AVG(number_of_rooms)` and `GROUP BY country` are distinguished by their columns). This is
  the object whose distribution defines our headline confidence.
- **Skeleton key** $s(q)$ — additionally abstracts column names to a placeholder and functions
  to a category (`AGG`/`SCALAR`), retaining only the *shape* (which clauses, predicate-tree
  topology, aggregate-vs-plain pattern). Schema-independent; used for localization.

Two queries differing only by alias, predicate order, or literal value share a canonical key;
two differing only by *which columns* they touch share a skeleton but not a canonical key.

### 3.2 The executable query space

We fix a single-table fragment (the DataCamp "SQL Basics" set): `SELECT [DISTINCT]`, `FROM`
(one table), `WHERE` (`= > >= < <=`, `BETWEEN`, `IN`, `LIKE`, `IS [NOT] NULL`, `AND`/`OR`),
`GROUP BY`, `HAVING`, `ORDER BY` (`ASC`/`DESC`), `LIMIT`, and aggregates `SUM/AVG/MIN/MAX/COUNT`.
Let $\mathcal Q(S)$ be the set of canonical structures executable against schema $S$, subject to
type/scope constraints $\Phi$ (numeric ops on numeric columns; GROUP BY consistency; HAVING
requires aggregation; ORDER BY keys in scope).

**Proposition 1.** *Modulo literal values, $\mathcal Q(S)$ is the set of clause tuples
$(\pi,\varphi,\gamma,\eta,\omega,\ell)$ admissible under the fragment grammar and satisfying
$\Phi$. It is countable, and finite once the WHERE/HAVING predicate trees are bounded in depth
$D$ and fan-in $W$.* (Proof sketch: every clause but the boolean predicate ranges over a finite
set given $S$; bounding $(D,W)$ truncates the predicate to a finite set.)

For the running `airbnb_listings` schema (5 columns: `id`, `city`, `country`,
`number_of_rooms`, `year_listed`; 3 numeric, 2 text), a closed-form count gives
$|\mathcal Q(S)|\approx 4.1\times10^6$ at tight bounds and $\approx 8.5\times10^{10}$ at looser
bounds — finite but far too large to enumerate, which is precisely why a *concentrating* prior
plus the LLM as a proposal is the right machinery. Dropping joins is what makes $\mathcal Q(S)$
characterizable; the multi-table extension (§8) re-introduces context-sensitive type/key
constraints and is left to the adaptor-grammar generalization.

### 3.3 What we quantify

For a question $x$, the LLM induces a distribution over queries; sampling $K$ times yields
$q_{1:K}$. We seek a posterior over the *correct* structure $Q^\star$ and a scalar **confidence**
$\widehat c(x)\in[0,1]$ that is (a) discriminative (ranks correct above incorrect), (b)
de-saturated (not pinned to 1.0 under agreement), and (c) open-world (assigns positive mass to
unseen structures). We then use $\widehat c$ for **selective prediction**: answer the modal
query iff $\widehat c \ge \tau$, choosing $\tau$ to control risk.

---

## 4. Method

### 4.1 Generative model (Model A: the LLM-wrapper posterior)

We treat the $K$ samples as i.i.d. draws from the model's latent per-question structure
distribution $G$, and place a Pitman–Yor prior on $G$:

$$
G \sim \mathrm{PY}(d,\theta,H), \qquad c(q_1),\dots,c(q_K)\mid G \overset{iid}{\sim} G,
$$

with discount $d\in[0,1)$, concentration $\theta>-d$, and base measure $H$ over canonical
structures. $H$ encodes which query shapes are a priori plausible (fit by empirical Bayes,
§4.5); $d=0$ recovers a Dirichlet process. This is the species-sampling view: each distinct
sampled structure is a "species", and the PY urn governs how mass concentrates on repeats vs.
novelty.

### 4.2 Headline confidence and discovery probability

Marginalizing $G$, the **posterior-predictive** probability of a structure $c$, after observing
$K$ samples with structure $c$ seen $n_c$ times among $\kappa$ distinct structures, is the
two-parameter urn:

$$
\widehat P(c \mid q_{1:K}) \;=\; \frac{n_c - d}{\theta + K}\;\mathbf 1[n_c>0]\;+\;\frac{\theta + d\kappa}{\theta+K}\,H(c).
$$

Our **confidence** is this mass evaluated at the modal (MAP) structure $\hat c=\arg\max_c n_c$:

$$
\boxed{\;\widehat c(x) \;=\; \widehat P(\hat c \mid q_{1:K})\;}
$$

Two properties matter. (i) **De-saturation**: even with unanimous samples ($n_{\hat c}=K$,
$\kappa=1$), $\widehat c = \frac{K-d}{\theta+K} + \frac{\theta+d}{\theta+K}H(\hat c) < 1$ — the
prior reserves mass for the possibility that agreement is coincidental. The frequency baseline
returns exactly $1$ here. (ii) **Discovery probability** — the posterior mass on a structure
*not yet observed*:

$$
\mathrm{disc}(x) \;=\; \frac{\theta + d\kappa}{\theta + K}\Big(1 - \!\!\sum_{\text{seen }c} H(c)\Big),
$$

a closed-form, Good–Turing-style answer to *"what is the probability the correct query is a
structure the model never sampled?"* — which frequency methods structurally assign $0$. We use
a de-rated headline score $\widehat c(x)\,(1-\mathrm{disc}(x))$ that folds in the open-world
penalty; the de-rating gives a small but consistent improvement.

### 4.3 Why full structures, not skeletons (a design decision validated empirically)

The same urn can be run over *skeletons* $s(q)$ instead of canonical structures $c(q)$. The
skeleton urn is coarser (column-blind). We initially used it and found (Table 1) that it wins on
easy data but **loses badly on hard data**, because on hard questions the model's errors are
"right shape, wrong column", invisible to a column-blind score. We therefore run the headline
confidence over **full canonical structures** and reserve the skeleton level for localization.

### 4.4 Localization (which component is uncertain)

For interpretability we additionally maintain (a) a PY urn over skeletons, and (b)
Dirichlet–multinomial posteriors over **binding slots** — `projection`, `filter_columns`,
`where_ops`, `group_columns`, `order_keys`, `agg_functions`, `has_having`, `has_limit` —
extracted from each sample. Slot $r$ with filler counts $c_{r,v}$ over $K$ samples has predictive
$\widehat P(b_r=v)=\frac{\alpha_r u_{r,v}+c_{r,v}}{\alpha_r+K}$ and a per-slot entropy. This
yields a chain-rule decomposition $\mathbb H(\text{query}) = \mathbb H(\text{skeleton}) +
\sum_r \mathbb E[\mathbb H(\text{slot }r)]$, so uncertainty can be reported as *"confident on the
table and grouping, unsure which aggregate"* — a readout no scalar baseline provides. (This is a
secondary output; it does not feed the headline score, which §4.3 shows must stay at full-structure
granularity.)

### 4.5 Empirical-Bayes hyperparameters

We fit, **offline on training gold queries only** (no test labels): (i) the base measures
$H$ (over gold canonical structures) and $u_r$ (over gold slot fillers) as smoothed empirical
frequencies; (ii) the urn parameters $(d,\theta)$ by maximizing the Pitman–Yor EPPF of the
**per-question sample partitions** (the multiset of structures each question's $K$ samples form).
The last point is a subtlety we got wrong initially: $(d,\theta)$ must be fit at the
*within-question* scale (how diverse are $K$ samples for one question), not the *cross-question*
scale (how diverse are queries across the corpus) — conflating them crushes confidence. Fitted
values are small ($d\approx0.04$, $\theta\approx0$), i.e. close to a Dirichlet-process urn.

### 4.6 Selective prediction and conformal risk control

Given a calibration set of (question, gold) pairs and confidence scores, we choose a threshold
$\tau$ and answer iff $\widehat c(x)\ge\tau$. We report three regimes:

- **Empirical / marginal**: smallest $\tau$ whose calibration selective-risk $\le\alpha$; the
  held-out risk is a valid in-expectation estimate under exchangeability.
- **Hoeffding** and **exact-binomial LTT** (fixed-sequence): distribution-free PAC certificates.
  We compute the binomial CDF in log-space (no scipy); the exact binomial tail is provably
  tighter than Hoeffding.
- **Bonferroni-over-grid**: tests a grid of nested answer-sets at level $\delta/G$ and returns
  the max-coverage certified threshold. Robust to a non-monotone confidence (one noisy small
  top-bucket does not abort the procedure, as it does for fixed-sequence LTT).

---

## 5. Implementation

The method is a small Python library (`src/bnp_nl2sql/`, 43 unit tests):

- `query_graph.py` — SQL → typed graph; `canonical_key`, `skeleton_key`.
- `pyp.py` — Pitman–Yor urn: predictive, discovery probability, log-EPPF.
- `posterior.py` — Model A: dual urns (skeleton + full), Dirichlet slots, `confidence()`,
  `discovery_probability`, localization, abstain rule.
- `fit.py` — EPPF maximization for $(d,\theta)$, empirical base measures, a logistic
  meta-calibrator.
- `uq_baselines.py` — structural/semantic self-consistency, predictive/semantic entropy.
- `calibrate.py` — risk–coverage, AURC, Hoeffding/LTT/Bonferroni certificates.
- `execeval.py` — execution-accuracy matching on SQLite.

Inference is $O(K)$ per question plus the parse; no model training, no gradient. The LLM is a
black box accessed only through its samples.

---

## 6. Experimental setup

### 6.1 Datasets

**airbnb (easy, controlled).** A single `airbnb_listings(id, city, country, number_of_rooms,
year_listed)` table. 31 (question, gold-SQL) pairs transcribed verbatim from the DataCamp "SQL
Basics" cheat sheet, plus 50 templated pairs spanning projection / numeric & text filters /
aggregates / group-by / order-by / limit / distinct / between / having / a few deliberately
ambiguous phrasings — **81 questions total**, every gold verified to execute. A 60-row SQLite
database is seeded deterministically with duplicate cities per country (so `COUNT` vs
`COUNT DISTINCT` differ) and ~10% NULL `number_of_rooms` (for the `IS NULL` questions).

**Spider single-table (real, external).** From the 1,034 Spider dev questions we keep those
whose gold SQL references exactly one table with no join and no subquery — **544 questions
across 20 databases**. Real, human-vetted golds. The 20 SQLite databases are downloaded from
the `premai-io/spider` HuggingFace mirror for execution accuracy.

### 6.2 LLM sampling protocol

For each question we draw $K=8$ samples at temperature $0.7$ via the OpenAI chat API in a
single call (`n=8`), with a system prompt giving the schema (`CREATE TABLE`-style column list)
and instructing a single one-line SQLite query, no prose. Samples are cached on disk (re-runs
cost nothing). The primary model is **gpt-4o-mini**; the cross-model sweep (§7.4) adds **gpt-4o**
and **gpt-3.5-turbo** on a 60-question slice. Every run is preceded by a token-based cost
estimate and is gated behind an explicit flag and a hard call cap. **Total API spend for the
entire study: ≈ \$0.26.**

### 6.3 Correctness: execution accuracy

A predicted query is **correct** iff, executed against the database, it returns the same
multiset of rows (value-equality, column-name- and order-insensitive) as the gold query.
This is essential: structural matching scores 0.839 on airbnb but execution scores 0.926,
because the model's "errors" include valid paraphrases — `MAX(x)` vs `ORDER BY x DESC LIMIT 1`,
`COUNT(city)` vs `COUNT(DISTINCT city)`, and pure alias renaming (`avg_rooms` vs `average_rooms`,
a *false* structural error). We report execution accuracy throughout.

### 6.4 Baselines

All from the same $K$ samples: **structural self-consistency** (`top_prob` over canonical
keys), **semantic self-consistency** (`top_prob` over *execution-result* clusters — the
strongest, meaning-aware baseline), **predictive entropy** (over canonical structures),
**semantic entropy** (over execution clusters). We also evaluate a **skeleton-level** PY variant
and a **logistic meta-combination** of all signals.

### 6.5 Metrics

**AURC** (area under the risk–coverage curve; lower is better) summarizes how well a confidence
ranks errors. **Selective risk @ coverage** under split-conformal calibration (calibrate on one
half, evaluate held-out). **Certified frontier**: the (risk target $\alpha$, max coverage) pairs
with a valid distribution-free guarantee at $\delta=0.1$.

### 6.6 Reproducibility

`scripts/`: `gen_eval.py` (build airbnb set), `make_db.py` (seed DB), `sample_openai.py` /
`spider_benchmark.py` (sampling, safe-by-default with cost estimate + cap), `run_benchmark.py`
(airbnb eval), `compare_baselines.py` (full comparison + meta + certificate), `model_sweep.py`
(cross-model), `ablate_scores.py` (granularity ablation). Lit/theory: `paper/lit_review.md`,
`paper/theory.md`, `paper/methods.md`.

---

## 7. Results

### 7.1 Main comparison (Table 1)

AURC (lower is better):

| Confidence | airbnb (n=81, acc .926) | Spider s-t (n=544, acc .805) |
|---|---|---|
| structural self-consistency `top_prob` | 0.049 | 0.172 |
| semantic self-consistency (execution) | 0.062 | 0.172 |
| predictive entropy | 0.048 | 0.170 |
| semantic entropy (execution) | 0.062 | 0.171 |
| skeleton-level BNP (naive) | 0.007 | 0.306 |
| logistic meta-combine | 0.023 | 0.128 |
| **PY full + discovery (ours)** | **0.006** | **0.094** |

At full scale our confidence is best on both datasets, beating even semantic-execution
self-consistency by ~45% on Spider. The advantage **widens with scale** (on an 80-question
Spider slice it was 0.153 vs 0.211; at 544 it is 0.094 vs 0.172).

### 7.2 The granularity result (a designed negative)

The skeleton-level variant is best on easy airbnb (0.007) but worst on hard Spider (0.306):
column-blind confidence cannot see "right shape, wrong column" errors. This drove the decision
to score at full-structure granularity (§4.3). The logistic meta-combination also
*underperforms* the single PY signal (0.128 > 0.094): combining dilutes the strong signal with
noisier ones. We keep both negatives in the paper.

### 7.3 Selective prediction and certificates (Spider, n=544)

Split-conformal, target selective risk $\le 0.10$, calibrate on 272 / test on 272:

| | coverage | held-out risk |
|---|---|---|
| **ours** | **0.51** | **0.079** |
| baseline (`top_prob`) | 0.00 (cannot) | — |

The frequency baseline **cannot offer any risk-≤0.10 operating point**: its most-confident
bucket (`top_prob`=1.0, unanimous samples) already exceeds 10% error because of
confidently-wrong queries, so no threshold isolates a safe subset. De-saturation is exactly
what fixes this. **Distribution-free certificate** (Bonferroni, $\delta=0.1$): selective risk
$\le 0.20$ at **51% coverage / 7.9% held-out risk**. Tighter targets ($\alpha\le 0.15$) are not
certifiable at $n=544$ — the **confident-wrong floor**: the top-confidence decile still carries
~15% error. (Fixed-sequence LTT certifies nothing here, being fragile to a noisy small
top-bucket; Bonferroni is the robust choice. Improving score *resolution* — a meta score with
81 vs 28 distinct values — did **not** unlock tighter certificates, confirming the binding
constraint is the floor, not resolution.)

### 7.4 Cross-model robustness

We test whether the result is model-specific by sampling gpt-4o and gpt-3.5-turbo. The headline
comparison is at **matched scale (full 544)** against the strongest baseline, semantic-execution
self-consistency (AURC, lower is better):

| model | n | exec acc | AURC ours | AURC semantic | AURC structural |
|---|---|---|---|---|---|
| gpt-4o-mini | 544 | 0.805 | **0.094** | 0.172 | 0.172 |
| gpt-4o | 544 | 0.825 | **0.100** | 0.151 | 0.156 |
| gpt-3.5-turbo | 60* | 0.667 | 0.339 | 0.281 | 0.346 |

\*gpt-3.5-turbo sampled on a 60-question slice only. **On both gpt-4o-mini and gpt-4o at full
scale, our PY confidence beats semantic-execution self-consistency (≈34–45% lower AURC) and
supports a valid certificate (≈0.5 coverage at ~10% risk)** — so the advantage is *not*
model-specific. (gpt-4o's stronger semantic baseline, 0.151 vs mini's 0.172, narrows the margin
slightly, as expected for a more self-consistent model.)

A cautionary sub-result: on a small, narrow 60-question / 3-database slice the same comparison
is roughly tied (e.g. gpt-4o-mini ours/semantic = 0.25/0.26 vs 0.09/0.17 at full scale). The
advantage is therefore **scale- and breadth-dependent** — largest on the broad distribution where
confidence saturation is common, and within noise on a narrow hard slice. We also note **model
strength barely moves accuracy** on this fragment (gpt-3.5 0.67, gpt-4o-mini 0.81, gpt-4o 0.83):
single-table SQL is near the ceiling, which is itself a reason to move to multi-table (§8).

### 7.5 The fundamental limit (of *sampling-based* UQ)

When the model returns one wrong query unanimously ($\kappa=1$), no sampling-based method — ours,
frequency, or semantic entropy — can flag it: there is no disagreement signal, and the discovery
probability is near zero. On Spider this floor is large: **453 of 544 questions are unanimous**,
with 16.3% error, and it is what caps the distribution-free certificate (§7.3).

### 7.6 Lifting the floor with token log-probabilities (white-box)

The floor is a limit of *sampling*, not of uncertainty per se: even when the $K$ samples agree,
the model's **mean token log-probability** of its output may still be lower for a wrong query. We
test this by re-sampling Spider with `logprobs=True` and measuring whether log-prob separates
correct from incorrect *within the unanimous ($\kappa=1$) subset* — exactly where sampling is
blind. It does, strongly: **AUROC 0.79**. Consequences (Table 2):

| Confidence (Spider, n=544) | AURC | role |
|---|---|---|
| PY full + discovery (ours, black-box) | 0.097 | best **sampling-only** signal |
| sequence log-probability (white-box) | **0.074** | stronger, but needs logit access |
| PY + log-prob (logistic) | 0.089 | combine helps PY, fusion still suboptimal |

Two honest conclusions. (i) **When logits are available, sequence log-probability is a stronger
single signal than our method.** We therefore position the PY posterior as the best **black-box**
method (the regime with sampling access only — common for hosted/reasoning models that hide
log-probs), and as the source of two signals log-prob *cannot* provide: the **open-world
discovery probability** and **component localization**. (ii) The two are **complementary** —
log-prob detects confident-wrong errors, PY detects diverse-sample uncertainty — and combining
lifts the Bonferroni certificate to **68% coverage at risk $\le0.20$** where PY alone abstains.
A better-than-logistic fusion of structural (PY), likelihood (log-prob), and semantic (execution)
signals is a clear next step. Caveat: this is short single-table SQL, where log-prob is unusually
well-behaved; its calibration is known to degrade on longer queries (length/frequency confounds),
so the white-box advantage may not survive into multi-table.

---

### 7.7 Where the Bayesian machinery earns its keep: open-world detection

A sequence of mechanism ablations (all on cached data) attributes each effect to the right
component, and the result is sharp:

- **Ranking (AURC) is driven entirely by the base measure $H$.** With a uniform base, the
  method collapses to the baseline (0.172); the PY discount ($d$ vs DP) and the discovery
  de-rating move AURC by $<10^{-3}$. So the ranking win is *prior-weighted self-consistency*.
  It is **not leakage**: cross-fitting $H$ (fit on one half of golds, score the other) gives
  AURC 0.096 vs in-sample 0.094 — the structural prior generalizes.
- **Calibration is not a durable differentiator.** De-saturation lowers raw ECE
  (0.158 → 0.138, more at small $K$), but a one-parameter post-hoc Platt scaling equalizes
  *all* methods to ECE $\approx 0.017$. Since scaling is monotone it cannot change ranking, so
  the durable, non-replicable advantage is AURC ($H$), not calibration.
- **The discovery probability is the genuinely Bayesian payoff.** It detects *open-world
  failure* — questions whose correct query was **never sampled** (39% of Spider; answering is
  then guaranteed wrong) — at **AUROC 0.868 (0.839 out-of-sample)**, versus **chance** for every
  disagreement-based signal (1−top_prob 0.514, #distinct 0.518, 1−semantic 0.557). Crucially,
  **172 of these 212 cases are unanimous ($K{=}1$)** — the confident-wrong floor — which no
  disagreement signal can see. Discovery cracks it through $H$: a unanimous-*wrong* query tends
  to use a *rare* structure (low $H$ → high discovery), a unanimous-*correct* query a *common*
  one (high $H$ → low discovery). This does not *fix* the wrong answer (the point-prediction
  limit of §7.5 stands) but it enables correct **abstention** where every other sampling signal
  is blind. This is the capability that justifies the nonparametric posterior.

## 8. Discussion: where this stands, and the option space

**What is solid (and mechanism-attributed by ablation).** (1) A learned **structural prior**
over canonical query graphs reweights self-consistency into the best black-box selective-
prediction *ranking* (AURC 0.094 vs 0.172), confirmed on two models and **out-of-sample**
($H$ cross-fit), and not replicable by any monotone post-hoc calibration. (2) The **discovery
probability** is a genuinely unique capability: it detects open-world failures (correct query
never sampled), including the unanimous confident-wrong cases, at **AUROC 0.84 out-of-sample vs
chance** for all disagreement baselines — enabling abstention where every other sampling signal
is blind. (3) A working, tested, cheap (~\$1.7 total) pipeline with execution-accuracy
evaluation, valid conformal certificates, and clean ablations attributing each effect to the
right component. The honest counterpoint: the *calibration* gain is post-hoc-achievable, and the
ranking win is "prior-weighted self-consistency" — the nonparametric apparatus earns its keep
through (2), not through AURC.

**What is genuinely uncertain.** (a) The margin over semantic-execution self-consistency, while
**now confirmed across two models at full scale** (gpt-4o-mini and gpt-4o, §7.4), is
scale/breadth-dependent — it shrinks to a tie on narrow slices, so the claim is "best at scale on
the broad distribution", not "uniformly best everywhere". (b) The single-table fragment is near
the accuracy ceiling, so it under-stresses UQ; the regimes where UQ matters most (joins,
enterprise schemas, Spider 2.0) are exactly the ones we have not entered. (c) Distribution-free
certificates are loose, capped by the confident-wrong floor.

**Option space (for choosing direction).**
- **Confirm-and-strengthen [DONE]:** the full-544 gpt-4o run (§7.4) confirms ours > semantic at
  scale on a second model. The headline empirical claim is robust; the natural next moves are
  below.
- **Raise the difficulty:** extend to multi-table Spider via a Pitman–Yor **adaptor grammar**
  over a typed SQL grammar (the planned generalization), where errors are richer and UQ matters
  more — higher upside, more method work (context-sensitive type/key constraints).
- **Attack the floor [DONE, §7.6]:** token log-probs detect confidently-wrong errors (AUROC 0.79
  on the unanimous subset) and lift the certificate to 68% coverage. But they also *beat* our
  method as a standalone signal — so the durable framing is "best black-box method + the
  open-world/localization signals log-prob lacks", and the open task is a better structural ×
  likelihood × semantic **fusion** (the logistic combine is suboptimal).
- **Reframe the contribution:** if the margin over semantic self-consistency proves fragile, the
  durable contributions are the **open-world discovery probability** and **de-saturation enabling
  abstention** (both things semantic self-consistency lacks), rather than raw AURC — a narrower
  but defensible claim.
- **The big bet (future work):** drop the LLM wrapper and *train* a structured predictor with the
  BNP prior as regularizer — the "fund the training job" path the PoC is meant to justify.

---

## 9. Limitations

Single-table fragment (no joins); modest $n$ for certificates; execution accuracy depends on
gold fidelity (some cheat-sheet golds are arguably worse than the model's answer); the marginal
conformal guarantee is in-expectation, the PAC certificate is loose; the cross-model sweep is a
60-question slice. None of these are hidden — they define the next experiments.

## 10. Conclusion

Casting text-to-SQL uncertainty as Bayesian inference over query structures turns
self-consistency into a calibrated, open-world signal: it de-saturates the frequency estimate,
quantifies the probability the correct query was never sampled, and localizes uncertainty to
query components. At scale it gives the best risk–coverage among standard baselines and the only
risk-controlled abstention. The honest counterweight — a strong semantic-execution competitor,
a near-ceiling fragment, and a confident-wrong floor — sharpens rather than negates the result,
and points to concrete next steps (matched-model confirmation, multi-table difficulty, a
white-box signal for the floor, and ultimately a trained model).
