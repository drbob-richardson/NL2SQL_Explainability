# Paper 2 Direction — BNP, Ambiguity, and Decision-Making for Text-to-SQL

**Status:** exploratory findings, not a paper draft. Records what we tested, the numbers, and
the resulting verdict on whether a Bayesian-nonparametric (BNP) / decision-theoretic
contribution is viable on top of Paper 1.

**Date of run:** 2026-06 · **Total API spend across all probes:** ≈ $1.66 (gpt-4o-mini + gpt-4o).

---

## 1. Motivation and framing

Paper 1 ("What Predicts Correctness in Text-to-SQL?") established that for *correctness*,
black-box agreement signals plateau (~0.61–0.68 AUROC) and only a reasoning **verifier** breaks
the ceiling (GPT-4o 0.77, two-provider ensemble 0.82). The open question for a second paper was
whether BNP has a genuine, load-bearing role somewhere adjacent.

We split the candidate project into two linked problems:

1. **Correctness checking** — given a candidate SQL, how likely is it correct?
2. **Decision-making** — given uncertainty over interpretations, should the system execute a
   particular query, ask a clarifying question, or abstain?

The proposal under test placed a **BNP prior over canonical SQL query graphs**
`P(G | x, S, E)`, with per-node posteriors (tables, columns, joins, aggregations) feeding both a
calibrated correctness model and a Bayes-decision rule (expected-loss over graph hypotheses,
value-of-information clarification, abstention).

The goal of this exploration was explicitly to find out **how much of this our own data already
rules out**, and **where (if anywhere) the nonparametrics are load-bearing rather than
decorative**.

---

## 2. Literature landscape (gap analysis)

Verified against primary sources (arXiv, ACL Anthology, NeurIPS, GitHub).

**Ambiguity benchmarks**
- **AmbiQT** — Bhaskar et al., EMNLP 2023, arXiv:2310.13659. 4 ambiguity types (column, table,
  join/table-split, precomputed-aggregate); ~3,046 examples, two gold SQLs each; synthetic
  *schema* perturbations of Spider; metric = coverage (both golds in top-k).
  github.com/testzer0/AmbiQT.
- **AMBROSIA** — Saparina & Lapata, NeurIPS 2024 D&B (spotlight), arXiv:2406.19073. 3 types
  (**scope, attachment, vague**); 1,277 ambiguous questions, 846 multi-table DBs / 16 domains;
  each question has multiple valid interpretations + SQL. CC BY 4.0. github.com/saparina/ambrosia.
  **This is the benchmark we used.**
- Also: PRACTIQ (NAACL 2025, arXiv:2410.11076, conversational), AmbiSQL (SIGMOD 2026 demo,
  arXiv:2508.15276).

**Clarification / when-to-ask**
- MISP (Yao et al., EMNLP-IJCNLP 2019, arXiv:1910.05389) — asks on per-component confidence
  threshold or MC-dropout uncertainty.
- DialSQL (Gur et al., ACL 2018) — learned error-detection policy.
- **Qiu et al., "Interactive Text-to-SQL via Expected Information Gain" (arXiv:2507.06467)** —
  the **nearest competitor** to our decision angle: maintains a distribution over candidate SQLs
  and picks clarifications by EIG. *But clarify-only — no abstention, no prior.*

**Abstention / selective prediction**
- TrustSQL (arXiv:2403.15879, penalty-based abstention), RTS (arXiv:2501.10858, abstain-or-ask,
  error-driven), Somov & Tutubalina (AAAI 2025, arXiv:2501.09527, entropy selective classifier).

**Materiality via execution**
- SOMA-SQL (arXiv:2606.11424, **June 2026, ~10 days before our run** — author list unverified):
  uses candidate disagreement + execution probing for *resolution*. Adjacent to our
  execution-divergence idea; differentiate on *calibrated materiality*.

**The three-part gap (white space)**
1. A prior-driven **posterior over SQL interpretations** — closest is Stengel-Eskin et al., ICLR
   2024, arXiv:2306.00824, but in FOL/Lisp, not SQL. *Not found for text-to-SQL.*
2. A **unified Bayes/expected-loss objective over execute-vs-clarify-vs-abstain** — EIG
   (clarify-only) and TrustSQL (abstain-only) exist separately, never combined. *Not found.*
3. **Calibrated result-set-divergence materiality** — adjacent to SOMA-SQL; narrow carve-out.

**BNP / open-world priors for ambiguity or novel-intent in text-to-SQL: CONFIRMED ABSENT.**
DP/CRP open-intent work exists only in dialogue intent classification, never semantic parsing.

---

## 3. Probes and results

All scripts under `scripts/`; analyses use cached data + the real databases.

### Probe 1 — Do BNP graph-posterior features add anything to the verifier for *correctness*?
`scripts/bnp_probes.py` (probe 1). BIRD, n=800. Features computed from the 8 samples per question
(skeleton entropy, structural modal mass, #distinct skeletons, node-minimum slot posterior),
compared and combined with the GPT-4o verifier via cross-fit logistic.

| signal | AUROC (correctness) |
|---|---|
| skeleton entropy | 0.589 |
| structural modal mass | 0.584 |
| node-minimum posterior | 0.592 |
| # distinct skeletons | 0.653 |
| structural self-consistency (ref.) | 0.62 |
| **GPT-4o verifier** | **0.770** |
| verifier + graph-posterior features | 0.742 |

Paired Δ (verifier+graph − verifier) = **−0.028, 95% CI [−0.054, −0.004]**.

**Reading:** the graph posterior carries the same structural information as self-consistency,
which sits well below the verifier and adds **no marginal value** (the combiner here is weaker
than Paper 1's, so we read the result as "no gain," not "it hurts"). **BNP-for-correctness is
retired.** This matches Paper 1's prediction: errors are computational, not structural, so a
structural posterior cannot see them.

### Probe 2 — Is the gold-query motif distribution open-world? (PYP discovery)
`scripts/bnp_probes.py` (probe 2). Gold queries pooled from BIRD + multi-table Spider (N=1290),
clustered into motifs at two granularities; Pitman–Yor (d, θ) fit by ML on the partition.

| motif level | K motifs | singletons | top-1 cov. | PYP d | θ | next-query discovery prob |
|---|---|---|---|---|---|---|
| **skeleton** (exact shape) | 909 | 69% of motifs / 48.6% of queries | 0.7% | **0.160** | 1101.7 | **0.521** |
| clause-set (8 binary flags) | 44 | 25% of motifs | 24.5% | 0.210 | 3.8 | 0.010 |

**Reading:** at **skeleton granularity the world is genuinely open** — power-law tail (d > 0),
~half of queries are the only instance of their structure, and a new query has a ~52% chance of a
never-seen skeleton. At coarse clause-flag level it is closed. **Caveat:** these are *benchmark*
gold queries curated for diversity, so 52% is an **upper bound** vs a real (repetitive) workload.
The nonparametrics are load-bearing here — d=0.16 is a real power law, not decoration.

### Probe 3 — Does the model posterior split into *materially* different answers? (BIRD)
`scripts/ambiguity_probe.py`. Executed the 8 samples per BIRD question against the real DBs;
clustered by result set. BIRD is unambiguous by design, so splits here are mostly *error*.

| | share | modal accuracy |
|---|---|---|
| unanimous (1 result cluster) | 62% | 0.581 |
| split (≥2 result clusters) | 38% | 0.236 |
| materially divergent (≥2 answers, each ≥2/8) | 21% | 0.247 |

- **Immaterial variation: 38%** of questions produce multiple distinct SQL *strings* that execute
  to the *same* result set (string self-consistency penalizes these; execution clustering does not).
- On materially-divergent questions: gold is **among** the candidate clusters only **37%**, gold
  is the modal cluster **24%**.
- Localization: split is a **single slot only 26%** of the time (loci: projection, filter_columns,
  where_ops dominate).

**Reading:** the machinery works (detect divergence, localize, measure materiality), but on BIRD
divergence is dominated by error (gold recoverable only 37%), confirming we need a benchmark where
the multiple answers are *genuinely valid* → AMBROSIA. The **38% immaterial-variation** result is a
clean, standalone critique of string self-consistency.

### Probe 4a — AMBROSIA gold interpretations: materiality and localization (no API)
`scripts/ambrosia_probe.py`. 1,277 ambiguous questions; executed the *provided* valid
interpretations against the real DBs.

| ambiguity type | n | interpretations give different answers | single-slot localizable |
|---|---|---|---|
| attachment | 362 | 100% | 0% |
| scope | 501 | 100% | 0% |
| vague | 414 | 100% | 83% |
| **overall** | 1277 | **100%** | 27% |

**Reading:** materiality is **total** — choosing the wrong interpretation always changes the
answer (the precondition the decision layer needs, which BIRD lacked). Localization is
type-dependent: *vague* is lexical/slot-localizable (83%); *scope/attachment* are **structural**
(the interpretations differ in several coupled slots — projection+group+agg+having move together),
so their clarification unit is the **interpretation**, not a single slot. AMBROSIA ships NL
`interpretation1/2` strings for exactly that.

### Probe 4b — Does the model's *sampling* posterior surface the ambiguity? ($0.66)
`scripts/ambrosia_generate.py` (gpt-4o-mini, K=8 via n=8) + `scripts/ambrosia_coverage.py`. 1,149
ambiguous test Qs + 1,000 unambiguous controls; samples executed and matched to the two golds.

| type | n | both | one | none |
|---|---|---|---|---|
| attachment | 326 | 0% | 29% | 71% |
| scope | 450 | 0% | 74% | 26% |
| vague | 373 | 2% | 33% | 65% |
| **all** | 1149 | **1%** | 48% | 52% |

Detection: mean #distinct result clusters/8 — ambiguous **1.55** vs control **1.66**; collapse
(all 8 → 1 result) 64% vs 60%; **AUROC of divergence predicting is_ambiguous = 0.475 (≈ chance)**.
Control single-interpretation accuracy 0.437 (sanity OK).

**Reading:** the generator **collapses** — it commits to one reading and agrees with itself.
Ambiguity **cannot be read off the sampling posterior**, and result-set divergence is **not** an
ambiguity detector. This is Paper 1's lesson, sharper: the model is *confidently single-minded* on
ambiguous questions.

### Probe 4c — Does *explicit elicitation* recover both interpretations? ($0.19 + $0.81)
`scripts/ambrosia_elicit.py`. Same task, but the model is told the question may be ambiguous and
asked to enumerate all interpretations (one SQL each); returned SQLs exec-matched to the two golds.

| model | n | both | one | none | mean #interp |
|---|---|---|---|---|---|
| gpt-4o-mini | 1149 | 4% | 35% | 61% | 1.86 |
| gpt-4o (stratified 100/type) | 300 | 6% | 29% | 65% | 1.64 |

By type (gpt-4o): attachment 0% both, scope 0% both, vague 19% both.

**Reading:** explicit elicitation barely helps (sampling 1% → mini 4% → gpt-4o 6%) and *raises*
"none." The model **does** enumerate (~1.6–1.9 interpretations) but the per-interpretation SQL
rarely exec-matches a gold. The bottleneck is **interpretation generation / SQL correctness**, not
willingness to enumerate.

**Important measurement caveat.** Coverage-both requires each returned SQL to **exactly**
exec-match a gold result set, which conflates "didn't find the reading" with "found it but wrote
slightly-off SQL." Our 6% is therefore a **strict lower bound**, and it is well below AMBROSIA's
published model numbers (their headline figures are in the ~30–65% range). The true surfacing rate
is somewhere between our strict 6% and their ~30–60%; we have **not** yet reconciled this against
AMBROSIA's own evaluation code.

---

## 4. Synthesis — dead / alive / contested

- **Dead (our data):** BNP graph posterior as a *correctness* signal (probe 1: 0.58–0.65 alone, no
  marginal value over the verifier).
- **Dead (cheap version):** detecting ambiguity from the generator's own posterior — sampling
  (coverage-both 1%, divergence AUROC 0.475) and explicit elicitation (4–6%) both fail.
- **Alive, empirically grounded:** open-world novelty via Pitman–Yor discovery (probe 2: d=0.16,
  ~52% skeleton discovery), with a real-workload caveat. **Literature gap confirmed absent.**
- **Real but contested / hard:** surfacing multiple valid interpretations (probe 4) — this is the
  unsolved bottleneck owned by the AMBROSIA / EIG line, and any decision/clarification layer must
  sit on top of it.
- **Standalone keeper:** 38% of BIRD questions vary in SQL string but not in result (probe 3) — a
  clean critique of string self-consistency; execution clustering collapses harmless variation.

**One-line verdict:** the generator's distribution is blind to *both* correctness and ambiguity;
both require an external reasoning/structured layer that *proposes* what the sampler will not. BNP
is load-bearing for **open-world novelty**, not for correctness, and the ambiguity/decision paper
depends on a hard, actively-researched interpretation-surfacing step.

---

## 5. Strategic options

1. **Pivot the BNP contribution to open-world novel-intent / OOD detection** (probe 2's
   territory). Clean white space, nonparametrics load-bearing (discovery probability), does **not**
   depend on the ambiguity-surfacing bottleneck. Most defensible BNP paper.
2. **Pursue the ambiguity/decision paper** — but first do the free reconciliation (read AMBROSIA's
   eval code, re-score with their metric) to learn whether real surfacing is 6% or ~50%. Only
   commit if high; differentiate from EIG via the BNP prior + unified execute/clarify/abstain
   objective + calibrated materiality.
3. **Stop and bank** — Paper 1 stands; keep the negatives as "we checked: BNP does not carry
   correctness and cannot cheaply surface ambiguity."

**Recommendation:** do the free reconciliation first (don't conclude from a possibly-unfair strict
6%), then lean toward option 1 for the actual paper.

---

## 6. Reproducibility

| script | what it does | API | cost |
|---|---|---|---|
| `scripts/bnp_probes.py` | probe 1 (marginal value) + probe 2 (motif tail / PYP) | none | $0 |
| `scripts/ambiguity_probe.py` | probe 3 (material divergence on BIRD) | none | $0 |
| `scripts/ambrosia_probe.py` | probe 4a (gold-interpretation materiality/localization) | none | $0 |
| `scripts/ambrosia_generate.py` | probe 4b generation (gpt-4o-mini, K=8 via n=8) | OpenAI | $0.66 |
| `scripts/ambrosia_coverage.py` | probe 4b analysis (sampling coverage + divergence) | none | $0 |
| `scripts/ambrosia_elicit.py` | probe 4c (explicit elicitation; `--model`, `--per-type`) | OpenAI | $1.00 |

**Data.** AMBROSIA is re-downloadable from the Edinburgh share linked at
ambrosia-benchmark.github.io (password `AM8R0S1A`); extracts to `data/ambrosia/` (gitignored:
16 MB CSV + 1,064 SQLite DBs). Sample caches `data/ambrosia_samples.json`,
`data/ambrosia_elicit_*.json` are gitignored (regenerable). All execution uses a 4 s per-query
wall-clock timeout via `conn.interrupt()` (some generated SQL is pathological).

Every API script is safe-by-default: a dry run prints a cost estimate and refuses to call without
`--run`, with a `--max-calls` guard.

---

## 7. Open items / caveats

- **Reconcile probe 4c against AMBROSIA's own eval metric** (free, no API) before any final verdict
  on the ambiguity direction — the 6%-vs-~50% gap is too large to conclude from.
- Probe 2's open-world tail is benchmark-inflated; validate on a real (repetitive) query workload
  before claiming the 52% discovery number holds in production.
- Probe 1's cross-fit combiner is weaker than Paper 1's (it scores verifier+string-SC at 0.715 vs
  Paper 1's 0.754); the "no marginal value" conclusion is robust to this, but absolute combined
  numbers are not directly comparable to Paper 1.
- Several 2026 ambiguity preprints (SOMA-SQL, CLARITY) had unverified author lists at survey time —
  verify before citing.

---

## 8. Follow-up (post-review): is the ambiguity "dead" verdict an artifact? (Exp 1, 2, 2b)

A reviewer note argued we *posteriorized the wrong object* — probes 4b/4c scored ambiguity at the
SQL layer, conflating "didn't find the reading" with "wrote slightly-off SQL." We tested this.
Additional spend here: Exp1 $0 (re-score), Exp2 $0.61, Exp2b $0.80.

### Exp 1 — re-score 4c with AMBROSIA's OFFICIAL metric (`scripts/ambrosia_rescore.py`, no API)
Their metric (`src/evaluation/metrics.py`) uses a **cell-multiset** comparison and reports
**recall** (fraction of gold interpretations matched) and **all_found** (all matched = our
"coverage-both"). Re-scoring our cached zero-shot elicitation:

| model | SQL recall | SQL all_found |
|---|---|---|
| gpt-4o | 0.23 | 1% |
| gpt-4o-mini | 0.22 | 1% |

The metric was **not** the artifact (all_found came in at 1%, not higher). The gap to AMBROSIA's
published ~30–65% is the **prompt** (they use few-shot + "no extra columns") and the
discovery-vs-realization split.

### Exp 2 — interpretation-FIRST elicitation (`scripts/ambrosia_interp.py`, $0.61)
Same model (gpt-4o), same 300 stratified Qs, same "may be ambiguous" framing; only the output
modality changes (English, not SQL). A gpt-4o-mini judge checks each of AMBROSIA's two gold NL
interpretations against the candidate list.

| type | n | NL recall | NL both | (SQL recall / both) |
|---|---|---|---|---|
| attachment | 100 | 0.90 | 80% | 0.07 / 0% |
| vague | 100 | 0.82 | 72% | 0.26 / 2% |
| scope | 100 | 0.54 | 7% | 0.36 / 0% |
| **all** | 300 | **0.75** | **53%** | **0.23 / 1%** |

**Judge caveat (important).** A spot-check caught the gpt-4o-mini judge **over-crediting** subtle
*attachment* distinctions (a single vaguely-worded candidate scored `[True, True]` against two
genuinely different golds). So **53% / attachment 80% are inflated**; vague 72% is supported by the
spot-check; scope 7% is a trustworthy low. The robust, defensible claim is **directional**:
discovery is far easier than realization (vague alone 2% → 72%).

### Exp 2b — two-stage discover→realize, execution-grounded (`scripts/ambrosia_realize.py`, $0.80)
Feed each discovered NL interpretation back, generate one SQL per interpretation ("no extra
columns"), score with the **official** metric (no soft judge).

| type | n | recall | all_found | (one-shot) |
|---|---|---|---|---|
| attachment | 100 | 0.16 | 0% | 0.07 / 0% |
| scope | 100 | 0.38 | 7% | 0.36 / 0% |
| vague | 98 | 0.34 | 14% | 0.26 / 2% |
| **all** | 298 | **0.29** | **7%** | **0.23 / 1%** |

Interpretation-conditioning lifts all_found **1% → 7%** (7×), recall 0.23 → 0.29 — real but modest.
Even handed the exact reading, exec-exact SQL succeeds ~7%; the residual is **output-exactness**
(columns/shape under strict cell-match), the part few-shot + execution-guided repair address.

### Revised verdict (supersedes §4's "ambiguity dead" line)
- **Discovery is tractable** and was NOT the bottleneck — the earlier "ambiguity undetectable"
  reading was a measurement artifact of scoring at the SQL layer.
- **Realization is the dominant bottleneck**; interpretation-conditioning helps 7× but absolute
  stays low. Closing it is known engineering (few-shot, execution-guided repair) — what AMBROSIA's
  own pipeline does to reach ~30–65%.
- The ambiguity/decision paper is **viable but must build on an AMBROSIA-style realization stack**;
  the novel BNP/decision layer (interpretation posterior + unified execute/clarify/abstain +
  calibrated materiality) sits above it.
- **Open-world novelty (probe 2) remains the lowest-dependency BNP contribution.**
- Still pending (free): Exp 5 decision-simulation on gold interpretations — demonstrates the
  decision-theoretic payoff independent of the realization bottleneck.

---

## 9. Experiment 5 — decision simulation (execute / clarify / abstain)

`scripts/ambrosia_decision_sim.py` (no API). AMBROSIA gold interpretations as oracle posterior
support; user's intended reading uniform over the K valid readings; wrong answer costs 1, abstain
0.5, clarify costs c then must *realize* the revealed reading (success prob r). All policies share
the same realization ability r and differ only in the chosen action — isolating the value of the
*decision*. r is a global knob (the SQL generator's realization quality). (Note: a first pass had a
bug — sampled coverage used as realization ability, which made the "oracle" worse than execute; fixed
with the explicit r knob.)

**r = 1.0 (perfect realization — value-of-discovery ceiling)**

| policy | loss c=0.1 | loss c=0.3 | loss c=0.5 | clarify amb/ctl |
|---|---|---|---|---|
| always-execute | 0.296 | 0.296 | 0.296 | 0 / 0 |
| always-clarify | 0.100 | 0.300 | 0.500 | 100 / 100 |
| **Bayes-oracle** | **0.053** | **0.160** | **0.267** | 100 / 0 |
| Bayes-realistic | 0.257 | 0.293 | 0.296 | ~0 / ~0 |

**r = 0.3 (today's realization, Exp 2b)**: all policies ≈ 0.79; oracle abstains (0.50) — clarify
can't be realized 70% of the time. Mean sampled modal confidence is **0.95 on ambiguous questions**
(the collapse).

**Reading.** (1) The decision layer has large value if discovery works: oracle cuts loss 0.296 →
0.053 at c=0.1 (~5×), beating both baselines (clarify ambiguous, execute controls). (2) Today's
collapsed posterior captures almost none of it (Bayes-realistic ≈ always-execute) — `oracle −
realistic` = the value of **discovery**. (3) Realization gates the payoff: at r=0.3 even the oracle
stops clarifying — `r=1 vs r=0.3` = the value of fixing **realization**.

## 10. Overall verdict after the follow-ups

The arc supports a coherent Paper 2 thesis: **decision-aware text-to-SQL under ambiguity.**
- Discovery is tractable (Exp 2); realization is the binding constraint (Exp 2b, ~0.3) but is
  known-engineering (few-shot + execution-guided repair, as AMBROSIA reaches ~30–65%).
- A Bayes execute/clarify/abstain layer over an interpretation posterior captures large value
  (Exp 5) that the collapsed generator posterior misses — and this unified objective is the
  confirmed white space (EIG clarify-only, TrustSQL abstain-only).
- BNP's load-bearing role is the **open-world interpretation/motif prior** (probe 2), not correctness.

Recommended next decision: either (a) replicate AMBROSIA's few-shot realization to lift r toward
~0.5–0.65 and instantiate the full discover→realize→decide pipeline, or (b) pursue the
lower-dependency open-world-novelty BNP paper. Total exploration spend ≈ $4.2.

---

## 11. Equivalence-class gate — does the open-world tail survive quotienting? (`scripts/bnp_equivclass.py`, no API)

The whole BNP-novelty story rests on the motif tail being real, not a syntactic/granularity artifact.
We re-fit the PYP on a canonicalization ladder and added a power-law diagnostic (species-accumulation
exponent: PYP => K_n ~ n^d power law; closed/DP world => saturates).

| level | K | singletons (of queries) | top-1 | d | θ | discovery | accum b | R² |
|---|---|---|---|---|---|---|---|---|
| skeleton (probe-2) | 909 | 49% | 0.7% | 0.160 | 1110 | 0.523 | 0.887 | 0.998 |
| **canon (semantic-ish)** | 332 | 14% | 12.3% | **0.490** | 30.3 | **0.146** | 0.676 | 0.997 |
| clause-set (closed ref.) | 46 | 1% | 21.9% | 0.000 | 9.1 | 0.007 | 0.326 | 0.989 |

Denotational inflation (BIRD model samples): 1.29 distinct skeletons / distinct result-set.

**Verdict: passes, with deflation.** The skeleton fit was degenerate (θ=1110≫N, accumulation b=0.887
≈ linear — the "nearly all unique" curation/granularity signature; probe 2's 52% was the
artifact-prone read). The **canon** level is the credible one and does *not* collapse to clause-set:
a genuine power law (d=0.49, real head top-1 12.3%, accumulation b=0.676, R²=0.997), with **~15%**
semantic-motif discovery — deflated 3–4× from 52% but robustly nonzero and load-bearing. *Caveat:*
quotienting removes syntactic inflation, not **curation** inflation; ~15% remains an upper bound vs a
repetitive production workload — the real-workload check is the next gate and we lack the data.

---

## 12. UQ-coverage experiment — Option C's detector fails (`scripts/ambrosia_uq_coverage.py`)

Make-or-break for Option C: does a PYP-reserve / elicited-interpretation-count score detect ambiguity
(to gate execute vs clarify) better than the failed sample baselines? 300 ambiguous + 300 control.

| score | AUROC(is_ambiguous) | cov@risk 0.1 / 0.2 / 0.3 |
|---|---|---|
| divergence | 0.407 | 2% / 8% / 33% |
| uncertainty | 0.411 | 2% / 8% / 33% |
| K_elicited | 0.557 | 0% / 6% / 22% |
| pyp_reserve | 0.443 | 0% / 0% / 0% |

All near/below chance. **Mechanism:** over-elicitation — P(K≥2 interpretations) is 70% on *controls*
vs 77% on ambiguous. The model has good ambiguity recall but poor precision; the count can't separate
the classes. **Verdict (pre-registered): Option C's cheap detector fails → collapse to Option B.** A
materiality-filtered detector might survive but inherits the realization gate (not cheap).

### Final standing of the options
- **Option B (open-world novelty via equivalence-class PYP): the validated, disciplined choice.** The
  motif tail survives quotienting as a real power law (canon d=0.49, §11). Remaining gate: real-workload
  (curation inflation).
- **Option C (interpretation posterior + decision): cheap version dead.** Discovery has recall but not
  precision; reserve score doesn't gate. Revivable only with a realization stack + materiality filter.
- **Option A (decision layer): valid applied work**, but depends on a detector/realization that we do
  not currently have. Exp 5 stands as a "what it would enable" ceiling result.
- **Total exploration spend ≈ $4.8.**

---

## 13. Option B downstream gate — does motif novelty predict error? (`scripts/bnp_novelty_value.py`)

Detecting novel motifs (§11) is only useful if novel queries are harder. Tested on BIRD with the
canon motif of the PREDICTED modal query (inference-realistic).

| predicted-motif frequency | n | accuracy |
|---|---|---|
| singleton (count=1) | 240 | 0.329 |
| mid (2–4) | 172 | 0.459 |
| common (≥5) | 346 | 0.587 |

Error-prediction AUROC: **novelty 0.628**, verifier (1−v4o) 0.755; verifier+novelty 0.715 vs verifier
alone 0.701 (+0.014). Risk-coverage @25% coverage: novelty 0.598, verifier 0.767, random 0.519.

**Verdict: passes, with caveats.** Novel-motif queries are ~26pp less accurate; novelty is a real,
**cheap (no LLM/execution)** abstention/triage signal — but weaker than and largely subsumed by the
verifier. Open question: is it just complexity? (rare=complex=hard). Need novelty-beyond-complexity
control. Edge: it flags *unfamiliarity*/OOD, which the correctness verifier does not.

### Option B now clears its gates on benchmark data
Tail is a real power law (§11, d=0.49) AND novelty has downstream value (§13). Coherent methods paper:
*a cheap, calibrated, execution-free open-world novelty signal for text-to-SQL (hierarchical
equivalence-class Pitman–Yor) for triage/abstention, complementary to a verifier.* Remaining gates:
(1) novelty-beyond-complexity control (free); (2) real workload (curation inflation).
