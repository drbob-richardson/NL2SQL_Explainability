# Paper 2 — Options, Opportunities, and Big Swings

**Purpose.** A decision document for the second paper's direction. Companion to the empirical
record in [paper2_bnp_decision_exploration.md](paper2_bnp_decision_exploration.md) (probe-by-probe
numbers, scripts, costs). This one is forward-looking: it lays out the candidate directions, what we
have already established toward each, their limitations and dead ends, and the high-ceiling
theoretical bets — with an explicit read on where a *methodological/theoretical* contribution (not
just an applied-ML result) actually lives.

**Audience note.** Written for a Bayesian-nonparametrics statistician moving into AI/actuarial
research. The ranking weights genuine statistical novelty (new prior/estimator/guarantee) over
engineering wins. Total exploration spend to date ≈ $4.2 (cached, reproducible).

---

## 0. Executive summary

- **Established:** Paper 1 (correctness selective-prediction) stands. For Paper 2 we probed whether
  BNP/decision-theory has a load-bearing role. Result: BNP-for-correctness is **dead**; ambiguity
  **discovery** is tractable; **realization** (exact-output SQL) is the binding constraint; a Bayes
  **execute/clarify/abstain** layer has large value *if* discovery works; and the LLM's own
  distribution is a **mode-collapsed proposal**, not a posterior.
- **The fork:** three coherent options (A applied decision layer, B open-world BNP novelty, C the
  synthesis: LLM-as-proposal → BNP posterior over latent interpretations).
- **Where the depth is:** Option **C** carries the only genuine theoretical contribution; A and B
  become its supporting acts. C is also the riskiest. B is the safe middle. A is the most
  conventional ML paper.
- **Cheapest de-risking step:** equivalence-class re-fit of the PYP (the whole BNP story rests on the
  motif tail being real, not a syntactic artifact — and our own data says it is *partly* artifact).

---

## 1. What we have established (compressed)

Full numbers and methods in the companion doc. The load-bearing facts:

| finding | evidence | status |
|---|---|---|
| Black-box agreement signals plateau; reasoning verifier wins (0.77), ensemble 0.82 | Paper 1 | solid |
| BNP graph-posterior features add nothing to the verifier for correctness | Probe 1: feats 0.58–0.65, combined Δ −0.028 [−0.054,−0.004] | **dead end** |
| Gold-query motif tail is open-world at skeleton level | Probe 2: PYP d=0.16, ~52% discovery, 69% singleton motifs | alive (caveated) |
| BIRD splits are mostly error, not ambiguity; 38% of variation is immaterial | Probe 3: gold-among-candidates 37%; 38% same-result-different-string | diagnostic |
| AMBROSIA interpretations are 100% material; localization type-dependent | Probe 4a: vague 83% single-slot, scope/attachment 0% | benchmark validated |
| Sampling does not surface ambiguity; divergence ≠ ambiguity detector | Probe 4b: coverage-both 1%, AUROC 0.475, modal conf 0.95 | **dead end (cheap version)** |
| Discovery ≫ realization: model names readings it can't write | Exp 2: NL recall 0.75 vs SQL 0.23 (judge-inflated; vague 72% solid) | key pivot |
| Two-stage discover→realize lifts SQL all_found 1%→7% | Exp 2b (official metric) | realization is the gate |
| Bayes decision layer ~5× value at the ceiling; collapsed posterior misses it | Exp 5: oracle 0.296→0.053 (c=0.1, r=1); realistic≈execute | payoff quantified |

**Literature anchors (verified):** AmbiQT (arXiv:2310.13659), AMBROSIA (arXiv:2406.19073, the
benchmark we use), EIG/Qiu (arXiv:2507.06467, nearest competitor, clarify-only), TrustSQL
(arXiv:2403.15879, abstain-only), SOMA-SQL (arXiv:2606.11424, very recent, resolution not UQ),
Stengel-Eskin et al. (arXiv:2306.00824, distribution over interpretations but FOL not SQL). BNP
open-world priors for ambiguity/novel-intent in text-to-SQL: **confirmed absent**.

---

## 2. Option A — Decision-aware text-to-SQL (execute / clarify / abstain)

**Idea.** A unified Bayes decision rule over actions {execute q, ask a targeted clarification,
abstain}, minimizing expected materiality-weighted loss given a posterior over interpretations.

**What we've done.** Exp 5 simulates it on AMBROSIA gold interpretations: at perfect realization the
oracle policy cuts loss 0.296→0.053 (c=0.1), beating always-execute and always-clarify; the collapsed
sampled posterior captures ~none of it. The white space is confirmed (EIG is clarify-only, TrustSQL
abstain-only; never unified).

**Methodological/theoretical potential: LOW–MODERATE.** The decision machinery is classical
(Raiffa–Schlaifer, Lindley value-of-information). Unifying clarify+abstain+execute under one
materiality loss is a real *applied* contribution but not a new method or theorem.

**Limitations.** Rides on upstream discovery+realization, which are the hard parts and not statistics.
Materiality is 100% on AMBROSIA (so the materiality-weighting only bites on production/BIRD-like data).

**Dead ends already ruled out.** Triggering clarification from sampling divergence (AUROC 0.475) or
from the generator's own confidence (modal 0.95 on ambiguous) — both blind to ambiguity.

**Opportunity.** A clean risk–coverage-style *clarification-efficiency frontier* (loss vs
clarification budget), with the decision layer dominating both baselines — a compelling figure.

**Big swing.** Targeted, *block-level* clarification (scope/attachment are coupled transformations,
not single slots — Probe 4a) chosen to maximize value-of-information; a principled clarification
policy that asks the single highest-EVI question. Still applied, but novel as a policy.

**Minimum publishable unit / risk.** ACL/EMNLP-Findings-tier applied paper. Risk: novelty overlaps
EIG; differentiation rests on the abstention arm + materiality + calibration.

---

## 3. Option B — Open-world novelty / OOD via a BNP prior over query motifs

**Idea.** Place a (hierarchical) Pitman–Yor prior over an open space of query motifs and use the
posterior-predictive *discovery probability* (θ+dK)/(θ+N) as a calibrated novelty/abstention signal:
"is this query structurally familiar enough to trust the usual machinery?"

**What we've done.** Probe 2: skeleton-level motif tail is heavy (d=0.16, ~52% next-query discovery,
69% singleton motifs); clause-flag level is closed. Literature gap confirmed absent.

**Methodological/theoretical potential: MODERATE.** A genuine BNP modeling contribution, but the core
machinery (species sampling, discovery probabilities — Pitman; Favaro–Lijoi–Prünster) is established.
It becomes more than an application only via the wrinkles below.

**Limitations.** (1) The 52% discovery is **benchmark-inflated** — benchmarks are curated for
diversity; real workloads repeat. (2) Skeleton equivalence is too granular: 38% of SQL variation is
immaterial (Probe 3), so part of the "novelty" is syntactic noise.

**Dead ends.** Raw-string or clause-flag motifs (string overstates novelty; clause-flag space is
closed at K=44).

**Opportunity (this is where it gets interesting).**
- **Equivalence-class PYP:** fit the prior over *denotational-equivalence classes* (semantic
  equivalence on the database), not raw skeletons. If the power-law tail survives quotienting, the
  novelty result is real.
- **Hierarchical/temporal PYP:** P_db ~ PYP(·, P_0), P_0 ~ PYP(·, H) across database/domain/workload;
  novelty becomes *relative* (global vs domain vs db vs genuinely new). Validate temporally: fit on
  early logs, predict discovery on later logs; compare PYP vs DP vs Good–Turing.

**Big swing.** Species sampling over a **quotient space with a noisy/approximate equivalence oracle**
(SQL semantic equivalence is undecidable; we approximate via execution on sampled DBs). Characterize
how approximate class-merging perturbs the discovery probability and posterior — genuine BNP theory
texture.

**Minimum publishable unit / risk.** A focused BNP-methods paper (stats or ML-methods venue). Risk:
needs a real workload to escape the benchmark-inflation critique; without one it reads as "applied
PYP on a curated set."

---

## 4. Option C — The synthesis: LLM-as-proposal → BNP posterior over latent interpretations

**Idea (the deep one).** Stop treating the LLM sampling distribution as a posterior. It is a
**mode-collapsed, biased proposal** over surface strings (Probe 4b: covers both readings ~1%,
confidence 0.95). The object of inference is a posterior over **latent interpretations** I (modulo
SQL equivalence), with the LLM as one noisy proposal/evidence source and a **BNP prior supplying
probability mass to interpretations the LLM never samples**.

Formally: π(I | x, S, E) ∝ π_BNP(I | S) · exp(−η · loss(I; evidence)), a generalized-Bayes update
(Bissiri–Holmes–Walker 2016) over an open, equivalence-class–structured interpretation space, with
π_BNP a (hierarchical) Pitman–Yor whose discovery parameter d>0 guarantees non-vanishing mass on
un-proposed atoms. The decision layer (Option A) acts on this posterior; the open-world prior
(Option B) *is* the mass-on-unobserved mechanism.

**What we've done toward it.** Every probe is a load-bearing piece: Probe 2 = the prior's tail;
Probe 4b = the proposal's collapse (the problem this solves); Exp 2/2b = discovery feasible,
realization the gate; Exp 5 = the decision payoff. The pieces already compose into this frame.

**Methodological/theoretical potential: HIGHEST.** This is the only path where the BNP does work an
ML researcher couldn't easily do. Three theorem-shaped targets, increasing ambition:

1. **Calibrated posterior mass on unsampled interpretations.** Under a PYP prior + LLM-proposal
   likelihood, bound the posterior probability that the *true* interpretation lies in the credible
   set *even when the LLM never proposed it*. A novel UQ guarantee against a mode-collapsed proposal —
   directly motivated by the 1%-coverage failure. **This is the cleanest single nugget.**
2. **Generalized-Bayes learning-rate calibration.** No tractable generative model for NL exists, so
   ordinary Bayes is out; calibrate η in the Gibbs posterior and give decision-theoretic guarantees.
   Connects BNP + generalized Bayes + decision theory into one object.
3. **Conformal validity over a growing label space.** Prediction sets over interpretations with
   finite-sample coverage when the label set itself grows (PYP). Connects to conformal-under-shift
   (Tibshirani et al. 2019); the open/growing-label case is genuinely under-developed.

Plus the equivalence-class / noisy-oracle texture from Option B.

**Limitations / honest risks.**
- **Empirically capped by realization.** Exp 5 at r=0.3: even an oracle posterior can't act. A
  beautiful posterior that doesn't move an end metric is a hard sell. Mitigation: anchor on a
  **UQ-side metric** (calibrated coverage of the true interpretation set) that does *not* depend on
  realization, so the theory stands on its own.
- **"LLM as proposal" is not brand new** (SMC steering of LLMs, Lew et al. 2023). The novelty is the
  **BNP open-world target + UQ guarantee on unsampled mass**, not the proposal idea per se.
- The theory must be *developed*, not gestured at — this is a months-not-weeks bet.

**Big swing.** A full **importance-reweighting / SMC scheme** that turns the collapsed LLM proposal
into a calibrated posterior over interpretations, with the PYP supplying the tail, and a proven
coverage guarantee for the true interpretation. If it works, it's a general recipe for UQ over
structured LLM outputs (code, programs, plans) — far beyond SQL.

**Minimum publishable unit / risk.** A methods/theory paper (NeurIPS/ICML/AISTATS or a stats
journal). Highest ceiling, highest risk; the theorem-#1 + a UQ-coverage experiment is the
defensible MPU even if the full SMC swing doesn't land.

---

## 5. Option D — Safe fallbacks (bankable now)

- **Equivalence-aware self-consistency.** Probe 3: 38% of questions have multiple SQL strings with
  identical results; string self-consistency penalizes harmless variation, execution clustering
  fixes it. A short, clean critique + fix — low risk, modest contribution, immediately writable.
- **Negative-results note.** "BNP does not carry correctness; ambiguity is invisible to the
  generator's posterior" — honest, citable, but thin alone.

These are insurance, not the main bet.

---

## 6. Cross-cutting limitations & risks

- **Realization is the universal gate.** Every ambiguity-side payoff is capped until exact-output SQL
  improves (r≈0.3 today). It's known-engineering (few-shot + execution-guided repair → AMBROSIA's
  ~30–65%), but it must be built or borrowed.
- **Benchmark inflation.** Motif novelty (Probe 2) and AMBROSIA materiality (100%) are curated;
  real-workload validation is the credibility lever.
- **Judge softness.** Exp 2's NL recall used an LLM judge that over-credited subtle distinctions;
  any discovery claim needs execution grounding or a stricter (gpt-4o / human-spot-checked) judge.
- **Active, crowded neighborhood.** AMBROSIA, EIG, SOMA-SQL (10 days old) are all circling
  ambiguity/disambiguation. Differentiation must be the *statistical* contribution, not the task.

---

## 7. Dead ends (do not revisit)

1. BNP graph-posterior as a **correctness** signal (Probe 1).
2. Detecting ambiguity from **sampling divergence / generator confidence** (Probe 4b: AUROC 0.475).
3. Scoring discovery at the **SQL layer** with strict exec-match (conflates discovery with
   realization — the artifact that produced the false "dead" verdict).
4. **Raw-skeleton / clause-flag** motif granularity for the open-world prior (too granular / too
   coarse).

---

## 8. Recommendation

For a **methodological/theoretical** contribution: pursue **Option C as the spine**, with
**contribution #1 (calibrated posterior mass on unsampled interpretations)** as the core theorem,
**Option B (open-world PYP)** as its prior, and **Option A (decision layer)** as the "what it
enables" payoff. This makes the BNP do real work, folds the other options in as supporting acts
rather than competing papers, and is defensible on a UQ metric that doesn't hostage the result to
realization engineering.

If appetite for risk is lower, **Option B + the equivalence-class/hierarchical wrinkles** is the safe
middle with genuine (if smaller) BNP content. **Option A alone** is the most conventional and the
least "yours."

---

## 9. Concrete next steps (cheap → expensive)

1. **Equivalence-class PYP re-fit** (free, ~1 hr). Quotient skeletons by denotational equivalence on
   the DB; re-estimate d, θ, discovery. *Gates the entire BNP story.* Do this first.
2. **Stricter / execution-grounded discovery metric** (≈$0.3). Re-judge Exp 2 with gpt-4o + a
   distinguishing-feature rubric, to get a discovery number we can stand behind.
3. **Theorem #1 sketch** (no compute). Write the generative model (PYP prior + proposal likelihood)
   and state the unsampled-mass coverage claim; decide if it's provable as stated.
4. **UQ-coverage experiment** (free–cheap). On AMBROSIA, measure whether a PYP-augmented posterior
   covers the true interpretation set better than the raw LLM proposal — the realization-independent
   metric that lets Option C stand alone.
5. **Realization lift** (≈$1–2). Replicate AMBROSIA few-shot + repair to push r toward ~0.5–0.65;
   re-run Exp 5 at realistic r to show the decision frontier with an achievable realizer.
6. **Hierarchical/temporal PYP on a real workload** (needs data). The credibility capstone for B/C.

---

## 10. Open questions to resolve before committing

- ~~Does the motif power-law tail survive denotational quotienting?~~ **ANSWERED (Exp, §11 of findings):** yes — at the semantic 'canon' level it is a clean power law (d=0.49, accumulation b=0.68, R²=1.0) with ~15% discovery (deflated from the inflated skeleton-level 52%). Syntactic inflation removed; **curation inflation (real-workload check) is now the open co-gate.**
- ~~Is theorem #1 provable?~~ **RESOLVED:** robust (conformal-wrapped) form is provable but borrows LTT validity; the novel part was the reserve-SCORE *efficiency*, which **failed empirically** (UQ-coverage AUROC 0.557 ≈ chance; over-elicitation kills precision). **Option C cheap version dead → collapse to Option B.**
- ~~Does novelty predict error (abstention value)?~~ **ANSWERED (findings §13):** yes — singleton-motif acc 0.33 vs common 0.59, error-AUROC 0.628; cheap/execution-free but verifier-dominated. New open gate: **is novelty just complexity?** (free control needed).
- Can we get *any* repetitive real-world SQL workload? (Decides whether the open-world story is
  benchmark-bound or production-credible.)
- Is the target venue statistics (theory-forward) or ML (method+benchmark)? This changes how much of
  C vs A/B to foreground.
