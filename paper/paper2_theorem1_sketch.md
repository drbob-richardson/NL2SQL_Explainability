# Theorem 1 sketch — calibrated reserve mass for un-elicited interpretations

**Goal.** Make precise the central theoretical claim of Option C ("the BNP prior supplies mass to
interpretations the LLM never proposed") and stress-test whether it is a *theorem* or *rhetoric*.

**Verdict up front (so the rest is read with the right expectation).** The pure-Bayesian form
("PYP discovery probability = probability the true interpretation was missed") is a genuine theorem
*under an exchangeability assumption that our own data shows is false* (the LLM's misses are
systematic, not fresh draws). The **defensible** form demotes the PYP reserve from a *probability*
to a *score*, and borrows finite-sample validity from conformal / Learn-then-Test (the same
machinery as Paper 1). In that form Theorem 1 is real but modest: **validity is inherited; the novel,
falsifiable contribution is that the PYP-reserve score is more *efficient* than divergence/confidence
baselines on an open interpretation space.** That efficiency claim is exactly the cheap UQ-coverage
experiment.

---

## 1. Objects and model

- Question `x`, schema `S`. Latent **intended interpretation** `I* ∈ 𝓘(x,S)`, an open/countable space.
- An **elicitation set** `Ĩ = {I_1,…,I_m}` of candidate interpretations from the LLM (Exp 2: NL
  elicitation, per-reading recall ≈ 0.75; sometimes misses `I*`). `K_m = |distinct(Ĩ)|`.
- An **evidence/score** `ℓ(I; E)` per candidate (verifier, execution, schema grounding), giving a
  within-set conditional posterior `p(I | Ĩ, E)` over the elicited candidates.
- A **species-sampling prior**: across a workload, intended interpretations follow a Pitman–Yor
  process `PYP(d, θ)`, `0 ≤ d < 1`, `θ > −d`. (Fitted at the semantic "canon" level: d≈0.49, θ≈30;
  see findings §11.)

The decision: **execute** the posterior-mode interpretation, **clarify**, or **abstain**, controlling
the probability of executing a *wrong* interpretation.

---

## 2. The naive claim and its (real) proof

**Claim (naive).** Treat `I*` as the `(m+1)`-th draw of the species-sampling sequence that produced
the `m` elicited interpretations. Then the probability `I*` is a *novel* species (not in `Ĩ`) is the
Pitman–Yor predictive reserve

```
ρ_m  =  P(I* ∉ Ĩ | Ĩ)  =  (θ + d·K_m) / (θ + m).
```

**Proof.** This is exactly Pitman's prediction rule for the PYP: given an exchangeable species-
sampling sequence with `K` distinct species in `n` draws, `P(draw n+1 is new) = (θ+dK)/(θ+n)`, and
`P(draw n+1 = species j) = (n_j − d)/(θ+n)`. Setting `n=m`, `K=K_m`, and identifying `I*` with draw
`m+1` gives `ρ_m`. ∎

This is **not** the tautology "the prior has positive tail mass." It is a closed-form, *estimable*
quantity that converts "did I miss a reading?" into the discovery probability — and it is
**falsifiable**: the realized miss-rate should match `ρ_m`.

**Decision rule.** Hold back `ρ_m` of the credible mass for "an interpretation I have not seen";
spend the remaining `1−ρ_m` on the elicited candidates via `p(·|Ĩ,E)`. Execute the mode iff the
residual risk of executing fits the budget; else clarify/abstain. Formally, execute iff
`ρ_m + (1 − ρ_m)·(1 − p(mode|Ĩ,E)) ≤ α`.

---

## 3. Why the naive claim fails empirically — the exchangeability break

The proof needs `I*` to be **exchangeable** with the elicited draws under the *same* PYP. It is not:

- The elicited draws come from the LLM proposal `q_LLM`, a **biased, mode-collapsed** distribution
  (Exp 4b: covers both readings 1%; modal confidence 0.95), **not** the PYP and **not** `q(I*)`.
- The LLM's misses are **systematic, not random**: it reliably fails the *same* reading types
  (Exp 2: attachment "same-location" readings essentially never produced; scope readings
  rarely). So when `I*` is missed it is disproportionately a *hard-for-the-model* species, not a
  fresh PYP draw.

Consequence: the realized `P(I* ∉ Ĩ)` is **larger** than `ρ_m`. The PYP reserve is therefore a
**lower bound** on the true miss probability, not an equality. Resting Option C on the naive
equality would be exactly the "vague tail-mass argument" we wanted to avoid.

(Two further gaps: (b) within-set calibration `p(·|Ĩ,E)` is only as good as the verifier — Paper 1:
0.77–0.82, decent not perfect; (c) `d, θ` are estimated, workload-level, and per-question `m≈2` is
tiny, so `ρ_m` is high-variance.)

---

## 4. The robust version — conformal / LTT with a Pitman–Yor reserve *score*

Demote `ρ_m` from a probability to a **gating score** `s(x) = ρ_m(x)` (high ⇒ likely to miss), and
recover validity by calibration, not by the PYP being correct.

**Setup (split conformal / Learn-then-Test, as in Paper 1 §4.5).** Hold out a calibration set of
questions `{x_i}` (exchangeable with test), each with known `I*_i`, elicited set, and within-set
selection. Define the **executed-error event** `M_i(τ) = 1[ s(x_i) ≤ τ and selected(x_i) ≠ I*_i ]`
(we executed and were wrong). Choose the largest threshold `τ̂` whose calibration risk passes the
LTT/Bonferroni-over-grid test at level `(α, δ)`.

**Theorem 1 (robust form).** The rule *"execute the mode when `s(x) ≤ τ̂`, else clarify/abstain"*
satisfies, with probability ≥ `1−δ` over the calibration draw,

```
P_test( executed AND wrong )  ≤  α,
```

distribution-free and finite-sample — requiring **only exchangeability of questions** (calibration ↔
test), **not** exchangeability of `I*` with the elicited draws, **not** correctness of the PYP.

**Proof.** Immediate from Learn-then-Test risk control (Angelopoulos et al. 2021) applied to the
monotone family of selective-risk thresholds indexed by `τ`; identical to the certificate in Paper 1,
with `s = ρ_m` as the score. ∎

So **validity is borrowed and solid.** The PYP is no longer load-bearing for *correctness* of the
guarantee — it is load-bearing for **efficiency**:

**Efficiency claim (the actual research bet, empirical).** At a fixed risk level `α`, gating on the
PYP-reserve score `s = ρ_m` yields **higher coverage** (fewer clarify/abstain actions) than gating on
- sample divergence (Exp 4b: AUROC 0.475 — useless),
- max within-set probability / generator confidence (modal 0.95 — overconfident),
- a fixed "always clarify if ≥2 candidates" rule.

This is *not* a theorem; it is the falsifiable claim that makes the BNP worth including, and it is
precisely the UQ-coverage experiment (ladder step 2).

---

## 5. Hierarchical refinement (needed for the score to transfer)

`ρ_m` uses workload-level `d, θ`, but novelty is relative to *which* workload. Use a hierarchical PYP

```
G_w ~ PYP(d_w, θ_w, G_0),   G_0 ~ PYP(d_0, θ_0, H),
```

indexing `w` by database/domain. Then the reserve becomes `ρ_m^{(w)}`, correctly tighter inside a
familiar domain and looser on a novel one. This also gives a principled handle on the **curation-
inflation co-gate**: benchmark `d` is a domain-mixture; a single workload's `d_w` should be smaller.

---

## 6. Proof obligations — what is provable vs empirical

| statement | status |
|---|---|
| Naive: `ρ_m = P(I*∉Ĩ)` under PYP-exchangeability | **theorem** (Pitman's rule) — but assumption empirically false |
| `ρ_m` is a **lower bound** on the true miss-rate under systematic model bias | provable under a stochastic-dominance condition on the proposal; **worth formalizing** |
| Robust: LTT control of executed-error at `α` with score `s=ρ_m` | **theorem** (inherited from LTT; needs only question-exchangeability) |
| Efficiency: PYP-reserve score dominates divergence/confidence baselines | **empirical** (UQ-coverage experiment) — the real contribution |
| Hierarchical-PYP transfer of `d,θ` across workloads | modeling + empirical |

**Honest read on novelty.** The robust theorem reuses Paper 1's conformal machinery, so its *validity*
is not new. The contributions that are genuinely new and defensible: (i) the **PYP-reserve score** for
an *open* interpretation space (a new, principled gating signal where divergence/confidence fail);
(ii) the **lower-bound** result quantifying how model bias makes naive Bayesian reserve optimistic;
(iii) the **hierarchical/relative-novelty** formulation. That is a solid *methods* contribution — not
a deep new theorem, but real, and honest about it.

---

## 7. Verdict and the one experiment that tests it

Theorem 1 is **real in its conformal-wrapped form** and **rhetoric in its naive Bayesian form**. The
right move is to lead with "a conformally-valid clarify/abstain rule whose score is a Pitman–Yor
reserve, provably tighter than confidence/divergence on open interpretation spaces," and to *prove*
the lower-bound result (item ii) for theoretical weight.

Everything now hinges on **one cheap, realization-independent experiment**: on AMBROSIA, does the
PYP-reserve score `ρ_m` control the executed-error rate at a target `α` with **higher coverage** than
the divergence and max-probability baselines? If yes, Option C has an empirical spine and the methods
contribution stands. If the reserve score is no better than baselines, Option C collapses to Option B
(novelty detection) and we should not force the interpretation-posterior framing.

Next step: implement the UQ-coverage experiment — compute `ρ_m` per AMBROSIA question from the
elicited set + fitted (d,θ), run LTT threshold selection on a calibration split, and compare the
risk–coverage frontier of `ρ_m` vs divergence vs max-prob on the test split.

---

## 8. UQ-coverage result — the efficiency claim FAILS (`scripts/ambrosia_uq_coverage.py`)

Pre-registered test (§7): does the PYP-reserve / elicited-count score control executed-error
(executing on an ambiguous question) with higher coverage than divergence/confidence? 300 ambiguous
+ 300 control, target = is_ambiguous.

| score | AUROC(amb) | cov@risk 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|
| divergence | 0.407 | 2% | 8% | 33% |
| uncertainty | 0.411 | 2% | 8% | 33% |
| K_elicited | 0.557 | 0% | 6% | 22% |
| pyp_reserve | 0.443 | 0% | 0% | 0% |

All scores are at/below chance. Sample-based scores are **below 0.5** (ambiguous questions are *more*
confident than controls — the collapse). The best, elicited-count, is **0.557** — not usefully above
chance. **Mechanism:** the model over-elicits — P(K≥2) is 70% on *controls* vs 77% on ambiguous
(mean 1.77 vs 1.94). Good recall, poor precision: it manufactures readings for clear questions, so
the count can't gate.

**Verdict (pre-registered): the cheap efficiency claim fails → Option C collapses to Option B.**
The only version of C that might survive is a **materiality-filtered** detector (realize each elicited
interpretation, execute, count *materially-distinct* result sets — spurious control paraphrases
should collapse to one). But that **inherits the realization gate** (Exp 2b, r≈0.3) and is no longer
cheap or realization-independent. Per the pre-registration we do not pursue it as a rescue without an
explicit decision to pay the realization cost.
