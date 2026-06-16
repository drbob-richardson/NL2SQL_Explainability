# What Predicts Correctness in Text-to-SQL? A Selective-Prediction Study

*Paper 1 draft — 2026-06-15. Scope: predicting whether a generated SQL query is execution-correct,
and using that signal to abstain under a risk target. Schema-linking uncertainty (which elements a
question implies) is a companion paper; here it appears only as one of the signals we test and rule
out as a correctness predictor.*

---

## Abstract

When should we trust an LLM's SQL? We study, on equal footing and with bootstrapped confidence
intervals, which signals predict whether a generated query is *execution-correct* on hard
multi-table text-to-SQL (BIRD). We find a sharp dichotomy. **Black-box statistical signals plateau:**
string, structural, and execution self-consistency, structural priors, and semantic schema-linking
confidence all sit at AUROC ≈ 0.61–0.68 (string self-consistency, 0.675, is the strongest), and
**white-box log-probability does not exceed them** (0.67; no significant gain when combined with the
best baseline). **What breaks the ceiling is verification:** an LLM judge scores 0.72 (GPT-4o-mini)
to 0.78 (Claude), and because independent-provider judges make *different* errors (GPT-4o vs Claude
score correlation 0.43) a two-provider ensemble reaches **AUROC 0.82** with a well-calibrated
correctness probability (ECE 0.03), supporting useful abstention frontiers (e.g., 27% coverage at
24% selective risk) where self-consistency yields no valid low-risk subset — though distribution-free
certificates stay conservative under the regime's low base accuracy. We further ask
whether the verifier can be *trained*: fine-tuned verifiers — encoder and generative alike — reach
in-distribution AUROC ≈ 0.77–0.79 (matching the frozen GPT-4o judge) but **do not transfer**,
dropping to ≈ 0.66 across unseen schemas, even though fine-tuning clearly helps (it lifts a small
generative judge +0.11 over its zero-shot self). Cross-schema transfer tracks **model scale and
reasoning**, not fine-tuning: the frozen large judge's 0.77 is already a transfer number, while a
small fine-tuned model memorizes schema surface form. The practical upshot: correctness UQ for
text-to-SQL is real but lives in *reasoning-based* signals; a fine-tuned verifier is an excellent
**per-deployment** tool, while a **universal** verifier requires a model that generalizes across
schemas — so far, a large frozen reasoning judge.

---

## 1. Introduction

Text-to-SQL systems are deployed where wrong answers are costly, so the decision *"answer or
abstain"* matters as much as raw accuracy. That decision needs a calibrated estimate of whether a
generated query is correct. A large UQ literature exists, but it is dominated by **black-box
statistical** signals — sampling agreement (self-consistency), semantic entropy, structural
frequency — that are attractive because they need no extra model. We ask a simple question
directly: **on hard text-to-SQL, which signals actually predict execution correctness, and by how
much?**

Our contribution is a careful, execution-grounded *map* with confidence intervals, plus the
deployment consequences:

1. **A correctness ceiling for black-box statistical UQ** (AUROC ≈ 0.61–0.68 on BIRD), shared by
   string/structural/execution self-consistency, structural priors, schema-linking confidence, and
   — once measured against the *strongest* baseline — white-box log-probability.
2. **Verification breaks it**: LLM judges clearly exceed the ceiling with bootstrapped CIs excluding
   zero — and the result is robust, replicating across **two generators** and **two judge providers**;
   independent-provider judges make different errors and **ensemble to the best, calibrated
   correctness signal** (AUROC 0.82, ECE 0.03).
3. **A selective-prediction frontier**: the combined score supports empirical abstention frontiers
   the self-consistency baseline cannot.
4. **A trained-verifier study**: fine-tuned verifiers win in-distribution but **fail to transfer**
   across schemas; we separate the *per-deployment* from the *universal* verifier problem.

## 2. Related work

Self-consistency and sampling-based UQ; semantic entropy via meaning clustering (Farquhar et al.,
*Nature* 2024); verbalized confidence; conformal / selective generation; sub-clause Platt scaling
(EMNLP 2025); RTS conformal abstention (SIGMOD 2025). LLM-as-judge is widely used for open-ended
evaluation; we evaluate it specifically as a *calibrated correctness predictor for SQL* against
black-box baselines, and we test whether judging can be *trained* and whether it *transfers*.

## 3. Setup

- **Benchmark and generation.** BIRD dev (11 databases with data dictionaries + evidence). We
  generate SQL with `gpt-4o-mini` (K = 8 samples per question, temperature 0.7, schema + evidence in
  the prompt) for an 800-question slice across 8 databases, and execute every sample against the
  real SQLite databases for ground-truth correctness. Modal-query execution accuracy is 0.451 — a
  genuinely hard regime with ample headroom for UQ.
- **Signals evaluated.**
  - *Black-box statistical:* string self-consistency (largest identical-string cluster), structural
    self-consistency (canonical-structure cluster), execution self-consistency (result-set cluster),
    and schema-linking confidence (companion-paper posteriors over the generated query).
  - *Logic-aware:* mean sequence **log-probability** of the modal query (white-box); an **LLM
    verifier** shown (question, evidence, schema, candidate SQL) — `gpt-4o-mini` and `gpt-4o` (P from
    YES/NO first-token logits), and **Claude-Sonnet-4.6** as an independent-provider judge (a
    verbalized 0–100 probability, since Anthropic exposes no token logprobs).
- **Metrics.** Tie-robust AUROC for correctness; cross-fit logistic combination (parity split) with
  2,000-sample bootstrap CIs on the AUROC delta; ECE; risk–coverage with a distribution-free
  (Bonferroni-over-grid, δ = 0.1) certificate and an empirical frontier.
- **Cost & reproducibility.** All generation/verification is safe-by-default (cost estimate, no
  calls without `--run`, hard caps, caching); total API spend for the study ≈ \$1.6 (including the
  cross-provider Claude judge). Code and cached signals are released.

## 4. Results

### 4.1 The black-box correctness ceiling (and where white-box logprob sits)

| signal (alone) | AUROC for correctness |
|---|---|
| **string self-consistency** (largest identical-string cluster) | **0.675** |
| structural self-consistency (largest canonical-structure cluster) | 0.619 |
| execution self-consistency (largest result-set cluster) | 0.613 |
| log-probability (white-box, mean sequence logprob) | 0.669 |
| schema-linking graph confidence | 0.553 |
| *"does it execute / return rows"* | 0.500 (chance) |

The black-box statistical signals plateau at AUROC ≈ 0.61–0.68; *string* self-consistency is the
strongest (0.675). Crucially, **white-box log-probability does not exceed this ceiling** (0.669):
added to string self-consistency it gives +0.005 (CI [−0.003, +0.012], not significant). Logprob
separated confident-wrong queries on *saturated single-table* data, but on hard multi-table BIRD it
is no better than sampling agreement. Wrong queries also execute and return rows just fine, so
executability is chance.

### 4.2 Verification breaks the ceiling

We combine each signal with the strongest baseline (string self-consistency) via cross-fit logistic;
AUROC alone and bootstrap CI of the combined delta:

| signal | AUROC alone | combined Δ over string self-consistency (95% CI) |
|---|---|---|
| log-probability (white-box) | 0.669 | +0.005 [−0.003, +0.012]  (n.s.) |
| LLM verifier (`gpt-4o-mini`) | 0.724 | — |
| LLM verifier (`gpt-4o`) | 0.770 | +0.079 [+0.054, +0.105] |
| LLM verifier (Claude-Sonnet-4.6) | **0.776** | +0.131 [+0.101, +0.163] |

Only the verifiers break the ceiling, with CIs excluding zero. The signal that works is the one not
blind to query logic: a model that can *reason* about whether the aggregation, filters, and joins
answer the question. Log-probability, by contrast, adds nothing over the strongest sampling
baseline here.

### 4.3 Cross-provider robustness: independent judges make different errors and ensemble

A verifier from the generator's own family could merely echo it, so we test an **independent
provider**. Claude-Sonnet-4.6 (judging GPT-4o-mini's SQL, via a verbalized 0–100 probability since
Anthropic exposes no token logprobs) is the *strongest single judge* (0.776 > GPT-4o's 0.770), and
the two judges' scores correlate only **r = 0.43** — they make genuinely different errors. As a
result a **two-provider ensemble (GPT-4o + Claude) reaches AUROC 0.822** (Δ over GPT-4o alone +0.052,
CI [+0.032, +0.073]), the strongest correctness signal we obtain, and is well-calibrated (ECE 0.031;
§4.5). This both rules out the self-agreement artifact and shows that combining independent reasoning
judges — not a single bigger one — is the most reliable correctness signal. (Caveat: the OpenAI
judges use YES/NO logits and the Anthropic judge a verbalized probability; a method-matched OpenAI
verbal judge also clears the ceiling at 0.709, confirming the gap is not an elicitation artifact.)

**A second generator.** The dichotomy is not unique to gpt-4o-mini. Regenerating the slice with
**gpt-4.1-mini** (modal accuracy 0.522 — a stronger generator) and judging with the same gpt-4o-mini
verifier reproduces it:

| generator | accuracy | string self-consistency | verifier (gpt-4o-mini) | combined Δ over SC (95% CI) |
|---|---|---|---|---|
| gpt-4o-mini | 0.451 | 0.675 | 0.724 | +0.059 [+0.032, +0.087] |
| gpt-4.1-mini | 0.522 | 0.641 | 0.677 | +0.033 [+0.007, +0.061] |

For both generators, self-consistency sits at the ceiling and the verifier exceeds it with a combined
delta whose CI excludes zero. The margin is smaller for gpt-4.1-mini — expected, since here the judge
(gpt-4o-mini) is *weaker* than the generator and so catches fewer of the stronger model's subtler
errors; a judge at least as strong as the generator (cf. the GPT-4o / Claude results above) should
widen it.

### 4.4 Can the verifier be trained? In-distribution yes, transfer no — across architectures and scale

We fine-tune verifiers on (question + evidence + schema + SQL → correct), 6,400 execution-labeled
pairs, evaluated with the same leave-one-database-out (LODO) protocol, spanning two architectures
(an encoder classifier and a generative LLM judge fine-tuned with LoRA) plus an un-fine-tuned
baseline.

| verifier | in-distribution AUROC | leave-one-DB-out (transfer) |
|---|---|---|
| cheap feature classifier (embeddings + logprob + self-consistency) | 0.768 | 0.661 |
| fine-tuned encoder, RoBERTa-base | 0.778 | — |
| fine-tuned encoder, ModernBERT-base (full schema context) | 0.785 | 0.670 |
| generative judge, Qwen2.5-1.5B-Instruct — **zero-shot** | 0.651 | 0.553 |
| generative judge, Qwen2.5-1.5B-Instruct — **LoRA fine-tuned** | 0.766 | 0.659 |
| *frozen `gpt-4o` judge (zero-shot)* | — | *0.770 (already a transfer number)* |

Three findings. **(i) Fine-tuning works** — it lifts the small generative judge +0.115
in-distribution and +0.106 on transfer over its own zero-shot baseline; training is not a no-op.
**(ii) But it does not transfer** — every fine-tuned verifier, encoder or generative, converges to
the same wall: ≈ 0.77 in-distribution, ≈ 0.66 across unseen schemas (a ~0.11 gap), worst on the most
domain-specific held-out databases (e.g., toxicology 0.56–0.64). Long context (ModernBERT) and a
generative architecture (Qwen) both fail to close it; the models *memorize schema surface form*.
**(iii) The frozen large judge is the exception** — GPT-4o's 0.77 *is* its cross-schema number, and
the small base model's near-chance zero-shot transfer (0.553) shows why: transfer here is a function
of **model scale / reasoning**, not of fine-tuning. A small model fine-tuned overfits; a large model
reasons.

Per database (Figure `paper1_lodo_perdb.png`), the frozen GPT-4o judge leads on *every* held-out
schema (per-DB mean 0.710) above both fine-tuned verifiers (encoder 0.670, Qwen-1.5B 0.659),
making the "reasoning transfers, fitting does not" pattern visible schema by schema.

**What the verifier actually uses.** An input ablation of the frozen judge confirms the mechanism.
Shown only the *question and SQL*, the GPT-4o-mini judge already scores AUROC 0.692; adding the full
*schema* changes nothing (0.688, Δ −0.004), and only BIRD's external-knowledge *evidence* helps
(+0.031 → 0.724). The frozen verifier reasons about whether the query answers the question rather
than looking up schema content — which is precisely why it generalizes to unseen schemas, and the
mirror image of the fine-tuned verifiers, whose in-domain edge comes from schema-specific patterns
that do not transfer.

**Consequence.** Two distinct problems. A **per-deployment** verifier (trained on a database's own
schemas) is an excellent, near-zero-inference-cost option at ≈ 0.77–0.79. A **universal** verifier,
on this evidence, is a *large frozen reasoning model*: no small fine-tuned verifier (≤ 1.5B, encoder
or generative) beats GPT-4o's 0.77 on transfer. Whether fine-tuning a *large* (7B+) generative judge
transfers is the open question this leaves.

### 4.5 Calibration and selective prediction

**Calibration.** The combined score (cross-fit logistic over all signals) is well-calibrated and can
be used directly as P(correct); the raw individual signals are not (Figure `paper1_reliability.png`):

| score | AUROC | ECE |
|---|---|---|
| string self-consistency | 0.675 | 0.212 |
| verifier (GPT-4o, raw P) | 0.770 | 0.319 |
| verifier (Claude, raw P) | 0.776 | 0.210 |
| **two-provider ensemble (cross-fit logistic)** | **0.822** | **0.031** |

The raw signals are over-confident (ECE 0.21–0.32); the cross-fit ensemble is well-calibrated
(ECE 0.031) and can be used directly as P(correct).

**Selective prediction.** Empirical risk–coverage (combined score, threshold calibrated on a
held-out half; full curves in Figure `paper1_risk_coverage.png`):

| target risk | self-consistency | all signals |
|---|---|---|
| 0.20 | — (no valid subset) | 9% coverage @ 14% risk |
| 0.30 | — | 27% coverage @ 24% risk |
| 0.40 | — | 58% coverage @ 38% risk |

Self-consistency cannot form a low-risk subset at any threshold; the logic-aware combination yields
useful empirical abstention frontiers. We are explicit that the *distribution-free* (PAC) certificate
is conservative under the regime's low base accuracy (0.45) — only "all signals at α = 0.40"
certifies (16% coverage / 22% risk) — so a tight guarantee needs a stronger generator (higher base
accuracy) or more calibration data, not a better signal. We therefore report the empirical frontier
as the practical result and the certificate as a conservative lower bound.

### 4.6 Reranking headroom (accuracy, not just abstention)

Selecting one of the K = 8 samples by different rules: modal/self-consistency 0.451, best-logprob
0.446, **oracle best-of-8 0.506** — a **+5.5-point** accuracy headroom a perfect reranker could
capture, which cheap signals (logprob) miss. This motivates verifier-guided selection/generation as
the route to *accuracy* gains (future work), complementary to the abstention result.

## 5. Discussion

Correctness UQ for text-to-SQL is tractable, but only with signals that engage the query's *logic*.
The popular black-box statistical signals are a ceiling, not a solution, on hard multi-table data.
Two practical recipes follow: (i) for a fixed deployment, a cheap trained verifier (or even a
feature classifier) delivers ≈ 0.78 at negligible inference cost; (ii) for an open/universal setting,
use a reasoning judge — the frozen LLM already transfers at 0.77, and the open question is whether
fine-tuning a generative judge beats it.

## 6. Limitations and planned strengthening runs

Current limitations, each with the run that addresses it:

- **Generators within one provider.** The dichotomy is shown for two generators (gpt-4o-mini and
  gpt-4.1-mini, §4.3), but both are OpenAI; *planned:* an open-weight generator and a second
  benchmark (Spider slice) to broaden further.
- **Frozen verifiers are within one provider.** *Planned (high priority):* a cross-provider judge
  (e.g., Claude / Gemini) on the same questions; the target result is that an independent-provider
  judge also beats self-consistency and that the two judges' errors are not identical.
- **The "memorizes schema surface form" claim (§4.4) is currently inferential.** *Planned:* a
  diagnostic — verifier-input ablation (question+SQL vs +schema vs +evidence+schema), per-DB
  performance vs schema/domain uniqueness, and retraining with column names normalized/removed — to
  show directly what the fine-tuned verifier relies on.
- **PAC certificates are loose** because base accuracy is 0.45; a stronger generator should make them
  fire at useful coverage (§4.5 reports the empirical frontier as the practical result).
- **Universal trained verifier — scale is the open variable:** a fine-tuned small generative judge
  (Qwen2.5-1.5B, LoRA) plateaus at the same transfer wall as the encoder (§4.4, LODO 0.659); the
  remaining test is whether a *large* (7B+) fine-tuned generative judge transfers, since the only
  verifier that generalizes across schemas so far is the large frozen judge.

## 7. Conclusion

On hard text-to-SQL, *what predicts correctness* is not sampling agreement, structure, or even
white-box log-probability — those plateau at AUROC ≈ 0.61–0.68 — but **verification**: an LLM judge
that reasons about whether the query answers the question. Independent-provider judges make different
errors and ensemble to AUROC 0.82 with calibrated probabilities, enabling empirical abstention the
sampling baseline cannot. Verifiers can be trained, but a fine-tuned model overfits schemas and fails
to transfer (≈ 0.78 in domain, ≈ 0.66 across schemas), so a *universal* verifier must be a model that
generalizes by reasoning. The map is the contribution: it tells practitioners which signal to pay
for, and when.

---

### Reproducibility

Scripts: `bird_generate.py` (generation + execution + logprobs), `bird_verify.py` (LLM verifier,
`--model`), `bird_correctness_uq.py` (unified comparison + bootstrap), `bird_exec_uq.py`,
`bird_abstention.py` (risk–coverage + certificate), `verifier_probe.py` (cheap classifier),
`paper1_figures.py` (figures + ECE table), `server_experiments/exp1_finetune_verifier.py`
(trained encoder verifier, in-dist + LODO), `server_experiments/exp3_finetune_llm_judge.py`
(generative judge, zero-shot + LoRA). Cached generations, verifier scores, and trained-verifier
results are in `data/` and `server_experiments/results/`; figures in `paper/figures/`.
