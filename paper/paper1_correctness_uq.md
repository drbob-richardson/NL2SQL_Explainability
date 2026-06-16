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
sampling self-consistency, execution self-consistency, structural priors, and semantic
schema-linking confidence all sit at AUROC ≈ 0.62, with only marginal gains from combining them.
**Logic-aware signals break the ceiling:** an LLM verifier (0.72; an independent stronger judge 0.77)
and white-box sequence log-probabilities (0.67) carry complementary information, combine to AUROC
0.76 and a well-calibrated correctness probability (ECE 0.05), and support useful abstention
frontiers (e.g., 27% coverage at 24% selective risk) where self-consistency yields no valid low-risk
subset — though distribution-free certificates remain conservative under the regime's low base
accuracy. We further ask
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

1. **A correctness ceiling for black-box statistical UQ** (AUROC ≈ 0.62 on BIRD), shared by
   self-consistency, execution self-consistency, structural priors, and schema-linking confidence.
2. **Two signals that break it** — an LLM verifier and white-box log-probabilities — with
   bootstrapped CIs excluding zero, and a cross-model robustness check.
3. **A selective-prediction frontier**: the combined score supports risk-controlled abstention the
   self-consistency baseline cannot.
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
  - *Black-box statistical:* string self-consistency (largest identical-query cluster),
    execution self-consistency (largest execution-result cluster), structural / schema-linking
    confidence (companion-paper posteriors composed over the generated query).
  - *Logic-aware:* mean sequence **log-probability** of the modal query (white-box); an **LLM
    verifier** that is shown (question, evidence, schema, candidate SQL) and answers Yes/No, with
    P(correct) read from the Yes/No first-token logits — using `gpt-4o-mini` and, independently,
    `gpt-4o`.
- **Metrics.** Tie-robust AUROC for correctness; cross-fit logistic combination (parity split) with
  2,000-sample bootstrap CIs on the AUROC delta; ECE; risk–coverage with a distribution-free
  (Bonferroni-over-grid, δ = 0.1) certificate and an empirical frontier.
- **Cost & reproducibility.** All generation/verification is safe-by-default (cost estimate, no
  calls without `--run`, hard caps, caching); total OpenAI spend for the study ≈ \$0.9. Code and
  cached signals are released.

## 4. Results

### 4.1 The black-box correctness ceiling

| signal (alone) | AUROC for correctness |
|---|---|
| string self-consistency | 0.616 |
| execution self-consistency | 0.613 |
| schema-linking graph confidence | 0.553 |
| *"does it execute / return rows"* | 0.500 (chance) |

Every black-box statistical signal plateaus at ≈ 0.62, and combining them yields only marginal gains
that stay near the same ceiling (self-consistency + execution self-consistency: +0.020, CI
[+0.005, +0.035]). Wrong queries execute and return rows just fine, so executability is
uninformative. This ceiling is a property of the regime, not of sample size (it is stable from
n = 200 to n = 800).

### 4.2 Logic-aware signals break the ceiling

Combined with string self-consistency via cross-fit logistic; AUROC and bootstrap CI of the delta:

| signal | AUROC alone | combined Δ over self-consistency (95% CI) |
|---|---|---|
| log-probability (white-box) | 0.669 | +0.054 [+0.027, +0.081] |
| LLM verifier (`gpt-4o-mini`) | 0.724 | +0.095 [+0.060, +0.131] |
| LLM verifier (`gpt-4o`, independent) | **0.770** | — |
| verifier + log-probability | — | +0.109 [+0.071, +0.147] |
| all signals | **0.763** | +0.117 [+0.077, +0.156] |

All positive deltas have P(Δ > 0) = 1.00 at n = 800. The signals that work are precisely those not
blind to query logic: white-box token confidence and a verifier that can *reason* about whether the
aggregation, filters, and joins answer the question.

### 4.3 The verifier is not a self-agreement artifact

A verifier from the same model family as the generator could merely echo it. It does not: an
**independent, stronger** judge (`gpt-4o` judging `gpt-4o-mini`'s SQL) scores **0.770 > 0.724**, i.e.
*better*, and the two judges correlate only **0.49** — they capture different errors and combine
(all-signals 0.763).

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
| self-consistency (`top_prob`) | 0.619 | 0.365 |
| verifier (GPT-4o, raw P) | 0.770 | 0.319 |
| **combined (cross-fit logistic)** | 0.763 | **0.046** |

Self-consistency and the raw verifier probability are badly over-confident (ECE > 0.3); the combined
logistic is calibrated (ECE 0.046).

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

- **One generator family** (`gpt-4o-mini`/`gpt-4o`) and an 800-question BIRD slice. *Planned:* repeat
  the ceiling/verifier comparison on a second generator (an open-weight model) and a Spider slice,
  even at n ≈ 200–300, to show the dichotomy is not unique to GPT-4o-mini.
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

On hard text-to-SQL, *what predicts correctness* is not sampling agreement or structure — those
plateau at AUROC ≈ 0.62 — but **reasoning-based signals**: an LLM verifier and white-box
log-probabilities, which combine to 0.76 and enable risk-controlled abstention the baseline cannot.
Verifiers can be trained, but a fine-tuned encoder overfits schemas and fails to transfer (0.785 in
domain, 0.670 across schemas), so a *universal* verifier must be a model that generalizes by
reasoning. The map is the contribution: it tells practitioners which signal to pay for, and when.

---

### Reproducibility

Scripts: `bird_generate.py` (generation + execution + logprobs), `bird_verify.py` (LLM verifier,
`--model`), `bird_correctness_uq.py` (unified comparison + bootstrap), `bird_exec_uq.py`,
`bird_abstention.py` (risk–coverage + certificate), `verifier_probe.py` (cheap classifier),
`paper1_figures.py` (figures + ECE table), `server_experiments/exp1_finetune_verifier.py`
(trained encoder verifier, in-dist + LODO), `server_experiments/exp3_finetune_llm_judge.py`
(generative judge, zero-shot + LoRA). Cached generations, verifier scores, and trained-verifier
results are in `data/` and `server_experiments/results/`; figures in `paper/figures/`.
