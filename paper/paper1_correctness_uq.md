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
schema-linking confidence all sit at AUROC ≈ 0.62 and add nothing to one another. **Logic-aware
signals break the ceiling:** an LLM verifier (0.72; an independent stronger judge 0.77) and
white-box sequence log-probabilities (0.67) carry complementary information, combine to AUROC 0.76,
and enable risk-controlled abstention that self-consistency structurally cannot offer (e.g., 27%
coverage at 24% selective risk where the baseline yields no valid low-risk subset). We further ask
whether the verifier can be *trained*: a fine-tuned transformer verifier reaches in-distribution
AUROC 0.785 (beating a cheap feature classifier and matching the frozen GPT-4o judge) but **does not
transfer** — leave-one-database-out AUROC drops to 0.670, the same overfitting the cheap classifier
showed. The frozen LLM judge's 0.77 is, by contrast, already a transfer number, because it reasons
rather than memorizes schema surface form. The practical upshot: correctness UQ for text-to-SQL is
real but lives in *reasoning-based* signals; a fine-tuned verifier is an excellent **per-deployment**
tool, while a **universal** verifier requires a model that generalizes across schemas.

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

Every black-box statistical signal plateaus at ≈ 0.62, and they do not complement each other
(self-consistency + execution self-consistency adds only +0.020, CI [+0.005, +0.035]). Wrong queries
execute and return rows just fine, so executability is uninformative. This ceiling is a property of
the regime, not of sample size (it is stable from n = 200 to n = 800).

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

### 4.4 Can the verifier be trained? In-distribution yes, transfer no

We fine-tune a transformer verifier on (question + evidence + schema + SQL → correct), 6,400
execution-labeled pairs, evaluated with the same leave-one-database-out (LODO) protocol used for
schema linking.

| verifier | in-distribution AUROC | leave-one-DB-out (transfer) |
|---|---|---|
| cheap feature classifier (embeddings + logprob + self-consistency) | 0.768 | 0.661 |
| fine-tuned encoder, RoBERTa-base | 0.778 | — |
| **fine-tuned encoder, ModernBERT-base (full schema context)** | **0.785** | **0.670** |
| *frozen `gpt-4o` judge (zero-shot)* | — | *0.770 (already a transfer number)* |

In-distribution, a fine-tuned verifier is strong — it beats the cheap classifier and matches the
frozen GPT-4o judge. But it **does not transfer**: LODO AUROC falls to 0.670 (a 0.115 gap), matching
the cheap classifier's collapse, and worst on the most domain-specific held-out schemas. Long
context (ModernBERT, no truncation) does not fix it — the fine-tuned encoder *memorizes schema
surface form*. The frozen LLM judge's 0.77, by contrast, *is* its cross-schema number: reasoning
generalizes where fitting does not.

**Consequence.** There are two distinct problems. A **per-deployment** verifier (trained on a given
database's own schemas) is an excellent, near-zero-inference-cost option at ≈ 0.78. A **universal**
verifier is *not* a fine-tuned encoder; the candidate route is a fine-tuned *generative* judge whose
reasoning may transfer — an experiment we report as ongoing (Section 6).

### 4.5 Selective prediction: abstention the baseline cannot do

Empirical risk–coverage (combined score), threshold calibrated on a held-out half:

| target risk | self-consistency | all signals |
|---|---|---|
| 0.20 | — (no valid subset) | 9% coverage @ 14% risk |
| 0.30 | — | 27% coverage @ 24% risk |
| 0.40 | — | 58% coverage @ 38% risk |

Self-consistency cannot form a low-risk subset at any threshold; the logic-aware combination can.
The distribution-free PAC certificate is conservative under the regime's low base accuracy
(0.45) — only "all signals at α = 0.40" certifies (16% coverage / 22% risk) — so tighter guarantees
require a stronger generator (higher base accuracy) or more calibration data, not a better signal.

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

## 6. Limitations and ongoing work

- **One generator family** (`gpt-4o-mini`/`gpt-4o`) and an 800-question BIRD slice; broader
  generators and benchmarks (Spider, Spider 2.0) would test breadth.
- **Frozen verifiers are within one provider**; a cross-*provider* judge is untested.
- **PAC certificates are loose** because base accuracy is 0.45; a stronger generator should make them
  fire at useful coverage.
- **Universal trained verifier — ongoing:** we are fine-tuning a small *generative* LLM judge (LoRA)
  with the same LODO protocol to test whether a reasoning judge transfers where the encoder did not;
  results will report its in-distribution vs. transfer gap against the frozen-judge bar (0.77).

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
`server_experiments/exp1_finetune_verifier.py` (trained encoder verifier, in-dist + LODO),
`server_experiments/exp3_finetune_llm_judge.py` (generative judge, ongoing). Cached generations,
verifier scores, and trained-verifier results are in `data/` and `server_experiments/results/`.
