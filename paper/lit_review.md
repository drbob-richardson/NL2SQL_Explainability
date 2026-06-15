# Literature review & gap analysis

*Produced 2026-06-13 via a fan-out web-search harness (24 primary sources, 110 claims
extracted, 25 adversarially verified, 22 confirmed). Confidence levels and vote tallies
noted per claim. This is a living document — re-run before submission, the area moves fast.*

## Verdict

**"BNP priors on query graphs for NL2SQL uncertainty quantification" is genuinely novel
and defensible.** No surviving evidence documents any prior work combining Bayesian
nonparametrics + priors over graphs + calibrated uncertainty over generated SQL/code.
The contributing fields are mature but disconnected.

## Pillar 1 — NL2SQL state of the art

- **LLM-based methods dominate** (supersede pre-trained-LM parsers). *IEEE TKDE 2025
  survey, [2406.08426](https://arxiv.org/pdf/2406.08426). 3-0.*
- **Public benchmarks** (difficulty gradient): Spider (~91% solved, saturated), BIRD
  (12,751 pairs, 95 DBs, 33 GB, dirty data + external knowledge), Spider 2.0 (632
  enterprise tasks, ~21% solved). Plus Spider-DK/SYN/CG/Realistic, WikiSQL, KaggleDBQA,
  CoSQL, DuSQL. *3-0.* Metric = **execution accuracy**.
- **Unsolved at the hard end**: o1-preview 21.3% on Spider 2.0 vs 91.2% Spider 1.0,
  73% BIRD. *ICLR 2025 Oral, [2411.07763](https://arxiv.org/abs/2411.07763). 3-0.*
  → Large headroom ⇒ knowing *when wrong* (UQ/abstention) is high value.

## Pillar 2 — UQ for SQL / code (the gap)

Recognized, **active but unsolved** (2024-2026 cluster). Every method flattens the output:
- Full-sequence probability **rescaling** (scalar over tokens). *[2411.16742](https://arxiv.org/abs/2411.16742). 3-0.*
- **Sub-clause-frequency + multivariate Platt scaling** (post-hoc, over clause counts).
  *EMNLP 2025, [2025.emnlp-main.859](https://aclanthology.org/2025.emnlp-main.859/). 3-0.*
- **RTS: conformal abstention** localized to schema-linking branch points (probabilistic
  guarantee, abstention). *SIGMOD/PACMMOD 2025, [2501.10858](https://arxiv.org/html/2501.10858).
  3-0.* → **nearest methodological neighbor / primary baseline.**
- **Consistency sampling** = strongest black-box signal. *[2508.14056](https://arxiv.org/pdf/2508.14056). 3-0.*
- Structure-aware for **code**: functional entropy ([2605.28500]) and **AST structural
  entropy** ([2508.14288]) — but ASTs used only as *features for an entropy heuristic*,
  not a Bayesian prior/posterior. **Nearest neighbor; stops one step short.** *3-0.*

**White space (3-0):** no existing SQL UQ method places a generative Bayesian (let alone
nonparametric) posterior over the query graph/AST.

## Pillar 3 — SQL as a graph

Canonical, but **encoder/input side only**: RAT-SQL relation-aware attention
([1911.04942]), LGESQL line graphs ([2106.01093]). Deterministic encoders of an input
graph; **nobody makes the output query graph a probabilistic object.** *3-0.* This
asymmetry is the structural hook: the graph machinery exists, unused on the output side.

## Pillar 4 — BNP priors on graphs/trees/combinatorial structures

**NOT independently verified in this batch** (key limitation). Candidate families to pin
down in the focused follow-up survey: order/modular DAG priors and MCMC-over-DAGs;
exchangeable random graphs / graphons / edge-exchangeable (Caron-Fox); nested CRP,
tree-stick-breaking, Dirichlet diffusion trees, Kingman's coalescent, Mondrian process;
**adaptor grammars / PYP-PCFG over a grammar** (handle type-valid structures natively);
CRP / Pitman-Yor / IBP / Beta process over combinatorial object structure.

## Pillar 5 — Intersection

No surviving evidence of BNP + graph priors + structured-output LLM uncertainty, nor any
"posterior over query graphs". *Synthesis, medium confidence — rests partly on
absence-of-evidence because Pillar 4 was not independently surveyed here.*

## Three refuted claims (do NOT use)

1. TKDE survey ignores UQ entirely — *overstated, refuted 1-2.*
2. BIRD 40.08% vs 92.96% human as worded — *refuted on attribution 1-2* (gap is real).
3. EMNLP 2025 = **first** SQL calibration benchmark — *refuted 1-2.* **Do not claim
   "first calibration for SQL."** The novel claim is the first *structural Bayesian posterior.*

## Caveats

Fast-moving area; strongest UQ-for-SQL sources are 2024-2026. Re-run a literature watch
before submission. Pillar 4 needs a dedicated survey (in progress) to convert the novelty
verdict from absence-of-evidence to a positive comparison of candidate priors.
