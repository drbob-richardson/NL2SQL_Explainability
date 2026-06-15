# Methods design: a BNP prior over executable SQL query graphs

*Design note distilled from the verified Pillar-4 prior survey (2026-06-13, 24/25 claims
confirmed). This pins down deliverables (b) theory and (c) methods.*

## Chosen prior: Pitman–Yor adaptor grammar (PYAG) over the SQL grammar

An **adaptor grammar** augments a PCFG so that each (adapted) nonterminal `A` caches
whole subtrees and re-generates them via a Pitman–Yor process:

- Base distribution `G_A` = the PCFG expansion of `A` (recursively).
- Adaptor = Pitman–Yor process `PY(d_A, α_A, G_A)` → a per-nonterminal CRP over
  previously-generated **subtrees** (fragments). Discount `d_A=1` recovers the PCFG;
  `d_A=0` gives a Dirichlet-process adaptor.
- **Support invariance:** the adapted distribution has the *same support* as the base
  grammar. ⇒ zero mass outside grammar-valid SQL. (Johnson/Griffiths/Goldwater 2006.)

Why this fits NL2SQL UQ:
| Requirement | How PYAG delivers |
|---|---|
| (a) support = valid queries | adaptor preserves base-grammar support (context-free part) |
| (b) inference tractable w/ LLM | component-wise MH; **LLM samples = on-the-fly proposal** |
| (c) interpretable uncertainty | posterior over derivation trees; fragment-reuse frequencies |
| (d) localize uncertainty | per-nonterminal cache = which fragments are (un)stable |

## The hard part = the contribution: context-sensitive (executable) support

Adaptor grammars guarantee only **context-free** validity. SQL executability is
**context-sensitive**: join keys must type-match, predicates reference in-scope columns,
GROUP BY consistency, aggregate/scope rules. No surveyed BNP prior enforces this
automatically — so this is where the novelty sits. Two routes (likely combine):

1. **Typed / attribute grammar.** Annotate grammar nonterminals with schema-derived
   attributes (available tables, column types, scope) so the *base* support is exactly
   the executable-query manifold for a given schema. The adaptor then caches fragments
   *within* that constrained base. Cleanest theoretically.
2. **Per-step masking sampler (GFlowNet-style).** Build the AST sequentially, masking
   actions that violate type/key/scope constraints at each step → exact valid support,
   amortizable to LLM scale (cf. DAG-GFlowNet's edge-masking, Deleu et al. UAI 2022).
   Cleanest for scalability.

## Inference

- **Treat the LLM as the proposal distribution** over derivation trees. Sample K SQL
  strings → parse each to a grammar derivation tree (extend `query_graph.py`, which
  already canonicalizes to a typed graph, into a grammar-aligned derivation).
- Component-wise Metropolis–Hastings over the adaptor seating (which fragments are
  cached), or stochastic variational inference. Posterior = distribution over derivation
  trees / fragment allocations given the question + K proposals.
- **Uncertainty readouts:** (i) posterior entropy over structures (global confidence);
  (ii) per-fragment posterior reuse probability (localized confidence); (iii) selective
  prediction: abstain when posterior mass on the MAP structure < threshold.

## Fallback / competitor

Constrained-DAG posterior: **DiBS** (soft-acyclicity continuous embedding, gradient VI,
NeurIPS 2021) or **DAG-GFlowNet** (exact valid support via masking, UAI 2022). Guarantees
acyclicity only, not typing → use as scalability fallback, and borrow GFlowNet masking for
route 2 above.

## Secondary prior (optional)

Feature-allocation prior (IBP / beta process) treating query components (tables, joins,
predicates, aggregations) as overlapping latent **features**, composed *with* the grammar
prior that assembles them into a valid AST. Open question worth one experiment.

## Evaluation plan

- Datasets: Spider (saturated, sanity), BIRD, Spider 2.0 (hard — UQ matters most).
- Baselines to beat: full-sequence probability rescaling; consistency/self-consistency
  sampling (current structural baseline in `uncertainty.py`); sub-clause Platt scaling
  (EMNLP 2025); **RTS conformal abstention (SIGMOD 2025)**.
- Metrics: **risk–coverage / selective-prediction curves**, AUROC of confidence vs.
  execution correctness, ECE (calibration). Headline: better selective prediction at
  matched coverage, *plus* localization no baseline offers.

## Open questions (from the survey)

1. Encode context-sensitive SQL constraints so the PYAG base support = executable manifold
   (typed/attribute grammar vs. masking).
2. Amortize adaptor-grammar inference to LLM scale, or use GFlowNet amortized constrained
   sampler with exact valid support.
3. Compose a feature-allocation prior over components with the grammar prior.
4. Calibrate posterior uncertainty against ground-truth correctness (fragment *reuse* ≠
   execution *correctness* — must be shown empirically, not assumed).
