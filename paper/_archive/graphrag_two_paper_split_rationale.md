# GraphRAG write-up — two-paper split (working drafts)

These are **scaffolding drafts** to externalize where the research program is going. The model and
methodology are developed formally; every claim is tagged so the *tested / proposed* boundary is unmissable.
For the full experimental log and numbers, see [`../active_retrieval_plan.md`](../active_retrieval_plan.md).

## The split, and why it's honest (not salami)

The corpus graph plays **two structurally different epistemic roles**, and that is the reason for two papers:

| | **Paper A (CS/ML)** | **Paper B (Statistics)** |
|---|---|---|
| file | `paperA_active_graph_retrieval.tex` | `paperB_llm_oracle_measurement_error.tex` |
| the graph is… | the **covariance** of a latent relevance field (propagates *belief*) | the **instrument that identifies** a biased judge (propagates *calibration*) |
| contribution | a retrieval **method** + mechanism + an alignment **law** | a **measurement-error model** + structural **identification** |
| survives if… | you hand it a perfect judge | you hand it a different retriever |
| venue lean | TMLR base / AISTATS reach | JASA A&CS / AoAS |

They share the application (multi-hop RAG) but not the contribution, and each cites the other. Duality =
program spine; disjoint contributions = two deserved papers. (This is contingent on Paper B's go/no-go below.)

## Status legend (used inline in both `.tex`)
- **[TESTED] / [EVIDENCE]** (teal) — established in current experiments.
- **[PARTIAL]** (orange) — partial evidence; a cheap experiment would settle it.
- **[PROPOSED / UNTESTED]** (violet) — the model/methodology we're aiming at; not yet run.

## Paper A — what's real vs proposed
**Real:** GP with semantic mean + graph covariance; the **correlation-form (normalization) win** (graph−cosine
chain-completion ≈ doubles to +0.12–0.15); **UCB beats EVOI** (kernel × acquisition entangled); the **alignment
law** with gold-connectivity + the **oracle ceiling** (+0.07–0.09); positive on Hotpot/2Wiki, boundary on
MuSiQue. **Proposed:** the hierarchical **mixture kernel** `K_q = (1−λ_q)K̃_E + λ_q K̃_G` with learned `λ_q`
(oracle-λ shows modest headroom + feasibility, learned version unrun); **adaptive structure learning** (infer
the graph from early judgments — the deepest open direction); a clean **theorem** for the alignment law.
**To finish the empirical story:** real BAGEL head-to-head, firm 2Wiki end-task, nDCG, chain-recall metric.

## Paper B — what's real vs proposed
**Real (grounding):** bridge-blindness is a **structured, role-dependent** bias (gold-recall 0.35 naive → 0.82
hop-aware; low precision; degrades on long chains / adversarial corpora). **Proposed (the whole model):** the
role-dependent **confusion / ordinal-probit** observation law; the **Potts/GMRF prior** tying latent roles to
the graph; **identification by structure** (the Proposition — anchors + graph smoothness substitute for
replication/gold); blocked-Gibbs/VI **inference**; the **calibration** guarantee; and the **value-of-correction**
go/no-go.

## The two experiments that decide the program (cheap, next)
1. **Dig 1 — fit the confusion `Π` against a gold-derived role proxy** and test role-dependence directly.
   Establishes Paper B's model is more than a story. ($0, cached data.)
2. **Dig 2 — does bias-corrected relevance beat raw-judge on a downstream decision?** This is Paper B's
   go/no-go: if yes, it stands alone; if no, it folds into Paper A as a modeling remark. ($0.)

If both land, Paper B is a genuine stats contribution; Paper A is already a coherent CS/ML paper today.
