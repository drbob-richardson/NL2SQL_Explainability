# The American Statistician — article scaffold

**Working title:** *Can Bayes Help Information Retrieval for LLM Systems?*
**Alternatives:** "When Does Bayes Help Retrieval in the LLM Era?" · "Structure, Not Uncertainty:
A Bayesian Audit of Retrieval for Large Language Models."

**Status:** scaffold/outline. Built from the full exploration in `CHECKPOINT.md`,
`retrieval_exploration.md`, `paper2_*`. This is the stats-venue track; the CS-conference track
(structured retrieval method) and Paper 1 (correctness) proceed in parallel.

---

## 0. Why this article, and why The American Statistician

TAS publishes accessible, broadly-interesting, rigorous pieces — "Statistical Practice" and "General"
columns especially welcome an honest, well-instrumented examination of *whether a fashionable method
actually helps*. "Can Bayes help X?" with a nuanced, reproducible "barely and sometimes, and here is
exactly when" is squarely that genre. The LLM/RAG era makes it timely: Bayesian and "uncertainty"
language is invoked all over retrieval-augmented generation, mostly without a controlled check of
whether it beats trivial baselines. We provide that check.

**The honest answer (the hook):** *Bayes helps retrieval mainly through **structure**, not through
**uncertainty**.* It earns a modest, real gain when there is relational structure a point estimate
ignores (and a combinatorial object to marginalize over); it largely **fails** at the
calibration/decision layer where it is most often invoked, because the LLM's output distribution is a
biased *proposal*, not a posterior, and simple heuristics already capture the calibration.

**JASA upside:** if the structured retriever wins decisively at enterprise scale (large-schema
validation), the *method* — Bayesian connected-subgraph selection for schema retrieval, with theory —
is a methods contribution; the TAS article is the conceptual map, the JASA/CS paper is the method.

---

## 1. Thesis (one paragraph)

We ask, concretely and reproducibly, whether Bayesian modeling improves information retrieval for LLM
systems. Using text-to-SQL retrieval/schema-linking as a controlled testbed (with the broader RAG
literature as context), we instrument the four places Bayes can enter a retriever — calibrated
relevance/uncertainty, signal fusion, structured/joint selection, and set-level diversity — and
compare each against the trivial baselines practitioners actually use (cosine top-k, RRF, a
max-excluded-similarity confidence). The verdict is consistent and, we argue, generalizable: **the
only place Bayes reliably helps is structured joint selection** (a graph prior over the relevant
table subgraph), where it improves multi-hop recall and downstream SQL accuracy — yet even there a
one-line shortest-path heuristic captures most of the gain. At the uncertainty/decision layer Bayes
does **not** beat simple baselines. We extract three lessons about *when* structure-aware Bayes is
worth the trouble, and offer practitioner guidance.

---

## 2. Scope & honesty note (state this plainly in the paper)

- Empirical evidence is **text-to-SQL-centric** (BIRD, Spider, AMBROSIA), where retrieval = schema
  linking / table selection. General open-domain RAG is covered via the literature, not our own
  large-scale experiments. We will *not* over-claim "all IR for LLMs" from SQL evidence; we frame SQL
  as a clean, structured, executable testbed and mark which lessons are SQL-specific vs general.
- All results are reproducible (scripts + cached data); APIs were safe-by-default and cheap (≈$6 total).

---

## 3. Section outline (with anchoring evidence)

1. **Introduction — the question.** Bayes/UQ is everywhere in RAG; is it doing work? Set up the honest
   audit. State the hook.
2. **Background: "Bayesian retrieval" already exists.** Probabilistic relevance (Robertson–Spärck
   Jones → BM25), language-model IR (Ponte–Croft, Dirichlet smoothing = Bayesian predictive),
   decision-theoretic IR (Lafferty–Zhai). Lesson: much "new Bayesian retrieval" reinvents this; we
   ask what *modern* (LLM-era) Bayes adds. [anchors: landscape scan in `retrieval_exploration.md`]
3. **A taxonomy: four places Bayes can enter an LLM retriever.** (a) calibrated relevance/UQ; (b)
   fusion of heterogeneous signals; (c) structured/joint subset selection (graph priors); (d)
   set-level diversity (DPP). For each: the natural Bayesian object, the trivial baseline, and what we
   find.
4. **A controlled study in text-to-SQL.** Setup: BIRD/Spider schemas, FK graphs, gold tables,
   execution. The methods and baselines. [anchors: scripts]
5. **Results, by taxonomy cell** (the master table, §4 below).
6. **Why: three lessons.** (i) Structure beats independence only when structure exists and is the
   *right* structure (FK ≠ cosine); (ii) the LLM distribution is a biased proposal, not a posterior,
   so UQ can't be read off it; (iii) marginalization over a combinatorial object is the one
   load-bearing Bayesian operation, and it buys little over a good heuristic.
7. **Practitioner guidance.** A short decision guide: when to reach for structured/Bayesian retrieval
   vs cosine; when to *not* trust posterior uncertainty.
8. **Conclusion.** "Barely and sometimes" — but the *when* is clear and useful. Note the scale caveat
   and the method/JASA upside.

---

## 4. Master evidence table (the spine of the paper)

| Taxonomy cell | Bayesian object | Trivial baseline | Verdict | Key numbers |
|---|---|---|---|---|
| **Calibrated relevance / UQ** | posterior `P(complete\|R)`; PYP reserve | cosine max-out; reasoning verifier | **Bayes loses** | abstention AUROC: posterior 0.700 vs cosine-maxout 0.763; PYP-reserve ambiguity-detection 0.557 ≈ chance |
| **Signal fusion** | learned/Bayesian per-query fusion | RRF / cosine | **no win on clean text** | fusion 0.734 < cosine 0.778 (Spider easy regime); per-query adaptive already exists (DAT, MoR) |
| **Structured joint selection** | FK-graph MRF posterior over subgraphs | shortest-path FK closure | **Bayes helps (modestly)** | recall@\|gold\| MRF 0.805/0.822/0.787 vs cosine 0.720/0.673/0.678; vs FK-closure heuristic Δ +0.018/+0.043/+0.027; downstream EX +5.7pp [+2.1,+9.4] |
| **Structured joint selection (multi-hop RAG)** | passage-link graph prior | cosine; PageRank | **Bayes/structure helps** (generalizes!) | recall@2 +10-13pp (bridge +17pp); PageRank≈MRF≫cosine on HotpotQA |
| **Set-level diversity** | cosine-repulsion / DPP term | MMR | **no win for tables** | repulsion hurts multi-hop recall (0.829→0.790); cosine-coupling can't substitute for FK |
| **Correctness UQ (Paper 1 context)** | agreement / structure | reasoning verifier | **Bayes/agreement loses** | black-box ≤0.68 AUROC; verifier 0.77; ensemble 0.82 |
| **Ambiguity detection (context)** | sampling posterior / reserve | — | **fails** | coverage-both 1%; divergence AUROC 0.475 |

Cross-cutting: structure's gain **grows with relational complexity** (FK density, query hop count);
`oracle > full` (+6pp EX) shows precise retrieval beats showing-everything (distractors hurt).

---

## 5. The three lessons (the intellectual payoff)

1. **Structure, not uncertainty.** Bayes helps when it injects *correct structure* a point estimate
   ignores — here, FK connectivity that recovers cosine-invisible bridge tables. It is the *graph*,
   not the *Bayes*, doing the work; the embedding-similarity graph is the wrong structure (connectors
   are dissimilar), so a metadata-free correlation prior does not substitute.
2. **The LLM distribution is a proposal, not a posterior.** It is mode-collapsed (0.95 confidence on
   ambiguous questions), so calibrated uncertainty cannot be read off sampling; UQ must come from a
   separate reasoner or a simple external signal — and simple signals win.
3. **Marginalization is the one load-bearing Bayesian step, and it is cheap to approximate.** Summing
   over subset configurations (vs hard selection) buys the small MRF-over-heuristic increment; almost
   everything else billed "Bayesian" (parameter priors, posterior-as-UQ) is decorative or fails.

---

## 6. Practitioner guidance (a TAS-friendly box)

- **Reach for structured/graph-aware retrieval** when: schemas/corpora are large, queries are
  multi-hop, and a *true* relational graph (FK, citation, hyperlink) is available and aligned with
  relevance. Expect bigger gains as the graph gets denser/larger.
- **Don't bother** when: the corpus is small/clean, queries are single-hop, or the only "structure" is
  embedding similarity (that's redundancy, not relevance-coupling).
- **Don't trust posterior uncertainty for abstention** — calibrate with a simple held-out threshold on
  a transparent signal (e.g., max excluded similarity, or a reasoning verifier) instead.
- **A shortest-path / connectivity heuristic is a strong, cheap baseline** — always include it before
  claiming a fancy model is needed.

---

## 7. What would upgrade this to a methods paper (JASA / CS)

- Large-schema validation (Spider 2.0-Lite / BEAVER) showing the structured posterior **pulls clearly
  ahead of the heuristic** where dense graphs make blind closure over-include (the selectivity
  argument) — turning "modest" into "decisive."
- Theory: a bridge-recovery proposition + a distractor/recall tradeoff result (see
  `paper2_theorem1_sketch.md` and the reviewer's theory menu) explaining *when* structure helps.
- A genuinely Bayesian adaptive variant (structural prior + online posterior updating from feedback)
  — the one untested route where Bayes would be load-bearing.

---

## 8. Open tasks to write the TAS article

1. Decide scope/title and confirm TAS "Statistical Practice" vs "General" fit.
2. Lift the master table + three lessons into prose; add one clean worked example (a multi-hop query
   where cosine misses a bridge and the graph prior recovers it → correct SQL).
3. Write the background section properly citing the probabilistic-IR lineage (verified refs in the
   landscape scan).
4. Decide whether to include large-schema validation (strengthens, but optional for TAS).
5. Reproducibility appendix: scripts + cached data manifest.
