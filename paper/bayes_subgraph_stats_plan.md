# Bayesian Selection of Relevant Subgraphs (stats paper first) — framing + the recovery result

**Working title:** *Bayesian Selection of Relevant Subgraphs, with an Application to Grounding Large
Language Models.*  Target: a Bayesian statistics venue (JRSS-C / Bayesian Analysis / TAS / Statistical
Science), NOT an IR venue. The contribution is a statistical model + theory; text-to-SQL schema linking
is the motivating application.

## 1. The statistical object
Given a graph G=(V,E) (tables = nodes, foreign keys = edges) and node-level relevance evidence
phi_t (cheap features: query-table cosine, token overlap), select the relevant subset S ⊆ V. Model S
by a binary vector x, with a pairwise Markov random field (Ising) prior coupled to the graph:
    p(x) ∝ exp( sum_t a_t x_t + beta * sum_{(s,t) in E} x_s x_t ),   a_t = theta^T phi_t  (unary log-odds).
The posterior marginal m_t = E[x_t] ranks tables; beta>=0 is a ferromagnetic coupling favouring the
selection of connected tables jointly. beta=0 = independent selection (marginal/cosine baseline).

## 2. Three statistical contributions (what makes it a stats paper, not "we used an MRF")
(a) **Sequential inference = interpretable belief growth + a stopping rule.** Inferred sequentially
(commit argmax, update each committed table's FK-neighbours by +beta to their conditional log-odds, stop
when no remaining posterior clears a threshold), the model yields an auditable belief trace and a
principled VARIABLE-SIZE stopping rule (optimal-stopping / expected-utility view). Demonstrated figure:
a join table invisible to cosine (prior P=0.35) recovered to P=0.64 via the FK edge, stopping at the
exact gold set (paper/figures/schema_growth).
(b) **A recovery / selection-consistency result (the theoretical backbone).** See section 3.
(c) **Uncertainty over structure.** A coherent posterior and credible sets over subgraphs, and a
transparent account of WHEN the graph helps (the recovery condition) vs hurts (over-selection) — which
formalises the audit's empirical "connectivity boundary" as a theorem.

## 3. The recovery result (planted-bridge model) — the crux
Setup. Relevant set contains a SALIENT node s (unary log-odds a_s large, committed first) and a BRIDGE
node b that is relevant but cosine-invisible: a_b = -delta with delta>0, so the MARGINAL selector
(threshold at logit(tau)=0 for tau=1/2) MISSES b whenever delta>0. Edge (s,b) in E. Irrelevant nodes r
have unary log-odds a_r<0 with margin |a_r|.

Claim (sequential-conditional recovery + selectivity).
- After committing k relevant neighbours, the bridge's conditional log-odds is a_b + k*beta = k*beta - delta.
  **b is RECOVERED iff  beta > delta / k.**  (connectivity k lowers the coupling needed.)
- An irrelevant node r with k' committed relevant neighbours is a FALSE POSITIVE iff a_r + k'*beta > 0,
  i.e. **over-selection iff beta > |a_r| / k'.**
- Therefore recovery WITHOUT over-selection is possible iff
        delta / k  <  beta  <  |a_r| / k'      (a non-empty window),
  i.e. iff the bridge's prior deficit (scaled by its relevant-connectivity) is smaller than the
  irrelevant nodes' prior margin (scaled by their spurious connectivity):  **delta * k' < |a_r| * k.**
- Interpretation = the connectivity boundary as a theorem: structure helps exactly when true relevant
  bridges are better-connected to the relevant set than irrelevant nodes are, relative to their prior
  gaps. This predicts the empirical beta tradeoff (recall up, precision down as beta grows) and the
  single-hop failure (no bridge, k=0 -> recovery needs beta=inf, only over-selection remains).
Extensions to state as propositions: (i) a probabilistic phase transition when a_t are random
(Gaussian) -> recovery probability is a smooth sigmoid in (k*beta - delta)/sigma; (ii) posterior-marginal
(joint MRF) version, not just sequential; (iii) relation to Ising selection-consistency / SBM thresholds.

## 4. Honest positioning (consistent with the audit)
- NOT an accuracy claim: sequential greedy F1 ~0.65 and the joint MRF is better; a shortest-path heuristic
  matches the posterior on accuracy (the audit). We say so. The value is interpretability, UQ over the
  subgraph, a stopping rule, and the RECOVERY THEORY that says WHEN/why structure helps.
- The "idea not apparatus" critique bites less: the contribution is the model + theorem + interpretation,
  not a benchmark number.

## 5. Open literature questions (deep-research sweep running: wf_22c4d0df-a46)
Is the MRF-subgraph-selection model a known stats model (Ising priors for variable selection: Li-Zhang,
Stingo, Vannucci)? Is the planted-bridge recovery a new phase transition or a special case of
planted-clique / SBM / graph-signal recovery? Is Bayesian schema linking / LLM context selection
unoccupied? -> determines whether the MODEL, the THEORY, or the APPLICATION is the novel part.

## 6. Plan if the white space holds
- Formalise section 3 as 1-2 propositions with proofs (deterministic recovery window + probabilistic
  phase transition). This is the paper's spine.
- Model: joint-MRF posterior + sequential inference; empirical bridge-recovery on BIRD/Spider; the figure.
- UQ: credible sets over subgraphs; calibration of subset posterior.
- Application framing: grounding LLMs (schema linking) as Bayesian structured selection; honest accuracy
  note; interpretability/trustworthy-AI payoff.
- Simulation: planted-bridge synthetic graphs confirming the phase transition matches the theory.

## 7. Simulation CONFIRMS the recovery theorem (scripts/sim_bridge_recovery.py)
Planted-bridge sim (s: a=2.5; bridge b: a=-0.6; 6 irrelevant a~-N(1.2,0.4); spurious edge p=0.4), sweep beta:
recover b% crosses 50% at beta=0.6 (=delta); over-select% crosses ~50% near beta=1.2 (=|a_r|); the CLEAN
(recover b, no false positive) rate PEAKS at beta~0.9, inside the predicted window [delta, |a_r|]=[0.6,1.2].
The phase transition matches the theorem -> the connectivity boundary, as a provable phase transition.

## 8. Preliminary literature positioning (2 manual searches; full sweep wf_22c4d0df-a46 running)
- MODEL is NOT novel: graph-structured Bayesian variable selection / Ising priors on inclusion indicators
  is established (Li & Zhang 2010; Rockova & George EMVS; spike-and-slab + MRF priors; MRF structure
  learning). Do NOT claim the model as new.
- APPLICATION partly occupied: EviLink (arXiv 2605.29670, 2025) does uncertainty-aware multi-path schema
  linking and criticizes deterministic keep-or-drop -- a direct neighbor to read and cite.
- NOVELTY, if any, is the RECOVERY THEOREM (delta/k < beta < |a_r|/k' phase transition) + the interpretable
  sequential belief-growth/stopping framing + the recovery-window-as-connectivity-boundary result. Pending
  the full sweep's check of whether this is subsumed by planted-clique/SBM/Ising-selection-consistency.
- HONEST RECALIBRATION: this is most likely a solid APPLIED-Bayesian-modeling contribution and/or the
  constructive centerpiece of the audit/RSS paper -- NOT a standalone methods breakthrough. Whether it
  merits its own paper hinges on the recovery-theorem novelty. If subsumed, fold the model+figure+theorem
  into the audit paper as the "what a Bayesian should do here" section (still a strong addition).

## 9. FULL SWEEP VERDICT (wf_22c4d0df-a46, 101/102 agents, adversarially verified) -- GO
- MODEL: established prior art (Li & Zhang 2010 JASA; Stingo & Vannucci 2011; Peterson/Stingo/Vannucci 2016
  Stat.Med.; Chang/Kundu/Long 1604.07264). Exact form p(gamma|G) ∝ exp(a·1'gamma + b·gamma'G·gamma). BUILD ON
  IT, cite it; do not claim as new.
- RECOVERY THEOREM: LIKELY GENUINELY NOVEL. Verified NOT subsumed by any of the four candidate literatures:
  (a) MRF-selection "phase transition" = prior pathology (model-size explosion), not recovery;
  (b) Ising selection-consistency (Santhanam-Wainwright 0905.2639; Anandkumar et al. AoS 2012) = edge-structure
      recovery from i.i.d. samples, frequentist, n=Omega(J_min^-2 log p) -- different problem;
  (c) SBM/planted-clique thresholds (Kesten-Stigum, Chernoff-Hellinger, Gaussian-SBM SNR=1) = planted-community
      detection from graph observations -- different setup;
  (d) nearest Bayesian neighbors: Peterson (sim only, no theorem); Chang (oracle under beta-min, signals bounded
      AWAY from zero = antithesis of recovering a low-signal connected node).
- APPLICATION: unoccupied as formal Bayesian structured selection. EviLink (2605.29670) = per-item INDEPENDENT
  Beta-Binomial (marginal) -- exactly the selector the theorem beats (perfect foil). SchemaGraphSQL (2505.18363)
  = deterministic FK pathfinding. Cite both.
- FRAMING (recommended): LEAD WITH THE RECOVERY THEOREM as the statistical contribution; NL2SQL schema linking as
  motivating application/case study; EviLink-style marginal selector = the incumbent the theorem improves on.
- VENUE: JASA (Applications & Case Studies, or Theory & Methods if the theory is developed) or Annals of Applied
  Statistics; Bayesian Analysis alternative; TAS only if lightened. A step above TAS.
- HONEST CAVEATS: (1) "likely novel" per sweep != proven; needs a rigorous written proof + expert check
  (Vannucci/Stingo lineage). (2) The deterministic theorem (recover iff beta>delta/k) is SIMPLE; teeth require
  the probabilistic phase transition (random a_t -> smooth recovery, sim already shows it), the joint-MRF marginal
  version, and proper conditions/minimax framing. That development = "note" vs "methods paper".
- DECISION: promote to a standalone stats-paper CANDIDATE (lead paper). Next spine step: formalize the theorem
  (deterministic recovery window + probabilistic phase transition + joint-MRF version) as 2-3 propositions with
  proofs. If the proofs hold and survive expert review, this outranks the ECIR structure paper.
