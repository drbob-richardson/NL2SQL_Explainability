# Theory: Bayesian nonparametric priors over a restricted SQL query space

*Scope fixed to the DataCamp "SQL Basics" fragment (single table, no joins). The point of
the restriction is that the space of valid query graphs becomes fully characterizable, so
a prior can be defined over a **known** support and its posterior reasoned about exactly.
This note develops the formal objects and **four candidate priors**, then recommends one.*

Notation: $[n]=\{1,\dots,n\}$; $(x)_m = x(x+1)\cdots(x+m-1)$ rising factorial;
$(x)_{m\uparrow d}=\prod_{i=0}^{m-1}(x+i d)$; $\mathbf 1[\cdot]$ indicator.

---

## 1. The fragment $\mathcal L$ and the schema

**Schema.** A single relation $T$ with typed columns $\mathcal C=\{c_1,\dots,c_m\}$, each
column $c_j$ carrying a type $\tau(c_j)\in\{\textsf{num},\textsf{text}\}$. Running example
(`airbnb_listings`): $\mathcal C=\{\texttt{id},\texttt{city},\texttt{country},
\texttt{number\_of\_rooms},\texttt{year\_listed}\}$ with
$\tau=\{\textsf{num},\textsf{text},\textsf{text},\textsf{num},\textsf{num}\}$, so
$m=5$, $m_{\text{num}}=3$, $m_{\text{text}}=2$.

**Grammar $G_{\mathcal L}$ (context-free skeleton).** A query is

$$
q \;=\; \underbrace{\textsf{SELECT}\;[\textsf{DISTINCT}]\;\pi}_{\text{projection}}\;
\underbrace{\textsf{FROM}\;T}_{\text{fixed}}\;
[\,\textsf{WHERE}\;\varphi\,]\;
[\,\textsf{GROUP BY}\;\gamma\;[\,\textsf{HAVING}\;\eta\,]\,]\;
[\,\textsf{ORDER BY}\;\omega\,]\;[\,\textsf{LIMIT}\;\ell\,]
$$

with clause grammars
- **Projection** $\pi$: either $\star$ (i.e. `*`) or an ordered list
  $(a_1,\dots,a_p)$, $p\ge1$, each item $a_i$ a *column term* $c_j$ or an *aggregate term*
  $f(c_j)$ / $\textsf{COUNT}(\star)$, with $f\in\mathcal F=\{\textsf{SUM},\textsf{AVG},
  \textsf{MIN},\textsf{MAX},\textsf{COUNT}\}$, optionally aliased.
- **Predicate** $\varphi$ (WHERE): a boolean tree over atoms combined by
  $\{\wedge,\vee,\neg\}$. Atoms $\alpha$: $c\;\textsf{op}\;v$ with $\textsf{op}\in\{=,>,
  \ge,<,\le\}$; $c\;\textsf{BETWEEN}\;v_1\;v_2$; $c\;\textsf{IN}(\dots)$;
  $c\;\textsf{LIKE}\;pat$; $c\;\textsf{IS [NOT] NULL}$.
- **Grouping** $\gamma$: a nonempty ordered subset of $\mathcal C$.
- **Having** $\eta$: a predicate whose atoms compare aggregates $f(c)$ to constants.
- **Ordering** $\omega$: an ordered list of $(\text{key},\textsf{dir})$, key a column or a
  projection alias, $\textsf{dir}\in\{\textsf{ASC},\textsf{DESC}\}$.
- **Limit** $\ell\in\mathbb Z_{\ge0}\cup\{\bot\}$.

**Type / executability constraints $\Phi$ (context-sensitive — the crux).**
A skeleton+binding is *executable* iff:
- (T1) numeric ops $\{>,\ge,<,\le\}$ and $\textsf{SUM},\textsf{AVG}$ apply only to
  $\textsf{num}$ columns; $\textsf{LIKE}$ only to $\textsf{text}$.
- (T2) **GROUP BY consistency:** if $\gamma$ present, every non-aggregate projection term
  $c_j\in\pi$ satisfies $c_j\in\gamma$. If any aggregate appears in $\pi$ with no $\gamma$,
  then *all* terms in $\pi$ are aggregates (single implicit group).
- (T3) $\textsf{HAVING}$ present $\Rightarrow$ aggregate context ($\gamma$ present or
  all-aggregate projection).
- (T4) ORDER BY keys are in scope (a projection alias or a column of $T$).

These four are precisely what a generic graph/grammar prior does **not** enforce; pushing
$\Phi$ onto the support is where the contribution lives (Sec. 5, 8).

---

## 2. The query graph object

Parse $q$ to a typed directed graph $g(q)=(V,E)$ (implemented in
`src/bnp_nl2sql/query_graph.py`): nodes typed in $\{\textsf{query},\textsf{table},
\textsf{column},\textsf{literal},\textsf{function},\textsf{operator},\textsf{clause}\}$,
edges typed by role $\in\{\textsf{select},\textsf{from},\textsf{clause},\textsf{cond},
\textsf{key},\textsf{arg},\textsf{operand}\}$. We work modulo a **canonicalization**
$\kappa$ that quotients out (i) alias names, (ii) commutative reordering of $\wedge,\vee,=$,
(iii) literal *values* (kept as typed placeholders $\textsf{num}/\textsf{text}$). Write
$[q]=\kappa(g(q))$ for the canonical structure. The prior lives on $\{[q]\}$, **not** on
surface strings — two prompts that differ only in a constant map to the same object.

---

## 3. The space $\mathcal Q(S)$ — the "full theoretical graph"

Let $\mathcal Q(S)$ be the set of canonical structures executable against schema $S$.

**Proposition 1 (characterization).** *Modulo literal values, $\mathcal Q(S)$ is the set of
tuples $(\pi,\varphi,\gamma,\eta,\omega,\ell)$ admissible under $G_{\mathcal L}$ and
satisfying $\Phi$. It is **countable**; it is **finite** once the WHERE/HAVING predicate
trees are bounded in depth $D$ and width (fan-in) $W$.*

*Sketch.* Every clause but $\varphi,\eta$ ranges over a finite set given $S$
(subsets/sequences of $\mathcal C$, of $\mathcal F\times\mathcal C$, of directions). The
only source of countable infinity is unbounded boolean nesting in $\varphi,\eta$; bounding
$(D,W)$ truncates each to a finite set. $\square$

This is the sense in which we can "construct the full theoretical graph": $\mathcal Q(S)$
is an explicit, enumerable support (see `scripts/enumerate_space.py` for a bounded count on
the airbnb schema). Three structural quantities organize it:

- **Skeleton** $s=\sigma([q])$: the structure with columns/values/aggregated-vs-not slots
  abstracted to typed placeholders (which clauses present, predicate-tree *shape*,
  projection arity, group/order arity). $\mathcal S$ = set of skeletons; finite under
  $(D,W)$, **schema-independent**.
- **Binding** $b$: the assignment filling a skeleton's typed slots with concrete columns /
  ops / value-placeholders from $S$. $\mathcal B(s,S)$ = admissible bindings (those passing
  $\Phi$).
- Assembly map $\textsf{asm}:(s,b)\mapsto[q]$ is a bijection onto $\mathcal Q(S)$ when
  restricted to $\Phi$-valid pairs. **This $(s,b)$ factorization drives Approach 2 and the
  uncertainty decomposition (Sec. 7).**

$\mathcal Q(S)$ carries a natural **partial order** $\preceq$ (refinement: add a clause,
conjoin a predicate atom, lengthen projection), with $\textsf{SELECT}\;\star\;\textsf{FROM}\;
T$ as bottom. This lattice structure is what Approach 4 exploits.

---

## 4. What we want from a prior $P$ on $\mathcal Q(S)$

Four desiderata (from the verified prior survey):
(a) **valid support:** $\operatorname{supp}(P)\subseteq\mathcal Q(S)$ — no mass on
non-executable queries; (b) **inference tractable** with the LLM as proposal;
(c) **interpretable** uncertainty; (d) **localizable** to query components. A fifth,
specific to BNP and central to our pitch: (e) **open-world mass** — $P$ assigns positive
probability to structures *not yet observed* among model samples (discovery probability),
giving a principled "the right query may be something the model never proposed" signal.

---

## 5. Four candidate priors

### Approach 1 — Pitman–Yor adaptor grammar (PYAG) over $G_{\mathcal L}$

For each adapted nonterminal $A$ (e.g. `Predicate`, `ProjItem`, `Query`) put a Pitman–Yor
adaptor $C_A\sim\mathrm{PY}(d_A,\theta_A,G_A)$ where the base $G_A$ is the PCFG expansion of
$A$ using the *adapted* child distributions. Generation top-down: at $A$ with $K_A$ cached
subtrees of counts $n_{A,1\!:\!K_A}$ ($n_A=\sum_k n_{A,k}$),
$$
\Pr(\text{reuse cached subtree }t_k)=\frac{n_{A,k}-d_A}{\theta_A+n_A},\qquad
\Pr(\text{fresh from }G_A)=\frac{\theta_A+d_A K_A}{\theta_A+n_A}.
$$
**Support invariance** (Johnson–Griffiths–Goldwater 2006): $\operatorname{supp}(C_A)=
\operatorname{supp}(G_A)$, so a PYAG over $G_{\mathcal L}$ places zero mass off
*context-free*-valid queries — desideratum (a) for the CF part. The context-sensitive
$\Phi$ (T1–T4) is imposed by making $G_{\mathcal L}$ a **typed/attribute grammar**:
annotate each nonterminal with inherited attributes (in-scope columns and their types,
group-by set) and let productions fire only when attributes satisfy $\Phi$. Then
$\operatorname{supp}=\mathcal Q(S)$ exactly.
*Pros:* general (extends to joins later), whole-subtree caching gives fragment-level
reuse (d,e). *Cons:* attribute grammar bookkeeping; inference is MH over adaptor seatings.

### Approach 2 — Hierarchical skeleton+binding model **(recommended for $\mathcal L$)**

Exploit the $(s,b)$ factorization of Sec. 3. Two levels:

1. **Structural prior (nonparametric):** a Pitman–Yor process over **skeletons**,
   $$
   s_1,s_2,\dots\mid C\stackrel{iid}{\sim}C,\qquad C\sim\mathrm{PY}(d,\theta,H),
   $$
   with base $H$ a branching process over $\mathcal S$ (a PCFG generating skeletons under
   $(D,W)$ bounds). The induced partition of repeated skeletons follows the two-parameter
   EPPF
   $$
   p(n_1,\dots,n_K)=\frac{(\theta+d)_{(K-1)\uparrow d}}{(\theta+1)_{n-1}}\prod_{k=1}^{K}(1-d)_{n_k-1}.
   $$
   $H$ being schema-independent means structural priors transfer across databases.
2. **Binding prior (parametric, schema-conditional):** given $s$, fill its typed slots
   independently,
   $$
   p(b\mid s,S)=\Big[\prod_{\text{slot }r\in s} \mathrm{Cat}\big(b_r\mid \beta_{r}(S)\big)\Big]\,
   \mathbf 1[\Phi(s,b)],
   $$
   where $\beta_r(S)$ ranges over the type-admissible fillers of slot $r$ (e.g. a
   numeric-comparison slot draws a column from the $m_{\text{num}}$ numeric columns). The
   indicator $\mathbf 1[\Phi]$ enforces cross-slot rules (T2 group-by consistency etc.);
   its normalizer is a finite sum over $\mathcal B(s,S)$.

**Joint:** $P([q]) = \mathbb E_{C}\!\big[C(\sigma([q]))\big]\;p(b_{[q]}\mid\sigma([q]),S)$.
*Pros:* (a) exact valid support; (c,d) the cleanest interpretability — structural vs.
binding uncertainty separate and each binding slot is a named component; (e) PY "new
skeleton" mass is explicit; most tractable here because bindings factorize. *Cons:* the
clause-factored base is bespoke to the join-free fragment (which is exactly our scope).

### Approach 3 — Feature-allocation (IBP / beta–Bernoulli) over query atoms

Enumerate the finite set of **atoms** $\mathcal A(S)$: each (column-in-projection),
(aggregate-term), (predicate-atom up to value), (group-by-column), (order-key). A query is
a subset $Z\subseteq\mathcal A(S)$. Put a beta–Bernoulli / Indian Buffet prior on the
inclusion vector, $\pi_a\sim\mathrm{Beta}(\tfrac{\alpha}{|\mathcal A|},1)$,
$z_a\mid\pi_a\sim\mathrm{Bern}(\pi_a)$, then **assemble** $Z$ into a query and reject if
$\Phi$ fails. *Pros:* directly models the *subset* combinatorics of SELECT/GROUP BY; sparse,
unbounded atom vocabulary (IBP). *Cons:* assembly+rejection means support is only
implicitly $\mathcal Q(S)$ (mass wasted on invalid subsets — violates (a) unless rejection
is folded in); ignores ordering and predicate nesting. Best as a **secondary** prior on the
projection/grouping atoms, composed under Approach 2's skeleton.

### Approach 4 — Constrained construction on the query lattice (masking / GFlowNet)

Build $[q]$ sequentially along $\preceq$ from $\bot$, at each step choosing a refinement or
$\textsf{stop}$, **masking** moves that would violate $\Phi$. A nonparametric flavor comes
from a PY-type stopping rule (rich-get-richer over previously constructed prefixes).
Support is **exactly** $\mathcal Q(S)$ by construction (mask = hard constraint), and the
sampler is amortizable to LLM scale (cf. DAG-GFlowNet). *Pros:* (a) exact, (b) scalable.
*Cons:* the induced $P$ is implicit (no closed-form density); calibration must be checked
empirically. Natural **engine** for Approaches 1–2 rather than a competing prior.

---

## 6. Inference: the LLM-wrapper posterior (Model A — committed for the paper)

**Decision.** The paper uses **Model A**: wrap a black-box LLM, treat its $N$ temperature
samples as the data, and form a *conjugate* component-wise posterior under the BNP prior. No
gold label is needed at test time; no model is trained. (The richer corruption-kernel model
and the fully-trained structured predictor are Sec. 9 future work.)

**Setup.** For NL question $x$ the LLM emits samples $q_{1:N}$; parse each to
$[q_i]=(s_i,b_i)$ with skeleton $s_i$ and slot-fillers $b_i=\{b_{i,r}\}_{r\in\text{slots}(s_i)}$
(`skeleton_key` / binding extractor in `query_graph.py`). We model the LLM as defining a
*latent per-component answer distribution* for $x$, observe $N$ draws from it, and put the
BNP prior on those latent distributions. Updates are then pure conjugate counting on graph
components — the prior contributes shrinkage and open-world mass that raw counting cannot.

**(1) Structural level — Pitman–Yor urn over skeletons.** With $G\sim\mathrm{PY}(d,\theta,H)$
and $s_{1:N}\mid G\stackrel{iid}{\sim}G$, the posterior predictive over skeletons is the
two-parameter urn: a skeleton seen $n_s$ times gets
$$
\widehat P(s\mid q_{1:N})=\frac{n_s-d}{\theta+N}+\frac{\theta+dK}{\theta+N}\,H(s),\qquad
\underbrace{\Pr(\text{unseen skeleton})=\frac{\theta+dK}{\theta+N}\big(1-\!\!\sum_{\text{seen }s}\!\!H(s)\big)}_{\text{discovery probability}},
$$
$K=$ #distinct skeletons observed. The discovery term is the closed-form answer to *"what is
the probability the correct query has a structure the model never proposed?"* — a
Good–Turing / unseen-species quantity the frequency baseline in `uncertainty.py`
structurally **cannot** produce (it assigns 0 to the unseen). **Sharpest contribution hook.**

**(2) Binding level — Dirichlet–multinomial per slot.** Within the samples sharing skeleton
$s$ (so the slot set is defined), each slot $r$ has filler distribution
$\beta_r\sim\mathrm{Dir}(\alpha_r u_r)$; with counts $c_{r,v}$ over the $n_s$ such samples,
$$
\beta_r\mid q_{1:N}\sim\mathrm{Dir}(\alpha_r u_r+c_r),\qquad
\widehat P(b_r=v)=\frac{\alpha_r u_{r,v}+c_{r,v}}{\alpha_r+n_s}.
$$
Minority skeletons have small $n_s$ ⇒ correctly wider binding posteriors. For open-ended
slots (literal values, rare columns) swap $\mathrm{Dir}$ for a PY slot to keep open-world mass.

**(3) Joint posterior predictive over query structures**, renormalized onto the executable
manifold (computable because $\mathcal Q(S)$ is enumerable, Prop. 1; or sampled via the
Approach-4 masking engine):
$$
\widehat P([q]\mid q_{1:N})\;\propto\;\widehat P(s\mid q_{1:N})\;\Big[\textstyle\prod_{r\in\text{slots}(s)}\widehat P(b_r\mid q_{1:N})\Big]\;\mathbf 1[\Phi(s,b)].
$$

**Readouts / abstention.** Point prediction $=\arg\max$ joint (the MAP query); confidence
signals = structural mass on the MAP skeleton, the **discovery probability**, and per-slot
posterior entropies (localized). **Abstain** when MAP structural mass $<\tau_1$, or discovery
probability $>\tau_2$, or any critical-slot entropy $>\tau_3$.

**Where training data enters (prior fitting + calibration, offline).** A training set of
$(x,\text{gold query})$ pairs is used *only* to (i) empirical-Bayes the base measures $H,u_r$
and PY/Dirichlet hyperparameters $(d,\theta,\alpha_r)$ by seating gold skeletons/components
(rich-get-richer over real query shapes), and (ii) **conformally calibrate** the thresholds
$\tau_{1:3}$ to control selective risk at a target level (à la RTS). This is what converts
the posterior from "model self-agreement" into a *correctness*-calibrated signal — necessary
because fragment reuse $\ne$ execution correctness.

---

## 7. Uncertainty decomposition and localization

Under Approach 2 the entropy of the posterior predictive factors by the chain rule:
$$
\underbrace{\mathbb H(Q)}_{\text{total}}
=\underbrace{\mathbb H(S_{\text{skel}})}_{\text{structural}}
+\underbrace{\mathbb E_{s}\,\mathbb H(B\mid s)}_{\text{binding}},\qquad
\mathbb H(B\mid s)=\sum_{\text{slot }r}\mathbb H(B_r\mid s)\ \text{(slots indep.\ given }s).
$$
So uncertainty is reported as: a structural bit-count (does the model agree on the query
*shape*?) plus a **per-slot** bit-count (which column? which aggregate? which op?). This is
the calibrated version of the `component_disagreement` readout already demoed, and is
exactly the localization no scalar/sequence baseline (prob-rescaling, semantic entropy)
offers.

---

## 8. Recommendation, theorems to target, and next steps

**Lead with Approach 2** (hierarchical skeleton+binding PY) for the paper's restricted
fragment: it is the only candidate that is simultaneously exact-support (a), tractable (b)
via factorized bindings, and cleanly decomposable (c,d), and it makes the open-world signal
(e) explicit. **Frame Approach 1 (PYAG)** as the principled generalization (the route to
joins/subqueries beyond this paper). **Use Approach 4 (masking)** as the sampler that
guarantees $\Phi$. **Reserve Approach 3 (IBP)** as an optional sub-prior on projection
atoms.

*Targets to prove:*
- **T-A (support).** With the typed/attribute base, $\operatorname{supp}(P)=\mathcal Q(S)$
  (no mass off-manifold). [Direct from attribute-grammar firing + $\Phi$ indicator.]
- **T-B (exchangeable consistency).** The skeleton partition is exchangeable with EPPF as
  in §5.2 ⇒ a well-defined random measure (de Finetti via Pitman–Yor). [Standard.]
- **T-C (calibration / discovery).** The discovery-probability estimator $\frac{\theta+dK}{
  \theta+N}$ is a consistent estimate of missing structural mass; relate to selective-risk
  guarantees à la conformal RTS. [The empirical contribution — must be benchmarked, since
  fragment *reuse* $\ne$ execution *correctness*.]

*Immediate next code steps (theory-linked):*
1. `schema.py` + `grammar.py`: encode $S$ (airbnb) and $G_{\mathcal L}$; emit skeletons.
2. `scripts/enumerate_space.py`: enumerate $\mathcal Q(S)$ under bounds $(D,W)$ to put a
   concrete number on Prop. 1 (the "full theoretical graph").
3. `pyp.py`: Pitman–Yor restaurant (seating, EPPF, predictive, discovery prob).
4. `adaptor2.py`: Approach-2 posterior from $N$ parsed samples → structural+binding
   uncertainty + discovery probability; drop-in successor to `uncertainty.py`.

*Open empirical question (do not assume):* does low posterior uncertainty actually predict
execution correctness? Reuse $\ne$ correctness; this is the experiment that makes or breaks
the contribution.

## 9. Future work (beyond this paper)

The paper deliberately scopes to Model A on the join-free fragment to get a fundable
proof-of-concept out. The natural extensions, roughly in order of ambition:

1. **Corruption-kernel observation model.** Replace the iid-draws assumption with a
   component-wise noise kernel ($\epsilon$ per slot / predicate atom) so the $N$ samples are
   explicitly noisy realizations of one latent $Q^\star$; learn $\epsilon$. More expressive
   uncertainty, no longer pure counting.
2. **Fully-trained structured predictor.** Drop the LLM wrapper: train a structured model on
   $(x,\text{gold-graph})$ pairs with the BNP prior as regularizer; posterior-predictive over
   graphs is the UQ. This is the *Bayesian model of NL2SQL* (and the "big training job" the
   PoC is meant to fund).
3. **Joins & subqueries** via the PYAG (Approach 1) with a typed/attribute base grammar —
   reintroduces the context-sensitive type/key machinery dropped here.
4. **Question-tilted prior.** Fold LLM token log-probs into $H_x\propto H\exp\{\lambda\,
   \textsf{score}_{\text{LLM}}\}$ for a white-box variant.
5. **Actuarial framing.** Cast abstention as a loss/risk problem (cost of a wrong query vs.
   cost of asking a human) — the selective-prediction decision theory the discovery
   probability feeds.
