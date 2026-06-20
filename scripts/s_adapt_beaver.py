"""Family 3 (adaptation): online Bayesian updating on BEAVER (correlated, repeated workload).

Prequential (test-then-train) learning-curve simulation: stream the dw queries; for each, retrieve
with the CURRENT model (trained on prior queries only), record recall, then update from the gold
feedback. Does retrieval IMPROVE as feedback accrues, and beat static cosine? This is the genuine
Bayes-adaptation test (distinct from ranking/decision/UQ) and the user's "learn as data comes in" idea.

Strengthened for TAS review:
  - models: static cosine (control); online MLE counts (NON-Bayesian: no prior smoothing);
    online popularity (Beta-Bernoulli prior); online naive-Bayes term->table (Dirichlet prior).
    The MLE-vs-Bayes contrast isolates whether the PRIOR/smoothing matters or just online counting.
  - R=20 random stream orders; report mean +/- sd and PAIRED diffs vs cosine across orders (CI).
  - prequential learning curve by quartile.
  - feedback-rate sweep (100/50/25% of queries give feedback) and NOISY feedback robustness.
No API (cached embeddings).   ./.venv/bin/python scripts/s_adapt_beaver.py
"""
from __future__ import annotations
import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB = os.path.join(ROOT, "data", "beaver_emb.json")


def toks(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > 1]


def z(x):
    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-9)


def main():
    tabs = json.load(open(os.path.join(ROOT, "data", "beaver", "dev_tables.json")))
    dwq = json.load(open(os.path.join(ROOT, "data", "beaver", "dev_dw.json")))
    cache = json.load(open(EMB))
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    dw = [k for k, v in tabs.items() if v["db_id"] == "dw"]
    low2key = {tabs[k]["table_name_original"].lower(): k for k in dw}
    text = {k: tabs[k]["table_name_original"] + ": " + ", ".join(tabs[k]["column_names_original"]) for k in dw}
    n = len(dw); TV = {k: vec(text[k]) for k in dw}
    items = []
    for q in dwq:
        gold = {low2key[g.replace("dw#sep#", "").lower()] for g in q["gold_tables"] if g.replace("dw#sep#", "").lower() in low2key}
        if len(gold) >= 2 and q["question"] in cache:
            items.append((q["question"], gold))
    Qc = []
    for q, gold in items:
        qv = vec(q); cos = np.array([float(qv @ TV[k]) for k in dw]); Qc.append((cos, set(toks(q)), [i for i in range(n) if dw[i] in gold]))
    print(f"S-adapt BEAVER dw: {n} tables, {len(items)} queries (mean gold {np.mean([len(g) for *_,g in Qc]):.1f})")

    LAM = 1.0
    def run_stream(seed, fb_rate=1.0, noise=0.0, models=("cosine", "MLE", "popularity", "naiveBayes")):
        rng = np.random.RandomState(seed); order = rng.permutation(len(items))
        gc = np.zeros(n); seen = 0                       # popularity / MLE counts
        wt = defaultdict(lambda: np.zeros(n)); wtot = Counter()
        # MLE (non-Bayesian) term->table counts share wt/wtot but score without smoothing
        recs = {m: [] for m in models}; bypos = {m: [[] for _ in range(4)] for m in models}
        for pos, qi in enumerate(order):
            cos, qterms, gidx = Qc[qi]; kk = len(gidx)
            pop_b = np.log((gc + 0.5) / (seen + 1.0))                       # Beta-Bernoulli
            pop_m = np.log((gc + 1e-9) / (seen + 1e-9) + 1e-9)             # MLE rate
            nb_b = np.zeros(n); nb_m = np.zeros(n)
            for w in qterms:
                if w in wtot:
                    nb_b += np.log((wt[w] + 0.1) / (wtot[w] + 0.1 * n))     # Dirichlet smoothing
                    nb_m += np.log((wt[w] + 1e-9) / (wtot[w] + 1e-9) + 1e-12)  # MLE (no prior)
            preds = {
                "cosine": cos,
                "MLE": z(cos) + (LAM * z(nb_m) if seen > 5 else 0),
                "popularity": z(cos) + LAM * z(pop_b),
                "naiveBayes": z(cos) + (LAM * z(nb_b) if seen > 5 else 0),
            }
            qb = min(3, int(pos / len(items) * 4))
            for m in models:
                top = set(np.argsort(-preds[m])[:kk]); r = len(set(gidx) & top) / kk
                recs[m].append(r); bypos[m][qb].append(r)
            # update (with feedback rate + noise)
            if rng.rand() < fb_rate:
                fb = list(gidx)
                if noise > 0 and rng.rand() < noise and len(fb) > 1:
                    fb = fb[:-1] + [int(rng.randint(n))]   # corrupt one label
                for i in fb:
                    gc[i] += 1
                for w in qterms:
                    wtot[w] += 1
                    for i in fb:
                        wt[w][i] += 1
                seen += 1
        return {m: np.mean(recs[m]) for m in models}, {m: [np.mean(bypos[m][b]) for b in range(4)] for m in models}

    # ---- main: R orders, mean +/- sd, paired diffs vs cosine ----
    R = 20
    per_order = {m: [] for m in ("cosine", "MLE", "popularity", "naiveBayes")}
    curves = {m: np.zeros(4) for m in per_order}
    for s in range(R):
        allm, byq = run_stream(s)
        for m in per_order:
            per_order[m].append(allm[m]); curves[m] += np.array(byq[m]) / R
    print(f"\n[A] All-stream recall@|gold| over {R} orders (mean +/- sd); paired diff vs cosine [95% CI over orders]:")
    co = np.array(per_order["cosine"])
    for m in ("cosine", "MLE", "popularity", "naiveBayes"):
        v = np.array(per_order[m]); d = v - co
        ci = "" if m == "cosine" else f"   diff {d.mean():+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]"
        print(f"    {m:<12} {v.mean():.3f} +/- {v.std():.3f}{ci}")
    print(f"\n[B] Prequential learning curve (recall by quartile, mean over {R} orders):")
    print(f"    {'model':<12}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'  Q4-Q1':>9}")
    for m in ("cosine", "MLE", "popularity", "naiveBayes"):
        c = curves[m]; print(f"    {m:<12}" + "".join(f"{x:>8.3f}" for x in c) + f"{c[-1]-c[0]:>+9.3f}")

    # ---- feedback-rate + noise robustness (naiveBayes) ----
    print("\n[C] naive-Bayes all-stream recall under partial / noisy feedback (mean over 10 orders):")
    for label, fr, nz in (("full feedback", 1.0, 0.0), ("50% feedback", 0.5, 0.0),
                          ("25% feedback", 0.25, 0.0), ("20% label noise", 1.0, 0.2)):
        vals = [run_stream(s, fb_rate=fr, noise=nz, models=("naiveBayes",))[0]["naiveBayes"] for s in range(10)]
        print(f"    {label:<18}{np.mean(vals):.3f} +/- {np.std(vals):.3f}")
    print("\nReading: Bayesian smoothing (Dirichlet/Beta) vs MLE counts isolates whether the PRIOR matters")
    print("or just online counting; CIs over orders + partial/noisy-feedback robustness harden the claim.")


if __name__ == "__main__":
    main()
