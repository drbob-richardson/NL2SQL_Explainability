"""Two cheap probes to decide whether the BNP graph-posterior core has empirical legs.

Probe 1 (marginal value): do graph-posterior features (structural entropy, modal
structural mass, node-minimum posterior) add anything to the reasoning verifier for
*correctness* prediction? We expect ~nothing (Paper 1: structural SC 0.619 < string SC
0.675; verifier+string-SC 0.754 < verifier 0.770). If confirmed, Path-1-as-correctness
is cleanly closed in writing.

Probe 2 (motif tail): cluster gold queries into structural motifs (skeleton keys + a
coarse clause-set), fit a Pitman-Yor process by ML, and inspect the tail. A heavy tail
with many singletons and d>0 grounds the open-world / discovery-probability story; a
short list says drop the nonparametric framing.

  ./.venv/bin/python scripts/bnp_probes.py
"""
from __future__ import annotations
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bnp_nl2sql.query_graph import sql_to_graph
from bnp_nl2sql.posterior import extract_slots

ROOT = os.path.join(os.path.dirname(__file__), "..")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def paired_delta(a, b, y, nb=2000):
    rng = np.random.RandomState(0); a, b, y = np.array(a), np.array(b), np.array(y); n = len(y); d = []
    for _ in range(nb):
        idx = rng.randint(0, n, n)
        if len(set(y[idx])) > 1:
            d.append(auroc(a[idx], y[idx]) - auroc(b[idx], y[idx]))
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5])


def xfit_logit(X, y, seed=0):
    """2-fold cross-fit standardized L2 logistic; returns out-of-fold scores."""
    X = np.asarray(X, float); y = np.asarray(y, float); n = len(y)
    rng = np.random.RandomState(seed); idx = rng.permutation(n); half = n // 2
    folds = [idx[:half], idx[half:]]
    oof = np.zeros(n)
    for tr, te in (folds, folds[::-1]):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w = np.zeros(Xtr.shape[1]); b = 0.0
        for _ in range(800):
            p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]
            w -= 0.3 * (Xtr.T @ g / len(tr) + 0.02 * w); b -= 0.3 * g.mean()
        oof[te] = Xte @ w + b
    return oof


def entropy(counts):
    tot = sum(counts); ps = [c / tot for c in counts if c]
    return float(-sum(p * math.log(p) for p in ps))


# --------------------------------------------------------------------------- #
# probe 1
# --------------------------------------------------------------------------- #
def graph_features(samples):
    """Structural-posterior features from the K samples of one question."""
    skels, slots_list = [], []
    for q in samples:
        try:
            skels.append(sql_to_graph(q, dialect="sqlite").skeleton_key())
        except Exception:
            pass
        try:
            slots_list.append(extract_slots(q, dialect="sqlite"))
        except Exception:
            pass
    K = len(samples)
    if skels:
        sk = Counter(skels)
        skel_entropy = entropy(list(sk.values()))
        struct_modal_mass = sk.most_common(1)[0][1] / len(skels)
        n_distinct = len(sk) / len(skels)
    else:
        skel_entropy, struct_modal_mass, n_distinct = 0.0, 1.0, 1.0
    # node-minimum posterior: per slot, agreement of samples with the modal value
    node_min = 1.0
    if slots_list:
        keys = slots_list[0].keys()
        agreements = []
        for k in keys:
            vals = Counter(str(s.get(k)) for s in slots_list)
            agreements.append(vals.most_common(1)[0][1] / len(slots_list))
        node_min = min(agreements) if agreements else 1.0
    return [skel_entropy, struct_modal_mass, n_distinct, node_min]


def probe1():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))
    y, v4o, vmini, ssc, G = [], [], [], [], []
    for e, s in zip(samp, sig):
        y.append(int(bool(s["ok"]))); v4o.append(s["v4o"]); vmini.append(s["vmini"]); ssc.append(s["top"])
        G.append(graph_features(e["samples"]))
    y = np.array(y); G = np.array(G); n = len(y)
    print(f"=== PROBE 1: marginal value of graph-posterior features (n={n}, acc={y.mean():.3f}) ===\n")

    fnames = ["skeleton_entropy", "struct_modal_mass", "n_distinct_skel", "node_min_post"]
    print("Graph-posterior features ALONE (single-signal AUROC for correctness):")
    for j, fn in enumerate(fnames):
        # orient each feature so higher = more likely correct via sign of correlation
        col = G[:, j]
        a = auroc(col, y); a = max(a, 1 - a)
        print(f"  {fn:<20} {a:.3f}")
    print(f"  {'string self-consist.':<20} {auroc(ssc, y):.3f}   (Paper 1 ceiling)")
    print()

    base = auroc(v4o, y)
    post_oof = xfit_logit(G, y)
    comb_oof = xfit_logit(np.column_stack([v4o, G]), y)
    sscomb_oof = xfit_logit(np.column_stack([v4o, ssc]), y)
    print("Combined models (cross-fit logistic, out-of-fold AUROC):")
    print(f"  verifier (GPT-4o) alone           {base:.3f}")
    print(f"  graph-posterior features alone     {auroc(post_oof, y):.3f}")
    print(f"  verifier + string self-consistency {auroc(sscomb_oof, y):.3f}")
    print(f"  verifier + graph-posterior feats   {auroc(comb_oof, y):.3f}")
    m, (lo, hi) = paired_delta(comb_oof, v4o, y)
    print(f"\n  paired Δ (verifier+graph) − verifier: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("  Reading: CI overlapping 0 => the graph posterior adds nothing to the verifier for correctness.\n")


# --------------------------------------------------------------------------- #
# probe 2
# --------------------------------------------------------------------------- #
def clause_set(sql):
    try:
        s = extract_slots(sql, dialect="sqlite")
    except Exception:
        return None
    try:
        g = sql_to_graph(sql, dialect="sqlite"); nt = g.node_type_counts()
    except Exception:
        nt = {}
    return (
        bool(s["agg_functions"]), bool(s["group_columns"]), bool(s["order_keys"]),
        bool(s["has_having"]), bool(s["has_limit"]), s["projection"][1],  # distinct
        bool(s["filter_columns"]), nt.get("join", 0) > 0 or nt.get("table", 0) > 1,
    )


def pyp_mle(sizes):
    """ML estimate of Pitman-Yor (d, theta) from cluster sizes via grid + refine."""
    sizes = [int(x) for x in sizes]; N = sum(sizes); K = len(sizes)
    def loglik(d, th):
        if d < 0 or d >= 1 or th <= -d:
            return -1e18
        ll = sum(math.log(th + i * d) for i in range(1, K))
        ll -= sum(math.log(th + i) for i in range(1, N))
        for nk in sizes:
            ll += sum(math.log(j - d) for j in range(1, nk))  # (1-d)_{nk-1}
        return ll
    best = (-1e18, 0.0, 1.0)
    for d in [i / 100 for i in range(0, 100)]:
        for th in [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]:
            if th <= -d:
                continue
            ll = loglik(d, th)
            if ll > best[0]:
                best = (ll, d, th)
    # refine theta around best on a finer grid
    _, d, th0 = best
    for th in np.linspace(max(0.01, th0 / 3), th0 * 3, 60):
        ll = loglik(d, th)
        if ll > best[0]:
            best = (ll, d, th)
    return best[1], best[2]


def motif_report(name, motifs):
    motifs = [m for m in motifs if m is not None]
    N = len(motifs); cnt = Counter(motifs); K = len(cnt)
    sizes = sorted(cnt.values(), reverse=True)
    singles = sum(1 for v in sizes if v == 1)
    top1 = sizes[0] / N; top5 = sum(sizes[:5]) / N
    d, th = pyp_mle(sizes)
    disc = (th + K * d) / (th + N)
    print(f"--- {name}: N={N} queries, K={K} motifs ---")
    print(f"  singletons: {singles} ({singles/K:.0%} of motifs, {singles/N:.1%} of queries)")
    print(f"  coverage: top-1 {top1:.1%}, top-5 {top5:.1%}")
    print(f"  Pitman-Yor MLE: d={d:.3f}, theta={th:.1f}")
    print(f"  next-query discovery probability (theta+Kd)/(theta+N) = {disc:.3f}")
    print(f"  tail: {'HEAVY (d>0, power-law)' if d > 0.1 else 'light (near-Dirichlet)'}\n")


def probe2():
    print("=== PROBE 2: motif tail of gold queries (open-world story) ===\n")
    golds = [e["gold"] for e in json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values()]
    sp = os.path.join(ROOT, "data", "spider_samples_multi.json")
    if os.path.exists(sp):
        golds += [e["gold"] for e in json.load(open(sp)).values() if e.get("gold")]
    skels, csets = [], []
    for q in golds:
        try:
            skels.append(sql_to_graph(q, dialect="sqlite").skeleton_key())
        except Exception:
            pass
        csets.append(clause_set(q))
    motif_report("skeleton key (fine structural motif)", skels)
    motif_report("clause-set (coarse motif: 8 binary clause flags)", csets)


if __name__ == "__main__":
    probe1()
    probe2()
