"""Equivalence-class re-analysis of the open-world motif tail (gate for the BNP novelty story).

Probe 2 found a heavy skeleton-level tail (PYP d=0.16, ~52% discovery). Two threats: (i) SYNTACTIC
inflation -- trivially-equivalent queries counted as distinct motifs; (ii) it may not actually be a
power law (could be DP-like log growth from benchmark curation). This script:

  1. Canonicalization ladder: re-fit the PYP at three motif granularities
     (skeleton -> coarser semantics-preserving 'canon' -> clause-set), reporting d, theta,
     discovery, singleton%. If the tail SURVIVES coarsening it is not a pure syntactic artifact.
  2. Power-law diagnostic: species-accumulation exponent. PYP => K_n ~ C * n^d (power law);
     a closed/DP world => K_n ~ theta*log n (saturates). Fit log K_n vs log n and report the
     growth exponent + R^2, compared to the partition-MLE d.
  3. Denotational inflation factor (execution-grounded): across cached model samples, distinct
     SKELETONS per distinct RESULT-SET within a question -- an empirical upper bound on how much
     skeleton-counting overcounts denotationally-equivalent variants.

No API.  ./.venv/bin/python scripts/bnp_equivclass.py
"""
from __future__ import annotations
import json, math, os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bnp_nl2sql.query_graph import sql_to_graph
from bnp_nl2sql.posterior import extract_slots
sys.path.insert(0, os.path.dirname(__file__))
from bird_error_analysis import features
from bnp_nl2sql.execeval import open_db, run_sql

ROOT = os.path.join(os.path.dirname(__file__), "..")


def pyp_mle(sizes):
    sizes = [int(x) for x in sizes]; N = sum(sizes); K = len(sizes)
    def ll(d, th):
        if d < 0 or d >= 1 or th <= -d:
            return -1e18
        v = sum(math.log(th + i * d) for i in range(1, K)) - sum(math.log(th + i) for i in range(1, N))
        for nk in sizes:
            v += sum(math.log(j - d) for j in range(1, nk))
        return v
    best = (-1e18, 0.0, 1.0)
    for d in [i / 100 for i in range(100)]:
        for th in [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]:
            if th > -d and ll(d, th) > best[0]:
                best = (ll(d, th), d, th)
    _, d, th0 = best
    for th in np.linspace(max(0.01, th0 / 3), th0 * 3, 80):
        if ll(d, th) > best[0]:
            best = (ll(d, th), d, th)
    return best[1], best[2]


def accumulation_exponent(labels, seed=0, pts=8):
    """Fit log K_n = a + b log n over the accumulation curve (avg of a few shuffles)."""
    rng = np.random.RandomState(seed); N = len(labels)
    ns = np.unique(np.linspace(N // 16, N, pts).astype(int))
    Ks = np.zeros(len(ns), float)
    for _ in range(5):
        order = rng.permutation(N)
        seq = [labels[i] for i in order]
        for i, n in enumerate(ns):
            Ks[i] += len(set(seq[:n]))
    Ks /= 5
    b, a = np.polyfit(np.log(ns), np.log(Ks), 1)
    pred = a + b * np.log(ns)
    ss = 1 - np.sum((np.log(Ks) - pred) ** 2) / np.sum((np.log(Ks) - np.log(Ks).mean()) ** 2)
    return b, ss, list(zip(ns.tolist(), np.round(Ks, 1).tolist()))


# ----- motif definitions (the canonicalization ladder) -----
def skeleton_motif(sql):
    try:
        return sql_to_graph(sql, dialect="sqlite").skeleton_key()
    except Exception:
        return None


def canon_motif(sql):
    """Coarser, semantics-preserving-ish: collapse predicate multiplicity/columns/operator identity,
    keep clause structure + agg identity + join-count bucket + op-CLASS set + subquery depth."""
    f = features(sql)
    if f is None:
        return None
    try:
        s = extract_slots(sql, dialect="sqlite")
        ops = frozenset(s["where_ops"])              # operator presence as a set (not count/order)
        aggs = frozenset(s["agg_functions"])
    except Exception:
        ops, aggs = frozenset(), frozenset()
    njoin = min(f["_n_join"], 3)
    return (njoin, aggs, ops, f["group_by"], f["having"], f["order_by"], f["limit"],
            f["distinct"], f["subquery"], f["set_op"], f["math"], f["case"])


def clause_motif(sql):
    f = features(sql)
    if f is None:
        return None
    return (f["aggregate"], f["group_by"], f["order_by"], f["having"], f["limit"],
            f["distinct"], bool(f["_n_join"]), f["subquery"])


def report(name, labels):
    labels = [l for l in labels if l is not None]
    N = len(labels); cnt = Counter(labels); sizes = sorted(cnt.values(), reverse=True)
    K = len(cnt); singles = sum(1 for v in sizes if v == 1)
    d, th = pyp_mle(sizes); disc = (th + K * d) / (th + N)
    b, r2, curve = accumulation_exponent(labels)
    print(f"--- {name}: N={N}, K={K} motifs ---")
    print(f"  singletons {singles} ({singles/K:.0%} of motifs, {singles/N:.0%} of queries); top-1 {sizes[0]/N:.1%}")
    print(f"  PYP MLE: d={d:.3f}, theta={th:.1f}, discovery={disc:.3f}")
    print(f"  accumulation K_n ~ n^b: b={b:.3f} (R^2={r2:.3f})  [PYP: b~=d power-law; DP: b->0, log]")
    print(f"  curve (n,K): {curve}")
    pl = "POWER-LAW (survives)" if (d > 0.08 and b > 0.08 and r2 > 0.95) else "weak/DP-like (artifact-prone)"
    print(f"  verdict: {pl}\n")


def load_golds():
    golds = [e["gold"] for e in json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values()]
    sp = os.path.join(ROOT, "data", "spider_samples_multi.json")
    if os.path.exists(sp):
        golds += [e["gold"] for e in json.load(open(sp)).values() if e.get("gold")]
    return golds


def denotational_inflation():
    """distinct skeletons per distinct result-set within a question (model samples, BIRD)."""
    data = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    DBDIR = os.path.join(ROOT, "data", "bird", "db")
    conns = {}
    ratios = []
    def sig(conn, sql):
        t = threading.Timer(3.0, conn.interrupt); t.start()
        try:
            r = run_sql(conn, sql); return tuple(sorted(repr(x) for x in r)) if r else ("e",)
        except Exception:
            return None
        finally:
            t.cancel()
    for e in data[:400]:
        p = os.path.join(DBDIR, f"{e['db_id']}.sqlite")
        if not os.path.exists(p):
            continue
        if p not in conns:
            conns[p] = open_db(p)
        conn = conns[p]
        sk = set(); rs = set()
        for q in e["samples"]:
            m = skeleton_motif(q)
            if m:
                sk.add(m)
            s = sig(conn, q)
            if s is not None:
                rs.add(s)
        if rs:
            ratios.append(len(sk) / len(rs))
    print(f"Denotational inflation (BIRD model samples, n={len(ratios)} questions):")
    print(f"  mean distinct-skeletons / distinct-result-sets = {np.mean(ratios):.2f}")
    print(f"  (>1 => skeleton-counting overcounts denotationally-equivalent variants; "
          f"upper bound on gold inflation)\n")


def main():
    golds = load_golds()
    print(f"=== Equivalence-class motif re-analysis (N={len(golds)} gold queries, BIRD+Spider) ===\n")
    report("skeleton (fine, probe-2 level)", [skeleton_motif(q) for q in golds])
    report("canon (coarser, ops/cols/multiplicity collapsed)", [canon_motif(q) for q in golds])
    report("clause-set (coarsest, closed-world ref.)", [clause_motif(q) for q in golds])
    denotational_inflation()
    print("Reading: if d and the accumulation exponent b stay well above 0 with high R^2 at the")
    print("'canon' level, the open-world tail survives semantics-preserving quotienting and is a real")
    print("power law -- not a syntactic artifact. If canon collapses toward clause-set, it was granularity.")


if __name__ == "__main__":
    main()
