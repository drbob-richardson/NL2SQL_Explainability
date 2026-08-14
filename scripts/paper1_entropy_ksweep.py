"""TMLR revision #9: entropy vs cluster-size self-consistency, and a K sweep, from cache. No API.

R3 asked to test semantic entropy (cited but untested) and to sweep K, to support/scope the
black-box "ceiling". We execute each of the 8 cached samples once per question (timeout-guarded),
then derive string/structural/semantic cluster distributions and, from them, both the cluster-size
(top-prob) confidence and the entropy confidence, plus a K in {2,4,6,8} sweep.

  ./.venv/bin/python scripts/paper1_entropy_ksweep.py
"""
from __future__ import annotations
import json, os, sys, threading, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bnp_nl2sql.uq_baselines import _canon, _result_key, _entropy
from bnp_nl2sql.execeval import open_db

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def safe_rkey(conn, sql, timeout=5.0):
    timer = threading.Timer(timeout, conn.interrupt); timer.start()
    try:
        return _result_key(conn, sql)
    except Exception:
        return ("__ERROR__",)
    finally:
        timer.cancel()


def topprob(cnt, k):
    return cnt.most_common(1)[0][1] / k


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))
    assert len(samp) == len(sig), (len(samp), len(sig))
    y = np.array([r["ok"] for r in sig], int)

    conns = {}
    def conn_for(db):
        if db not in conns:
            conns[db] = open_db(os.path.join(DBDIR, f"{db}.sqlite"))
        return conns[db]

    # execute each sample once; cache per-question the string/structural/result keys (all 8)
    STR, STRUCT, SEM = [], [], []
    for i, e in enumerate(samp):
        ss = e["samples"]  # samp and sig are positionally aligned (as in bird_correctness_final.py)
        STR.append(list(ss))
        STRUCT.append([_canon(s) or "<unparseable>" for s in ss])
        conn = conn_for(e["db_id"])
        SEM.append([safe_rkey(conn, s) for s in ss])
        if (i + 1) % 200 == 0:
            print(f"  executed {i+1}/{len(samp)} questions", file=sys.stderr)

    def sweep(keys_list, k):
        tp = np.array([topprob(Counter(ks[:k]), k) for ks in keys_list])
        ent = np.array([-_entropy(Counter(ks[:k]), k) for ks in keys_list])
        return tp, ent

    print(f"n={len(y)}  accuracy={y.mean():.3f}\n")
    print("=== K=8: cluster-size (top-prob) vs entropy — AUROC for correctness ===")
    for name, keys in (("string", STR), ("structural", STRUCT), ("execution/semantic", SEM)):
        tp, ent = sweep(keys, 8)
        print(f"  {name:<20} top-prob {auroc(tp, y):.3f}    entropy {auroc(ent, y):.3f}")

    print("\n=== K sweep — top-prob AUROC (label = modal correctness at full K) ===")
    print(f"  {'K':>3} {'string':>8} {'structural':>11} {'semantic':>9}")
    for k in (2, 4, 6, 8):
        st, _ = sweep(STR, k); sc, _ = sweep(STRUCT, k); se, _ = sweep(SEM, k)
        print(f"  {k:>3} {auroc(st, y):>8.3f} {auroc(sc, y):>11.3f} {auroc(se, y):>9.3f}")


if __name__ == "__main__":
    main()
