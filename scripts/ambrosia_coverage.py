"""Probe 4b analysis: does the model's K-sample posterior SURFACE the ambiguity?

For each AMBROSIA question with cached samples (scripts/ambrosia_generate.py), execute the
K samples and ask:
  - ambiguous Qs: do the samples cover BOTH valid interpretations, only ONE, or NEITHER?
    (coverage-both is the precondition for detecting ambiguity from the posterior alone;
     if the posterior collapses to one interpretation, detection needs a prior/flag.)
  - detection: does result-set DIVERGENCE among the samples separate ambiguous questions
    from unambiguous controls? (AUROC of #distinct-result-clusters predicting is_ambiguous.)

  ./.venv/bin/python scripts/ambrosia_coverage.py
"""
from __future__ import annotations
import ast, csv, json, math, os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter, defaultdict
import numpy as np
csv.field_size_limit(10**7)
from bnp_nl2sql.execeval import open_db, run_sql

ROOT = os.path.join(os.path.dirname(__file__), "..")
ADIR = os.path.join(ROOT, "data", "ambrosia")
CACHE = os.path.join(ROOT, "data", "ambrosia_samples.json")


def result_sig(conn, sql, timeout=4.0):
    timer = threading.Timer(timeout, conn.interrupt); timer.start()
    try:
        rows = run_sql(conn, sql)
        return tuple(sorted(repr(r) for r in rows)) if rows else ("<empty>",)
    except Exception:
        return None
    finally:
        timer.cancel()


def db_path(db_file):
    rel = db_file[5:] if db_file.startswith("data/") else db_file
    return os.path.join(ADIR, rel)


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    cache = json.load(open(CACHE))
    rows = list(csv.DictReader(open(os.path.join(ADIR, "ambrosia.csv"))))
    meta = {f"{r['db_file']}||{r['question']}": r for r in rows}

    conns = {}
    def conn_for(db_file):
        p = db_path(db_file)
        if p not in conns:
            conns[p] = open_db(p)
        return conns[p]

    cov = defaultdict(lambda: dict(both=0, one=0, none=0, n=0))   # by ambig_type
    div_amb, div_ctrl = [], []   # #distinct result clusters
    collapse_amb = collapse_ctrl = 0
    ctrl_acc = ctrl_n = 0
    for key, e in cache.items():
        r = meta.get(key)
        if r is None:
            continue
        conn = conn_for(e["db_file"])
        sigs = [result_sig(conn, s) for s in e["samples"]]
        valid = [s for s in sigs if s is not None]
        n_clusters = len(set(valid)) if valid else 0
        if e["is_ambiguous"]:
            div_amb.append(n_clusters)
            if n_clusters <= 1:
                collapse_amb += 1
            try:
                golds = [g for g in ast.literal_eval(r["ambig_queries"]) if isinstance(g, str)]
            except Exception:
                golds = []
            gsigs = [result_sig(conn, g) for g in golds[:2]]
            gset = [g for g in gsigs if g is not None]
            sampset = set(valid)
            matched = [g for g in gset if g in sampset]
            typ = r["ambig_type"]; cov[typ]["n"] += 1
            if len(gset) >= 2 and len(matched) >= 2:
                cov[typ]["both"] += 1
            elif len(matched) == 1:
                cov[typ]["one"] += 1
            else:
                cov[typ]["none"] += 1
        else:
            div_ctrl.append(n_clusters)
            if n_clusters <= 1:
                collapse_ctrl += 1
            gsig = result_sig(conn, r["gold_queries"])
            ctrl_n += 1
            if gsig is not None and gsig in set(valid):
                ctrl_acc += 1

    print("=== PROBE 4b: does the posterior surface the ambiguity? ===\n")
    tot = dict(both=0, one=0, none=0, n=0)
    print(f"  {'ambig type':<12}{'n':>5}{'both':>8}{'one':>8}{'none':>8}")
    for typ, c in sorted(cov.items()):
        for k in tot: tot[k] += c[k]
        print(f"  {typ:<12}{c['n']:>5}{c['both']/c['n']:>8.0%}{c['one']/c['n']:>8.0%}{c['none']/c['n']:>8.0%}")
    print(f"  {'ALL':<12}{tot['n']:>5}{tot['both']/tot['n']:>8.0%}{tot['one']/tot['n']:>8.0%}{tot['none']/tot['n']:>8.0%}")
    print(f"\n  coverage-both = {tot['both']/tot['n']:.0%}: the K=8 posterior contains BOTH valid")
    print(f"    interpretations (ambiguity detectable from samples alone).")
    print(f"  coverage-none = {tot['none']/tot['n']:.0%}: neither interpretation surfaced (model wrong/other).")

    print(f"\n=== Detection: result-set divergence, ambiguous vs control ===")
    print(f"  mean #distinct result-clusters/8:  ambiguous {np.mean(div_amb):.2f}  control {np.mean(div_ctrl):.2f}")
    print(f"  collapse rate (all 8 -> 1 result): ambiguous {collapse_amb/len(div_amb):.0%}  control {collapse_ctrl/len(div_ctrl):.0%}")
    y = [1] * len(div_amb) + [0] * len(div_ctrl)
    s = div_amb + div_ctrl
    print(f"  AUROC of divergence predicting is_ambiguous: {auroc(s, y):.3f}")
    print(f"  control single-interpretation accuracy (sanity): {ctrl_acc/max(1,ctrl_n):.3f}")
    print("\nReading: high coverage-both => ambiguity is detectable from the posterior and the")
    print("clarify/select policy has candidates. High divergence-AUROC => result-set spread is")
    print("itself an ambiguity detector. Low coverage-both => detection needs the BNP prior.")


if __name__ == "__main__":
    main()
