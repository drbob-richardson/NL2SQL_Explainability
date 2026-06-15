"""Open-world discovery on BIRD (Part A's BNP win, ported to hard multi-table; no API).

Pitman-Yor discovery probability (theta + d*K)/(theta + N) over the sampled query STRUCTURES
predicts open-world failure = "no sampled query is execution-correct" (answering is guaranteed
wrong). On Spider single-table this detected gold-unseen at AUROC ~0.84 oos where disagreement
signals were at chance. Here we test whether it survives to BIRD (only ~24% unanimous, ~49%
gold-unseen). Reuses cached self-consistency / execution signals (data/bird_signals.json).

  ./.venv/bin/python scripts/bird_discovery.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
from bnp_nl2sql import sql_to_graph, model_a_posterior
from bnp_nl2sql.fit import fit_pyp_partitions, empirical_base
from bird_column_posterior import auroc

ROOT = os.path.join(os.path.dirname(__file__), "..")


def ck(s):
    try:
        return sql_to_graph(s).canonical_key()
    except Exception:
        return "<unparseable>"


def sk(s):
    try:
        return sql_to_graph(s).skeleton_key()
    except Exception:
        return "<unparseable>"


def disc_scores(parts, keyfn_parts):
    fit = fit_pyp_partitions(keyfn_parts)
    d, th = fit.discount, fit.concentration
    out = []
    for keys in keyfn_parts:
        N = len(keys); K = len(set(keys))
        out.append((th + d * K) / (th + N))
    return out, d, th


def main():
    samples = json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))
    items = list(samples.values())
    sig = json.load(open(os.path.join(ROOT, "data", "bird_signals.json")))  # aligned order
    assert len(sig) == len(items)

    parts_ck = [[ck(s) for s in e["samples"]] for e in items]
    parts_sk = [[sk(s) for s in e["samples"]] for e in items]
    n = len(items)

    # labels
    ow_fail = [(not any(e["ok"])) for e in items]                 # open-world: no correct sample
    modal_wrong = []
    for e in items:
        mq = Counter(e["samples"]).most_common(1)[0][0]
        modal_wrong.append(not bool(e["ok"][e["samples"].index(mq)]))

    # discovery (in-sample fit) on canonical + skeleton structures
    disc_c, dC, thC = disc_scores(items, parts_ck)
    disc_s, dS, thS = disc_scores(items, parts_sk)

    # baselines (disagreement) from cached signals + simple counts
    one_minus_top = [1 - r["top"] for r in sig]
    one_minus_sem = [1 - r["sem"] for r in sig]
    n_distinct = [len(set(p)) / len(p) for p in parts_ck]

    print(f"BIRD discovery: n={n}; gold-unseen(ow_fail)={sum(ow_fail)} ({sum(ow_fail)/n:.1%}); "
          f"modal-wrong={sum(modal_wrong)} ({sum(modal_wrong)/n:.1%})")
    print(f"  PY fit: canonical d={dC:.3f} theta={thC:.3f} | skeleton d={dS:.3f} theta={thS:.3f}")

    def block(label, y):
        print(f"\n  AUROC for predicting {label}:")
        print(f"    discovery (canonical)   : {auroc(disc_c, y):.3f}")
        print(f"    discovery (skeleton)    : {auroc(disc_s, y):.3f}")
        print(f"    1 - top_prob (disagree) : {auroc(one_minus_top, y):.3f}")
        print(f"    1 - semantic_top        : {auroc(one_minus_sem, y):.3f}")
        print(f"    n_distinct / N          : {auroc(n_distinct, y):.3f}")
    block("OPEN-WORLD failure (no correct sample)", ow_fail)
    block("modal-query INCORRECT", modal_wrong)

    # parity cross-fit for discovery (fit PY on half, score other half) -> oos check
    A = list(range(0, n, 2)); B = list(range(1, n, 2))
    oos = [None] * n
    for tr, te in ((A, B), (B, A)):
        fit = fit_pyp_partitions([parts_ck[i] for i in tr])
        d, th = fit.discount, fit.concentration
        for i in te:
            N = len(parts_ck[i]); K = len(set(parts_ck[i]))
            oos[i] = (th + d * K) / (th + N)
    print(f"\n  discovery (canonical) OUT-OF-SAMPLE AUROC for ow_fail: {auroc(oos, ow_fail):.3f}")

    # ---- H-WEIGHTED discovery (the version that won on Spider): base measure from gold structures.
    # This is the signal that detects rare/unseen structures via H. Test in-distribution (pooled,
    # optimistic -- includes same-db golds) AND leave-one-DB-out (the data-lake reality).
    gold_ck = [ck(e["gold"]) for e in items]
    dbs = [e["db_id"] for e in items]
    fitC = fit_pyp_partitions(parts_ck)
    def hweighted(train_idx, eval_idx):
        H = empirical_base([gold_ck[i] for i in train_idx])
        disc, conf = {}, {}
        conc = max(fitC.concentration, 0.05)
        for i in eval_idx:
            post = model_a_posterior(items[i]["samples"], discount=fitC.discount,
                                     concentration=conc, full_discount=fitC.discount,
                                     full_concentration=conc,
                                     full_base=lambda s: H.get(s, 0.0))
            disc[i] = post.full_discovery_probability
            conf[i] = post.confidence()
        return disc, conf
    # in-distribution (pooled): train H on all, eval all
    dH, cH = hweighted(list(range(n)), list(range(n)))
    discH = [dH[i] for i in range(n)]
    print(f"\n  H-WEIGHTED discovery (in-distribution H, pooled): "
          f"AUROC ow_fail {auroc(discH, ow_fail):.3f}  modal-wrong {auroc(discH, modal_wrong):.3f}")
    # leave-one-DB-out: H from other DBs only (data-lake reality)
    dHl = {}
    for held in set(dbs):
        tr = [i for i in range(n) if dbs[i] != held]
        te = [i for i in range(n) if dbs[i] == held]
        d_, _ = hweighted(tr, te); dHl.update(d_)
    discHl = [dHl[i] for i in range(n)]
    print(f"  H-WEIGHTED discovery (LEAVE-ONE-DB-OUT H, the real setting): "
          f"AUROC ow_fail {auroc(discHl, ow_fail):.3f}  modal-wrong {auroc(discHl, modal_wrong):.3f}")

    print("\nReading: if discovery >> disagreement baselines for OPEN-WORLD failure, the PY signal")
    print("survives to hard multi-table BIRD and Paper 2's BNP spine holds. If it's ~ the same as")
    print("1-top_prob, discovery's edge was specific to the saturated single-table regime.")


if __name__ == "__main__":
    main()
