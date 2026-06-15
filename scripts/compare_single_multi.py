"""Single-table vs multi-table Spider: is multi-table the healthier UQ regime? (no API)

Determines whether Option C is viable. For each regime we report, with TIE-ROBUST metrics:
  - accuracy (headroom for UQ),
  - saturation (% unanimous K=1) -- the thing that broke single-table,
  - self-consistency AUROC (does disagreement carry signal here?),
  - structural-prior (PY/membership) AUROC and whether it ADDS over self-consistency,
  - AURC vs tie-robust AUROC gap (how misleading is AURC under this regime's saturation),
  - open-world failure rate (corrected label) and discovery AUROC.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import json
from collections import Counter
from bnp_nl2sql import sql_to_graph, model_a_posterior
from bnp_nl2sql.calibrate import aurc
from bnp_nl2sql.execeval import exec_match, open_db
from bnp_nl2sql.fit import empirical_base, fit_pyp_partitions
from bnp_nl2sql.uq_baselines import structural_top_prob
from spider_benchmark import fetch_db

ROOT = os.path.join(os.path.dirname(__file__), "..")
def ck(s):
    try: return sql_to_graph(s).canonical_key()
    except Exception: return "<u>"
def auroc(sc, lab):
    pos=[s for s,y in zip(sc,lab) if y]; neg=[s for s,y in zip(sc,lab) if not y]
    if not pos or not neg: return float("nan")
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))


def analyze(cache_file, label):
    cache = json.load(open(os.path.join(ROOT, "data", cache_file)))
    conns = {}
    items = []
    for e in cache.values():
        db = e["db_id"]
        if db not in conns:
            conns[db] = open_db(fetch_db(db))
        items.append((e["samples"], e["gold"], conns[db]))
    Hf = empirical_base([ck(g) for _, g, _ in items])
    fit = fit_pyp_partitions([[ck(s) for s in samp] for samp, _, _ in items])

    rows = []
    for samp, gold, conn in items:
        p = model_a_posterior(samp, discount=fit.discount, concentration=fit.concentration,
                              full_discount=fit.discount, full_concentration=fit.concentration,
                              full_base=lambda s: Hf.get(s, 0.0))
        mq = p.map_query()
        try: ok = exec_match(mq, gold, conn) if mq else False
        except Exception: ok = False
        ow = True
        for s in samp:
            try:
                if exec_match(s, gold, conn): ow = False; break
            except Exception: pass
        rows.append(dict(correct=ok, top=structural_top_prob(samp),
                         py=p.confidence()*(1-p.full_discovery_probability),
                         mem=1.0 if Hf.get(ck(mq), 0) > 0 else 0.0,
                         disc=p.full_discovery_probability, K=p.pyp_full.K, ow_fail=ow))
    c = [r["correct"] for r in rows]
    n = len(rows)
    print(f"\n{'='*60}\n{label}: n={n}\n{'='*60}")
    print(f"  execution accuracy        : {sum(c)/n:.3f}")
    print(f"  saturation (% unanimous)  : {sum(r['K']==1 for r in rows)/n:.0%}")
    print(f"  -- tie-robust AUROC for correctness --")
    print(f"    self-consistency top_prob : {auroc([r['top'] for r in rows], c):.3f}")
    print(f"    structural membership     : {auroc([r['mem'] for r in rows], c):.3f}")
    print(f"    PY confidence (full)      : {auroc([r['py'] for r in rows], c):.3f}")
    print(f"  -- AURC (tie-sensitive) --")
    print(f"    PY={aurc([r['py'] for r in rows], c):.3f}  baseline={aurc([r['top'] for r in rows], c):.3f}")
    nf = sum(r["ow_fail"] for r in rows)
    print(f"  open-world failure (no exec-correct sample): {nf}/{n} ({nf/n:.0%})")
    print(f"    discovery AUROC for open-world failure   : {auroc([r['disc'] for r in rows], [r['ow_fail'] for r in rows]):.3f}")


def main():
    analyze("spider_samples.json", "SINGLE-TABLE")
    analyze("spider_samples_multi.json", "MULTI-TABLE")
    print("\nOption C is viable if multi-table shows: lower accuracy (headroom), LOWER saturation,")
    print("and the structural prior still ADDS AUROC over self-consistency -- with AURC closer to")
    print("the tie-robust AUROC (i.e., the metric is no longer a saturation artifact).")


if __name__ == "__main__":
    main()
