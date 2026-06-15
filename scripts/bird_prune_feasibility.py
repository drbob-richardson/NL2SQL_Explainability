"""Schema-pruning FEASIBILITY (free precondition for Paper 2's downstream payoff; no API).

If we keep only the top-ranked tables/columns by the calibrated posterior, do we RETAIN the gold
elements? Pruning can only help generation if gold recall stays ~1.0 at a small schema budget.
We sweep the kept fraction and report (a) mean gold recall and (b) the fraction of questions that
retain ALL gold tables (the critical metric: dropping any gold table makes the query unanswerable).
Posteriors fit leave-one-DB-out (the realistic data-lake setting).

  ./.venv/bin/python scripts/bird_prune_feasibility.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from bird_lib import load_questions, load_schema, gold_columns
from bird_join_posterior import tables_in, table_text
from bird_column_posterior import gauss_posterior
from table_selection import EMB_CACHE

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    qs = load_questions()
    emb = json.load(open(EMB_CACHE))
    def vec(t):
        v = np.array(emb[t], dtype=np.float32); return v / (np.linalg.norm(v) + 1e-9)

    schemas = {}
    rows = []  # per question: db, tsims(list), t_is_gold(list), tables, csims, c_is_gold
    for q in qs:
        db = q["db_id"]
        if db not in schemas:
            schemas[db] = load_schema(db)
        sch = schemas[db]
        if q["question"] not in emb:
            continue
        qv = vec(q["question"])
        gtabs = tables_in(q["SQL"])
        gcols, _, _ = gold_columns(q["SQL"], sch)
        tabs = list(sch.keys())
        tsim = [float(qv @ vec(table_text(db, t, sch))) for t in tabs]
        tgold = [t in gtabs for t in tabs]
        # columns (all candidate cols, desc rep)
        cols = [(t, c) for t in sch for c in sch[t]]
        csim = [float(qv @ vec(sch[t][c]["desc"])) for (t, c) in cols]
        cgold = [(t, c) in gcols for (t, c) in cols]
        rows.append(dict(db=db, tabs=tabs, tsim=tsim, tgold=tgold,
                         cols=cols, csim=csim, cgold=cgold))

    dbs = sorted({r["db"] for r in rows})
    # LODO posterior calibration for tables and columns
    def lodo_post(simkey, goldkey):
        post = {}
        for held in dbs:
            s_tr = [s for r in rows if r["db"] != held for s in r[simkey]]
            y_tr = [g for r in rows if r["db"] != held for g in r[goldkey]]
            for i, r in enumerate(rows):
                if r["db"] != held:
                    continue
                p, _ = gauss_posterior(s_tr, y_tr, r[simkey])
                post[i] = p
        return post
    tpost = lodo_post("tsim", "tgold")
    cpost = lodo_post("csim", "cgold")

    print(f"BIRD schema-pruning feasibility (LODO posteriors), n={len(rows)} questions")
    print("\n  TABLE pruning (keep top fraction of tables by posterior):")
    print(f"  {'keep frac':>9} | {'mean gold-table recall':>22} | {'% q retain ALL gold tables':>27}")
    for frac in (0.25, 0.40, 0.50, 0.75):
        recs, allret = [], []
        for i, r in enumerate(rows):
            ngold = sum(r["tgold"])
            if ngold == 0:
                continue
            k = max(ngold, int(np.ceil(frac * len(r["tabs"]))))
            keep = set(np.argsort(-tpost[i])[:k].tolist())
            gidx = {j for j, g in enumerate(r["tgold"]) if g}
            rec = len(keep & gidx) / ngold
            recs.append(rec); allret.append(rec == 1.0)
        print(f"  {frac:>9.0%} | {np.mean(recs):>22.3f} | {np.mean(allret):>27.1%}")

    print("\n  COLUMN pruning (keep top fraction of columns by posterior):")
    print(f"  {'keep frac':>9} | {'mean gold-col recall':>22} | {'% q retain ALL gold cols':>27}")
    for frac in (0.25, 0.40, 0.50, 0.75):
        recs, allret = [], []
        for i, r in enumerate(rows):
            ngold = sum(r["cgold"])
            if ngold == 0:
                continue
            k = max(ngold, int(np.ceil(frac * len(r["cols"]))))
            keep = set(np.argsort(-cpost[i])[:k].tolist())
            gidx = {j for j, g in enumerate(r["cgold"]) if g}
            rec = len(keep & gidx) / ngold
            recs.append(rec); allret.append(rec == 1.0)
        print(f"  {frac:>9.0%} | {np.mean(recs):>22.3f} | {np.mean(allret):>27.1%}")

    print("\nReading: if we retain ~all gold tables/cols at a small keep-fraction, pruning is viable")
    print("and the paid accuracy test is worth running. If gold recall drops fast, pruning is unsafe.")


if __name__ == "__main__":
    main()
