"""Graph-level schema-linking UQ: does the composed per-node posterior confidence over the LLM's
GENERATED query (table + join-edge + SELECT-column relevance) predict execution correctness, and
ADD to self-consistency? (BIRD, reads data/bird_samples.json; class-conditional models fit on the
NON-slice BIRD questions so there is no leakage.)

Per generated modal query we score:
  f_table_rel : mean posterior P(table relevant|q) over the query's tables  (low => chose an
                off-question table)
  f_table_ret : peakedness of the table-retrieval posterior (max prob)      (Exp-0 signal)
  f_edge_rel  : mean posterior P(edge|q) over the query's join edges
  f_col_rel   : mean posterior P(col in SELECT|q) over the query's SELECT columns
  f_graph     : mean of the available node confidences
Compared to / combined with self-consistency top_prob. Cross-fit logistic for the combine.

  ./.venv/bin/python scripts/bird_graph_uq.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import numpy as np
import sqlglot
from sqlglot import exp
from bird_lib import load_questions, load_schema, gold_columns, select_columns
from bird_column_posterior import auroc, gauss_posterior
from bird_join_posterior import fk_graph, tables_in, table_text
from bnp_nl2sql.fit import LogisticCalibrator
from table_selection import embed_all, EMB_CACHE

ROOT = os.path.join(os.path.dirname(__file__), "..")


def tables_of(sql):
    try:
        return {t.name.lower() for t in sqlglot.parse_one(sql, dialect="sqlite").find_all(exp.Table)}
    except Exception:
        return set()


def main():
    cache = json.load(open(os.path.join(ROOT, "data", "bird_samples.json")))
    slice_q = list(cache.values())
    slice_ids = {(e["db_id"], e["question_id"]) for e in slice_q}
    print(f"slice: {len(slice_q)} generated questions, dbs={sorted({e['db_id'] for e in slice_q})}")

    allq = load_questions()
    tbls = {x["db_id"]: x for x in json.load(open(os.path.join(ROOT, "data", "bird", "dev_tables.json")))}
    schemas, graphs = {}, {}
    emb = json.load(open(EMB_CACHE))
    def vec(t):
        v = np.array(emb[t], dtype=np.float32); return v / (np.linalg.norm(v) + 1e-9)

    def get_schema(db):
        if db not in schemas:
            schemas[db] = load_schema(db)
            graphs[db] = fk_graph(tbls[db])
        return schemas[db]

    # ---- fit class-conditional posteriors on NON-slice questions (no leakage) ----
    tab_s, tab_y, edge_s, edge_y, col_s, col_y = [], [], [], [], [], []
    miss = set()
    for q in allq:
        if (q["db_id"], q["question_id"]) in slice_ids:
            continue
        db = q["db_id"]; sch = get_schema(db)
        if q["question"] not in emb:
            miss.add(q["question"]); continue
        qv = vec(q["question"])
        used = tables_in(q["SQL"]); selcols = select_columns(q["SQL"], sch)
        # table node
        for t in sch:
            tt = table_text(db, t, sch)
            if tt in emb:
                tab_s.append(float(qv @ vec(tt))); tab_y.append(t in used)
        # column node (SELECT)
        for t, cols in sch.items():
            for c, info in cols.items():
                if info["desc"] in emb:
                    col_s.append(float(qv @ vec(info["desc"]))); col_y.append((t, c) in selcols)
        # edge node
        for e in graphs[db]["edges"]:
            ta, tb = tuple(e)
            ta_t, tb_t = table_text(db, ta, sch), table_text(db, tb, sch)
            if ta_t in emb and tb_t in emb:
                sm = (float(qv @ vec(ta_t)) + float(qv @ vec(tb_t))) / 2
                edge_s.append(sm); edge_y.append(ta in used and tb in used)
    if miss:
        print(f"  ({len(miss)} non-slice questions missing embeddings, skipped in fit)")

    def fitter(s, y):
        s = np.array(s); y = np.array(y, dtype=bool)
        def post(x):
            p, _ = gauss_posterior(s, y, np.atleast_1d(x))
            return float(p[0])
        return post
    P_tab, P_edge, P_col = fitter(tab_s, tab_y), fitter(edge_s, edge_y), fitter(col_s, col_y)

    # ---- score the slice's generated modal queries ----
    rows = []
    need = set()
    for e in slice_q:
        if e["question"] not in emb:
            need.add(e["question"])
    if need:
        embed_all(sorted(need)); emb.update(json.load(open(EMB_CACHE)))

    for e in slice_q:
        db = e["db_id"]; sch = get_schema(db)
        mq = Counter(e["samples"]).most_common(1)[0][0]
        ok = bool(e["ok"][e["samples"].index(mq)])
        top = max(Counter(e["samples"]).values()) / len(e["samples"])  # self-consistency
        qv = vec(e["question"])
        # table-retrieval peakedness over candidates
        cand_t = list(sch.keys())
        tsims = np.array([float(qv @ vec(table_text(db, t, sch))) for t in cand_t])
        pt = np.exp((tsims - tsims.max()) / 0.05); pt = pt / pt.sum()
        f_table_ret = float(pt.max())
        gt, gcol, ge = tables_of(mq), select_columns(mq, sch), None
        # table relevance of the generated tables
        f_table_rel = float(np.mean([P_tab(float(qv @ vec(table_text(db, t, sch))))
                                     for t in gt if t in sch])) if any(t in sch for t in gt) else 0.5
        # SELECT col relevance
        cvals = [P_col(float(qv @ vec(sch[t][c]["desc"]))) for (t, c) in gcol if t in sch and c in sch[t]]
        f_col_rel = float(np.mean(cvals)) if cvals else 0.5
        # edge relevance over generated joins (FK edges among generated tables)
        gedges = [tuple(sorted((a, b))) for a in gt for b in gt if a < b
                  and frozenset((a, b)) in graphs[db]["edges"]]
        evals = []
        for (a, b) in gedges:
            at, bt = table_text(db, a, sch), table_text(db, b, sch)
            evals.append(P_edge((float(qv @ vec(at)) + float(qv @ vec(bt))) / 2))
        f_edge_rel = float(np.mean(evals)) if evals else 0.5
        comps = [f_table_rel, f_col_rel] + ([f_edge_rel] if evals else [])
        f_graph = float(np.mean(comps))
        rows.append(dict(ok=ok, top=top, f_table_rel=f_table_rel, f_table_ret=f_table_ret,
                         f_edge_rel=f_edge_rel, f_col_rel=f_col_rel, f_graph=f_graph))

    c = [r["ok"] for r in rows]
    n = len(rows)
    print(f"\nBIRD graph-level UQ: n={n}, modal exec accuracy={sum(c)/n:.3f}")
    print("  AUROC for predicting CORRECTNESS (each signal alone):")
    for k in ("top", "f_table_ret", "f_table_rel", "f_edge_rel", "f_col_rel", "f_graph"):
        print(f"    {k:<14}: {auroc([r[k] for r in rows], c):.3f}")

    # cross-fit logistic: self-consistency alone vs + graph features
    feats_base = [[r["top"]] for r in rows]
    feats_full = [[r["top"], r["f_table_ret"], r["f_table_rel"], r["f_edge_rel"], r["f_col_rel"]]
                  for r in rows]
    y = [1.0 if r["ok"] else 0.0 for r in rows]
    A = list(range(0, n, 2)); B = list(range(1, n, 2))
    def crossfit(feats):
        out = [None] * n
        for tr, te in ((A, B), (B, A)):
            clf = LogisticCalibrator().fit([feats[i] for i in tr], [y[i] for i in tr])
            for p, i in zip(clf.predict_proba([feats[i] for i in te]), te):
                out[i] = float(p)
        return out
    base_p, full_p = crossfit(feats_base), crossfit(feats_full)
    # export a graph-ONLY (no self-consistency) cross-fit confidence per question for the
    # unified correctness comparison (bird_correctness_uq.py)
    feats_graph = [[r["f_table_ret"], r["f_table_rel"], r["f_edge_rel"], r["f_col_rel"]] for r in rows]
    graph_only = crossfit(feats_graph)
    out = {}
    for e, g in zip(slice_q, graph_only):
        out[f"{e['db_id']}||{e['question_id']}"] = g
    json.dump(out, open(os.path.join(ROOT, "data", "bird_graph_conf.json"), "w"))
    print("\n  cross-fit logistic combine:")
    print(f"    self-consistency alone     : {auroc(base_p, c):.3f}")
    print(f"    self-consistency + graph UQ: {auroc(full_p, c):.3f}")
    # bootstrap the delta (resample questions)
    rng = np.random.RandomState(0)
    cb = np.array(base_p); cf = np.array(full_p); cc = np.array(c)
    deltas, bases, fulls = [], [], []
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        if len(set(cc[idx])) < 2:
            continue
        ab = auroc(cb[idx], cc[idx]); af = auroc(cf[idx], cc[idx])
        bases.append(ab); fulls.append(af); deltas.append(af - ab)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"    bootstrap delta (full-base): {np.mean(deltas):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"  P(delta>0)={np.mean(np.array(deltas)>0):.2f}")
    print("\nReading: if (self-consistency + graph UQ) > self-consistency alone, the composed")
    print("schema-linking posterior carries complementary error-prediction signal.")


if __name__ == "__main__":
    main()
