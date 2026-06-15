"""Join-edge node of the query-graph posterior on BIRD (no LLM generation).

Candidate edges = FK-connected table pairs (the join graph). An edge is POSITIVE for a question
if both endpoint tables appear in the gold SQL. Question: is join-edge inclusion SEMANTIC (driven
by whether the question is about both endpoint entities -- like the table node) or STRUCTURAL
(just on the join path -- like filter columns)?

We compare, per edge:
  SEMANTIC score : aggregation of endpoint table->question cosine similarities (min/mean/prod),
                   fed through the same class-conditional Bayesian update used for columns.
  STRUCTURAL base: marginal edge-usage frequency (fit on train) -- structure-only prior.
  COMBINED       : both via cross-fit logistic.
Metrics: AUROC + per-question AUROC + recall@k + ECE, parity split and leave-one-DB-out.

Embeddings cached (table-level dict texts; ~$0.001). Run:
  ./.venv/bin/python scripts/bird_join_posterior.py            # dry
  ./.venv/bin/python scripts/bird_join_posterior.py --run
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import sqlglot
from sqlglot import exp
from bird_lib import load_questions, load_schema
from bird_column_posterior import auroc, ece, gauss_posterior
from bnp_nl2sql.fit import LogisticCalibrator
from table_selection import embed_all, EMB_CACHE

ROOT = os.path.join(os.path.dirname(__file__), "..")


def fk_graph(tbl_meta):
    """-> dict db -> {'edges': set frozenset{ta,tb}, 'tables': [..]}"""
    cn = tbl_meta["column_names_original"]
    tn = [t.lower() for t in tbl_meta["table_names_original"]]
    edges = set()
    for a, b in tbl_meta["foreign_keys"]:
        ta = tn[cn[a][0]]; tb = tn[cn[b][0]]
        if ta != tb:
            edges.add(frozenset((ta, tb)))
    return {"edges": edges, "tables": tn}


def tables_in(sql):
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        try:
            tree = sqlglot.parse_one(sql)
        except Exception:
            return set()
    return {t.name.lower() for t in tree.find_all(exp.Table)}


def table_text(db, table, schema):
    """Table-level data-dictionary representation: 'table: col, col, ...' (clean, no boilerplate)."""
    cols = list(schema.get(table, {}).keys())
    return f"{table}: " + ", ".join(cols[:30])


def build():
    qs = load_questions()
    tbls = {x["db_id"]: x for x in json.load(open(os.path.join(ROOT, "data", "bird", "dev_tables.json")))}
    schemas, graphs, texts = {}, {}, set()
    data = []
    for q in qs:
        db = q["db_id"]
        if db not in schemas:
            schemas[db] = load_schema(db)
            graphs[db] = fk_graph(tbls[db])
        used = tables_in(q["SQL"])
        edges = sorted(tuple(sorted(e)) for e in graphs[db]["edges"])
        pos = {e for e in edges if e[0] in used and e[1] in used}
        for (ta, tb) in edges:
            texts.add(table_text(db, ta, schemas[db]))
            texts.add(table_text(db, tb, schemas[db]))
        texts.add(q["question"])
        data.append(dict(db=db, q=q["question"], edges=edges, pos=pos))
    return data, sorted(texts), schemas


def assemble(data, schemas, emb):
    def vec(t):
        v = emb[t]; return v / (np.linalg.norm(v) + 1e-9)
    # structural prior: edge-usage frequency over ALL questions' positives (will cross-fit below)
    for d in data:
        qv = vec(d["q"])
        rows = []
        for (ta, tb) in d["edges"]:
            sa = float(qv @ vec(table_text(d["db"], ta, schemas[d["db"]])))
            sb = float(qv @ vec(table_text(d["db"], tb, schemas[d["db"]])))
            rows.append(dict(e=(ta, tb), z=((ta, tb) in d["pos"]),
                             smin=min(sa, sb), smean=(sa + sb) / 2, sprod=sa * sb,
                             db=d["db"]))
        d["rows"] = rows


def edge_freq(train):
    """structural prior: P(edge used) marginal, per (db, edge)."""
    num, den = {}, {}
    for d in train:
        for r in d["rows"]:
            k = (r["db"], r["e"])
            den[k] = den.get(k, 0) + 1
            num[k] = num.get(k, 0) + (1 if r["z"] else 0)
    glob = sum(num.values()) / max(sum(den.values()), 1)
    return {k: (num[k] + glob) / (den[k] + 1) for k in den}, glob


def evaluate(data, split, score="smin", combine=False):
    tr = [d for d, s in zip(data, split) if s]
    te = [d for d, s in zip(data, split) if not s]
    freq, glob = edge_freq(tr)
    s_tr = [r[score] for d in tr for r in d["rows"]]
    y_tr = [r["z"] for d in tr for r in d["rows"]]
    s_te = [r[score] for d in te for r in d["rows"]]
    y_te = [r["z"] for d in te for r in d["rows"]]
    post, _ = gauss_posterior(s_tr, y_tr, s_te)
    struct_te = [freq.get((r["db"], r["e"]), glob) for d in te for r in d["rows"]]
    out = {}
    out["sem_auroc"] = auroc(s_te, y_te)
    out["struct_auroc"] = auroc(struct_te, y_te)
    out["sem_ece"] = ece(post, y_te)
    if combine:
        feats_tr = [[r[score], freq.get((r["db"], r["e"]), glob)] for d in tr for r in d["rows"]]
        feats_te = [[r[score], freq.get((r["db"], r["e"]), glob)] for d in te for r in d["rows"]]
        clf = LogisticCalibrator().fit(feats_tr, [float(v) for v in y_tr])
        comb = clf.predict_proba(feats_te)
        out["comb_auroc"] = auroc(comb, y_te)
    # per-question AUROC + recall@k on semantic posterior
    perq, recs = [], []
    idx = 0
    for d in te:
        n = len(d["rows"]); p = post[idx:idx + n]; idx += n
        zt = np.array([r["z"] for r in d["rows"]])
        k = int(zt.sum())
        if 0 < k < n:
            perq.append(auroc(p, zt))
            topk = set(np.argsort(-p)[:k].tolist())
            recs.append(len(topk & set(np.where(zt)[0].tolist())) / k)
    out["sem_perq"] = float(np.mean(perq)) if perq else float("nan")
    out["recall_at_k"] = float(np.mean(recs)) if recs else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    data, texts, schemas = build()

    cache = json.load(open(EMB_CACHE)) if os.path.exists(EMB_CACHE) else {}
    todo = [t for t in texts if t not in cache]
    print(f"unique texts: {len(texts)}; to embed: {len(todo)}; est cost "
          f"${sum(min(len(t),400) for t in todo)/4/1e6*0.02:.4f}")
    if todo and not args.run:
        print("[dry run] re-run with --run."); return
    emb = embed_all(texts)
    assemble(data, schemas, emb)

    nq_multi = sum(any(r["z"] for r in d["rows"]) for d in data)
    ne = sum(len(d["rows"]) for d in data)
    npos = sum(sum(r["z"] for r in d["rows"]) for d in data)
    print(f"\nBIRD join-edge node: {len(data)} questions ({nq_multi} have >=1 join edge), "
          f"{ne} candidate edges, {npos} positive")
    cand_per_q = ne / len(data)
    print(f"avg candidate edges/q: {cand_per_q:.1f}")

    parity = [i % 2 == 0 for i in range(len(data))]
    print("\nParity split:")
    print(f"  {'score':<8} {'SEM AUROC':>10} {'STRUCT AUROC':>13} {'COMB AUROC':>11} "
          f"{'SEM perQ':>9} {'recall@k':>9} {'SEM ECE':>8}")
    for sc in ("smin", "smean", "sprod"):
        r = evaluate(data, parity, score=sc, combine=True)
        print(f"  {sc:<8} {r['sem_auroc']:>10.3f} {r['struct_auroc']:>13.3f} "
              f"{r['comb_auroc']:>11.3f} {r['sem_perq']:>9.3f} {r['recall_at_k']:>9.3f} "
              f"{r['sem_ece']:>8.3f}")

    print("\nLeave-one-DB-out (score=smean):")
    dbs = sorted({d["db"] for d in data})
    agg = {k: [] for k in ("sem_auroc", "struct_auroc", "comb_auroc", "sem_perq", "recall_at_k")}
    for held in dbs:
        split = [d["db"] != held for d in data]
        r = evaluate(data, split, score="smean", combine=True)
        for k in agg:
            if not np.isnan(r.get(k, float("nan"))):
                agg[k].append(r[k])
    print("  " + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in agg.items() if v))

    print("\nReading: if STRUCT AUROC >> SEM, joins are structural (FK-frequency predicts better");
    print("than question semantics). If SEM is competitive, the edge node is semantically rankable")
    print("like the table node and the embedding posterior earns its place on the graph.")


if __name__ == "__main__":
    main()
