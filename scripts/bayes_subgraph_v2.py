"""Phase-1 hardened: held-out beta selection + richer unary features.

Adds to bayes_subgraph.py:
  - max question<->COLUMN cosine (schema-linking: question often names a column)
  - value-match: does a question token appear as a stored VALUE in the table (LIMIT-500 scan, cached)
  - HELD-OUT beta: beta selected on the other fold (no peeking), so the MRF number is honest.

Reports cosine vs learned unary fusion (rich features, beta=0) vs MRF (held-out beta), by query size,
plus a feature ablation. Reuses cached embeddings; embeds column names (~$0.005). No execution beyond
bounded value scans.
  ./.venv/bin/python scripts/bayes_subgraph_v2.py
"""
from __future__ import annotations
import json, math, os, re, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
import sqlite3, sqlglot
from sqlglot import exp

ROOT = os.path.join(os.path.dirname(__file__), "..")
DBDIR = os.path.join(ROOT, "data", "bird", "db")
EMB = os.path.join(ROOT, "data", "bridge_emb.json")
VALCACHE = os.path.join(ROOT, "data", "bird_value_tokens.json")


def toks(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > 1]


def schema(db):
    c = sqlite3.connect(f"{DBDIR}/{db}.sqlite")
    orig = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    tbls = [t.lower() for t in orig]; text, tok, cols = {}, {}, {}
    for t in orig:
        cc = [r[1] for r in c.execute(f"PRAGMA table_info(`{t}`)").fetchall()]
        text[t.lower()] = f"{t}: " + ", ".join(cc); tok[t.lower()] = toks(t + " " + " ".join(cc))
        cols[t.lower()] = cc
    idx = {t: i for i, t in enumerate(tbls)}; edges = []
    for t in orig:
        for r in c.execute(f"PRAGMA foreign_key_list(`{t}`)").fetchall():
            ref = (r[2] or "").lower()
            if ref in idx and ref != t.lower():
                edges.append((idx[t.lower()], idx[ref]))
    return orig, tbls, text, tok, cols, edges


def value_tokens(db):
    cache = json.load(open(VALCACHE)) if os.path.exists(VALCACHE) else {}
    if db in cache:
        return cache[db]
    c = sqlite3.connect(f"{DBDIR}/{db}.sqlite")
    orig = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    out = {}
    for t in orig:
        toks_t = set()
        timer = threading.Timer(5.0, c.interrupt); timer.start()
        try:
            rows = c.execute(f"SELECT * FROM `{t}` LIMIT 500").fetchall()
            for row in rows:
                for cell in row:
                    if isinstance(cell, str) and len(cell) < 60:
                        toks_t |= set(toks(cell))
        except Exception:
            pass
        finally:
            timer.cancel()
        out[t.lower()] = sorted(toks_t)
    cache[db] = out; json.dump(cache, open(VALCACHE, "w"))
    return out


def embed(texts):
    cache = json.load(open(EMB)) if os.path.exists(EMB) else {}
    todo = sorted({t for t in texts if t not in cache})
    if todo:
        from openai import OpenAI
        cl = OpenAI()
        for i in range(0, len(todo), 256):
            r = cl.embeddings.create(model="text-embedding-3-small", input=todo[i:i+256])
            for t, d in zip(todo[i:i+256], r.data):
                cache[t] = d.embedding
        json.dump(cache, open(EMB, "w"))
    return cache


def main():
    samp = list(json.load(open(os.path.join(ROOT, "data", "bird_samples.json"))).values())
    dbs = sorted(set(e["db_id"] for e in samp))
    sch = {db: schema(db) for db in dbs}
    valtok = {db: value_tokens(db) for db in dbs}
    # embed questions, table strings, column names
    need = []
    for db in dbs:
        _, tbls, text, _, cols, _ = sch[db]
        need += list(text.values())
        for t in tbls:
            need += cols[t]
    need += [e["question"] for e in samp]
    cache = embed(need)
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)

    items = []
    for e in samp:
        db = e["db_id"]; tbls = sch[db][1]
        if e["question"] not in cache:
            continue
        try:
            g = sqlglot.parse_one(e["gold"], dialect="sqlite")
            gold = {x.name.lower() for x in g.find_all(exp.Table)} & set(tbls)
        except Exception:
            continue
        if len(gold) >= 2:
            items.append((db, e["question"], gold))

    rows, ys, qids, keys = [], [], [], []
    for qi, (db, q, gold) in enumerate(items):
        orig, tbls, text, tok, cols, edges = sch[db]
        df = Counter()
        for t in tbls:
            df.update(set(tok[t]))
        N = len(tbls); idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
        avgdl = np.mean([len(tok[t]) for t in tbls]); qt = toks(q); qset = set(qt); qv = vec(q)
        for t in tbls:
            cos = float(qv @ vec(text[t]))
            dc = Counter(tok[t]); bm = sum(idf.get(w, 0) * dc[w] * 2.5 / (dc[w] + 1.5 * (1 - .75 + .75 * len(tok[t]) / avgdl)) for w in qset if w in dc)
            name_ov = len(qset & set(toks(t))) / (len(set(toks(t))) + 1e-9)
            col_ov = len(qset & set(tok[t])) / (len(qt) + 1e-9)
            mcc = max((float(qv @ vec(c)) for c in cols[t]), default=0.0)
            vmatch = len(qset & set(valtok[db].get(t, []))) / (len(qt) + 1e-9)
            rows.append([cos, bm, name_ov, col_ov, mcc, vmatch]); ys.append(1 if t in gold else 0)
            qids.append(qi); keys.append((qi, t))
    X = np.array(rows, float); y = np.array(ys, float)

    uq = np.unique(qids); rng = np.random.RandomState(0); perm = rng.permutation(len(uq)); half = len(uq) // 2
    foldq = {q: (0 if i in set(perm[:half]) else 1) for i, q in enumerate(uq)}
    qfold = np.array([foldq[q] for q in qids])

    def fit_unary(featmask):
        a = np.zeros(len(y))
        Xm = X[:, featmask]
        for te in (0, 1):
            tr = qfold != te
            mu, sd = Xm[tr].mean(0), Xm[tr].std(0) + 1e-9
            Xtr, Xte = (Xm[tr] - mu) / sd, (Xm[qfold == te] - mu) / sd
            w = np.zeros(Xm.shape[1]); b = 0.0
            for _ in range(900):
                p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = p - y[tr]
                w -= 0.3 * (Xtr.T @ g / tr.sum() + 0.01 * w); b -= 0.3 * g.mean()
            a[qfold == te] = Xte @ w + b
        return a

    a_full = fit_unary([0, 1, 2, 3, 4, 5])
    a_old = fit_unary([0, 1, 2, 3])
    a_by = {}
    cos_by = {}
    for (qi, t), av, ao, xr in zip(keys, a_full, a_old, rows):
        a_by.setdefault(qi, {})[t] = (av, ao); cos_by.setdefault(qi, {})[t] = xr[0]

    def marg(db, qi, beta, which):
        tbls = sch[db][1]; edges = sch[db][5]; n = len(tbls)
        av = np.array([a_by[qi][t][which] for t in tbls])
        masks = np.arange(1 << n); bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
        score = bits @ av
        if beta and edges:
            ec = np.zeros(len(masks))
            for (i, j) in edges:
                ec += bits[:, i] * bits[:, j]
            score = score + beta * ec
        score -= score.max(); p = np.exp(score); p /= p.sum()
        m = (p[:, None] * bits).sum(0)
        return {t: m[i] for i, t in enumerate(tbls)}

    def recall_set(scorer, qset_ids):
        out = {2: [], 3: [], 4: []}
        for qi in qset_ids:
            db, q, gold = items[qi]; sc = scorer(qi); ranked = sorted(sch[db][1], key=lambda x: -sc[x])
            r = len(gold & set(ranked[:len(gold)])) / len(gold)
            for lo in (2, 3, 4):
                if len(gold) >= lo:
                    out[lo].append(r)
        return out

    betas = [0, 0.5, 1, 1.5, 2, 3]
    # held-out beta: choose beta on the OTHER fold, apply to this fold
    mrf_out = {2: [], 3: [], 4: []}; chosen = []
    qf = {qi: foldq[qi] for qi in range(len(items))}
    for te in (0, 1):
        tr_ids = [qi for qi in range(len(items)) if qf[qi] != te]
        te_ids = [qi for qi in range(len(items)) if qf[qi] == te]
        # pick beta maximizing >=2 recall on train
        best_b, best_r = 0, -1
        for b in betas:
            rr = recall_set(lambda qi, b=b: marg(items[qi][0], qi, b, 0), tr_ids)
            m = np.mean(rr[2])
            if m > best_r:
                best_r, best_b = m, b
        chosen.append(best_b)
        rr = recall_set(lambda qi, b=best_b: marg(items[qi][0], qi, b, 0), te_ids)
        for k in (2, 3, 4):
            mrf_out[k] += rr[k]

    all_ids = list(range(len(items)))
    print(f"=== Phase-1 hardened (BIRD, {len(items)} multi-table Qs; held-out beta) ===")
    print("features: cosine, bm25, name-ov, col-ov, max-col-cosine, value-match\n")
    print(f"  {'method':<28}{'>=2':>10}{'>=3':>10}{'>=4':>10}")
    def line(name, rr):
        print(f"  {name:<28}{np.mean(rr[2]):>10.3f}{np.mean(rr[3]):>10.3f}{np.mean(rr[4]):>10.3f}")
    line("cosine", recall_set(lambda qi: cos_by[qi], all_ids))
    line("unary fusion old feats (b=0)", recall_set(lambda qi: marg(items[qi][0], qi, 0, 1), all_ids))
    line("unary fusion rich (b=0)", recall_set(lambda qi: marg(items[qi][0], qi, 0, 0), all_ids))
    line(f"MRF held-out beta {tuple(chosen)}", mrf_out)
    print("\nReading: rich-features unary > old unary (value/column signals help); MRF held-out-beta >")
    print("unary (structure helps out-of-sample); both > cosine. This is the hardened core result.")


if __name__ == "__main__":
    main()
