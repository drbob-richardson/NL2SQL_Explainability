"""Open-world table retrieval detection (Paper 2's candidate downstream payoff; no API).

In a data lake the correct table may be ABSENT. Can the retrieval posterior's confidence detect
that? For each question we score two scenarios:
  PRESENT : candidate set contains the gold table(s)      -> label 0
  ABSENT  : gold table(s) removed from the candidate set  -> label 1 (open-world / unanswerable)
and ask whether a confidence signal (max cosine, softmax top-prob peakedness, top-1/top-2 margin,
entropy) separates ABSENT from PRESENT. Two candidate sets: WITHIN-DB and the full CROSS-DB LAKE
(all tables across all BIRD dbs). This is the species-sampling 'is the truth out-of-set' question
at the table node -- the one Bayesian capability verifier/logprob correctness signals cannot give.

  ./.venv/bin/python scripts/bird_openworld.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from bird_lib import load_questions, load_schema
from bird_join_posterior import tables_in, table_text
from bird_column_posterior import auroc
from table_selection import EMB_CACHE

ROOT = os.path.join(os.path.dirname(__file__), "..")


def confidence_signals(sims):
    """sims: 1d array of cosine sims to candidate tables. Return dict of confidence signals."""
    s = np.sort(sims)[::-1]
    p = np.exp((sims - sims.max()) / 0.05); p = p / p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return dict(max_sim=float(s[0]),
                top_prob=float(p.max()),
                margin=float(s[0] - s[1]) if len(s) > 1 else 1.0,
                neg_entropy=-ent)


def main():
    qs = load_questions()
    emb = json.load(open(EMB_CACHE))
    def vec(t):
        v = np.array(emb[t], dtype=np.float32); return v / (np.linalg.norm(v) + 1e-9)

    schemas = {}
    lake = {}  # (db, table) -> vec
    for q in qs:
        db = q["db_id"]
        if db not in schemas:
            schemas[db] = load_schema(db)
            for t in schemas[db]:
                lake[(db, t)] = vec(table_text(db, t, schemas[db]))
    lake_keys = list(lake)
    lake_mat = np.stack([lake[k] for k in lake_keys])
    print(f"lake: {len(lake_keys)} tables across {len(schemas)} dbs")

    SIGS = ["max_sim", "top_prob", "margin", "neg_entropy"]
    within = {s: ([], []) for s in SIGS}   # (scores, labels) ; score: lower conf -> absent
    crossdb = {s: ([], []) for s in SIGS}
    ret_acc_present = []  # cross-db top-1 retrieval accuracy when gold present

    for q in qs:
        db = q["db_id"]
        if q["question"] not in emb:
            continue
        qv = vec(q["question"])
        gtabs = {t for t in tables_in(q["SQL"]) if t in schemas[db]}
        if not gtabs:
            continue

        # WITHIN-DB
        wtabs = list(schemas[db])
        wsims = np.array([float(qv @ lake[(db, t)]) for t in wtabs])
        present = confidence_signals(wsims)
        keep = [j for j, t in enumerate(wtabs) if t not in gtabs]
        if len(keep) >= 2:
            absent = confidence_signals(wsims[keep])
            for s in SIGS:
                within[s][0].append(present[s]); within[s][1].append(0)
                within[s][0].append(absent[s]); within[s][1].append(1)

        # CROSS-DB LAKE
        csims = lake_mat @ qv
        present_c = confidence_signals(csims)
        gold_idx = [i for i, k in enumerate(lake_keys) if k[0] == db and k[1] in gtabs]
        # retrieval accuracy: is the top-1 lake table a gold table (same db)?
        top1 = lake_keys[int(csims.argmax())]
        ret_acc_present.append(top1[0] == db and top1[1] in gtabs)
        mask = np.ones(len(lake_keys), bool); mask[gold_idx] = False
        absent_c = confidence_signals(csims[mask])
        for s in SIGS:
            crossdb[s][0].append(present_c[s]); crossdb[s][1].append(0)
            crossdb[s][0].append(absent_c[s]); crossdb[s][1].append(1)

    def report(name, d):
        print(f"\n  {name}: AUROC for detecting GOLD-TABLE ABSENT (open-world):")
        for s in SIGS:
            sc, lab = d[s]
            # lower confidence should mean absent -> use negative confidence as the absent score
            score = [-x for x in sc]
            print(f"    {s:<12}: {auroc(score, lab):.3f}")

    print(f"\ncross-db top-1 retrieval accuracy (gold present): {np.mean(ret_acc_present):.3f}")
    report("WITHIN-DB (small candidate set)", within)
    report("CROSS-DB LAKE (all tables)", crossdb)
    print("\nReading: AUROC>>0.5 means retrieval confidence detects when the correct table is")
    print("missing from the lake -- a calibrated open-world abstention signal for data lakes, the")
    print("Bayesian capability correctness-side signals (verifier/logprob) cannot provide.")


if __name__ == "__main__":
    main()
