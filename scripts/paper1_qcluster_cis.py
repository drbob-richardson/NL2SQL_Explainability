"""TMLR revision: question-CLUSTER bootstrap CIs for the trained-verifier transfer table (Table 4), addressing
the reviewer's point that the K=8 candidates per question are correlated. In-distribution and pooled AUROCs are
resampled by QUESTION (all 8 candidates of a resampled question move together); the LODO macro stays a
cluster-bootstrap by DATABASE. Also adds CIs for the feature-classifier row. Reconstructs the question grouping of
the saved per-example scores from data/verifier_data.jsonl + the deterministic seed-0 split, with a hard sanity
check that reconstructed labels match the saved labels. Cache only, no GPU/API.

  ./.venv/bin/python scripts/paper1_qcluster_cis.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "server_experiments", "results")


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def qcluster_ci(scores, labels, groups, nb=2000):
    """resample whole GROUPS (questions) with replacement."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    idx_of = {}
    for i, g in enumerate(groups):
        idx_of.setdefault(g, []).append(i)
    gkeys = list(idx_of); gidx = [np.array(idx_of[g]) for g in gkeys]
    rng = np.random.RandomState(0); v = []
    for _ in range(nb):
        pick = rng.randint(0, len(gkeys), len(gkeys))
        idx = np.concatenate([gidx[j] for j in pick])
        if len(set(labels[idx].tolist())) > 1:
            v.append(auroc(scores[idx], labels[idx]))
    return auroc(scores, labels), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def db_ci(per_db_auroc, nb=2000):
    a = np.array(list(per_db_auroc.values())); rng = np.random.RandomState(0)
    m = [a[rng.randint(0, len(a), len(a))].mean() for _ in range(nb)]
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---- reconstruct question grouping of the trained-verifier scores ----
rows = [json.loads(l) for l in open(os.path.join(ROOT, "server_experiments", "data", "verifier_data.jsonl"))]
qtup = [(r["db_id"], r["question_id"]) for r in rows]           # per candidate (6400)
lab = np.array([int(r["label"]) for r in rows])
uq = sorted(set(qtup)); np.random.RandomState(0).shuffle(uq); test_q = set(uq[:len(uq) // 5])  # 80/20, seed 0
indist_pos = [i for i in range(len(rows)) if qtup[i] in test_q]  # aligns with exp indist_scores order
indist_groups = [qtup[i] for i in indist_pos]
db_order = sorted(set(r["db_id"] for r in rows))
perdb_pos = {db: [i for i in range(len(rows)) if rows[i]["db_id"] == db] for db in db_order}  # aligns w/ lodo_per_db


def trained_row(name, path):
    j = json.load(open(path))
    # in-distribution: sanity-check reconstruction, then question-cluster CI
    idl = j["indist_labels"]
    assert list(lab[indist_pos]) == list(idl), f"{name}: in-dist label reconstruction MISMATCH"
    ind = qcluster_ci(j["indist_scores"], idl, indist_groups)
    # per-db (for the figure) + macro (db cluster) + pooled (question cluster over all held-out)
    ps, pl = j["lodo_per_db_scores"], j["lodo_per_db_labels"]
    per_au, pooled_s, pooled_l, pooled_g = {}, [], [], []
    for db in ps:
        assert list(lab[perdb_pos[db]]) == list(pl[db]), f"{name}/{db}: lodo label reconstruction MISMATCH"
        gq = [qtup[i] for i in perdb_pos[db]]
        per_au[db] = auroc(ps[db], pl[db])
        pooled_s += list(ps[db]); pooled_l += list(pl[db]); pooled_g += gq
    macro = db_ci(per_au); pooled = qcluster_ci(pooled_s, pooled_l, pooled_g)
    return name, ind, macro, pooled


def feature_row():
    ind = json.load(open(os.path.join(ROOT, "data", "verifier_probe_scores_indist.json")))
    lo = json.load(open(os.path.join(ROOT, "data", "verifier_probe_scores_lodo.json")))
    ci_ind = qcluster_ci(ind["oof"], ind["y"], ind["qids"])       # grouped-CV over all 6400, cluster by question
    per_au = {}
    for db in sorted(set(lo["dbs"])):
        m = [i for i in range(len(lo["dbs"])) if lo["dbs"][i] == db]
        per_au[db] = auroc(np.array(lo["oof"])[m], np.array(lo["y"])[m])
    macro = db_ci(per_au)
    return "feature classifier", ci_ind, macro, None


def fmt(t):
    return f"{t[0]:.3f}[{t[1]:.3f},{t[2]:.3f}]" if t else "---"


print("Table 4 with QUESTION-CLUSTER CIs (in-dist & pooled by question; macro by database):\n")
print(f"  {'verifier':<28}{'in-dist':<22}{'LODO macro':<22}{'LODO pooled'}")
res = [feature_row(),
       trained_row("ModernBERT-base", os.path.join(RES, "exp1_verifier_ModernBERT-base.json")),
       trained_row("Qwen2.5-1.5B", os.path.join(RES, "exp3_judge_Qwen2.5-1.5B-Instruct.json")),
       trained_row("Qwen2.5-7B", os.path.join(RES, "exp3_judge_Qwen2.5-7B-Instruct.json"))]
for name, ind, macro, pooled in res:
    print(f"  {name:<28}{fmt(ind):<22}{fmt(macro):<22}{fmt(pooled)}")
print("\n(all in-dist/pooled reconstructions passed the saved-label sanity check)")
