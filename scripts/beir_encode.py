"""Stage 1 of the hierarchical few-shot reranking study: encode + first-stage retrieve + features.

For each BEIR domain: encode corpus and test queries with a small dense encoder (bge-small-en-v1.5,
MPS/CPU), dense-retrieve a top-100 candidate pool per query, cross-encoder score the pool
(ms-marco-MiniLM-L-6-v2), and build a shared per-(query,doc) feature vector:
  [dense_cos, ce_score, bm25, token_jaccard, unigram_overlap, log_doclen, log_qlen] ++ 32-dim random
  projection of the query-doc embedding elementwise product (a shared, domain-agnostic semantic basis).
Saves one .npz per domain with, per eval query, the candidate feature matrix and graded relevance
labels. Reranking heads (stage 2, beir_hier.py) differ only in how they pool across domains.
  ./.venv/bin/python scripts/beir_encode.py
"""
from __future__ import annotations
import json, math, os, re, sys
from collections import Counter, defaultdict
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOMAINS = ["nfcorpus", "arguana", "scidocs", "fiqa", "scifact"]
POOL = 100          # dense candidate pool per query
QMAX = 220          # cap eval queries per domain (few-shot uses a slice; rest for eval)
PROJ = 32           # random-projection dims of the q*d embedding product
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def toks(s):
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(w) > 2]


def main():
    print(f"device={DEV}")
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEV)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEV, max_length=512)
    rng = np.random.RandomState(0)
    R = rng.randn(384, PROJ).astype(np.float32) / math.sqrt(384)   # shared projection across domains

    for dom in DOMAINS:
        dd = os.path.join(ROOT, "data", "beir", dom)
        out = os.path.join(dd, "features.npz")
        if os.path.exists(out):
            print(f"[{dom}] cached, skipping"); continue
        corpus = pq.read_table(os.path.join(dd, "corpus.parquet")).to_pylist()
        qtab = pq.read_table(os.path.join(dd, "queries.parquet")).to_pylist()
        qtext = {str(q["_id"]): q["text"] for q in qtab}
        qrels = defaultdict(dict)
        with open(os.path.join(dd, "qrels_test.tsv")) as f:
            next(f)
            for line in f:
                a = line.split()
                if len(a) >= 3 and int(a[2]) > 0:
                    qrels[a[0]][a[1]] = int(a[2])
        docids = [str(c["_id"]) for c in corpus]
        did2i = {d: i for i, d in enumerate(docids)}
        dtexts = [((c.get("title") or "") + ". " + (c.get("text") or "")).strip() for c in corpus]
        dtok = [toks(t) for t in dtexts]
        df = Counter()
        for tk in dtok:
            df.update(set(tk))
        N = len(dtexts); idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
        avgdl = np.mean([len(tk) for tk in dtok]) + 1e-9

        # encode corpus (passages as-is) and queries (bge query instruction)
        print(f"[{dom}] encoding {N} docs ...", flush=True)
        D = enc.encode(dtexts, batch_size=256, normalize_embeddings=True, show_progress_bar=False,
                       convert_to_numpy=True).astype(np.float32)
        qids = [q for q in qtext if q in qrels and any(d in did2i for d in qrels[q])][:QMAX]
        qinstr = ["Represent this sentence for searching relevant passages: " + qtext[q] for q in qids]
        Q = enc.encode(qinstr, batch_size=256, normalize_embeddings=True, show_progress_bar=False,
                       convert_to_numpy=True).astype(np.float32)

        feats_all, labels_all, qptr = [], [], [0]
        ce_pairs, ce_index = [], []      # collect CE pairs then batch-score
        cand_all = []
        sims = Q @ D.T                    # (nq, N)
        for qi, q in enumerate(qids):
            cand = np.argpartition(-sims[qi], POOL)[:POOL]
            cand = cand[np.argsort(-sims[qi][cand])]
            cand_all.append(cand)
            for di in cand:
                ce_pairs.append([qtext[q], dtexts[di][:2000]]); ce_index.append((qi, di))
        print(f"[{dom}] CE scoring {len(ce_pairs)} pairs ...", flush=True)
        ce_scores = ce.predict(ce_pairs, batch_size=256, show_progress_bar=False)
        ce_map = {}
        for (qi, di), s in zip(ce_index, ce_scores):
            ce_map[(qi, di)] = float(s)

        for qi, q in enumerate(qids):
            cand = cand_all[qi]; qtk = toks(qtext[q]); qset = set(qtk); qv = Q[qi]
            rows, labs = [], []
            for di in cand:
                dc = Counter(dtok[di]); dl = len(dtok[di])
                bm = sum(idf.get(w, 0) * dc[w] * 2.5 / (dc[w] + 1.5 * (1 - .75 + .75 * dl / avgdl)) for w in qset if w in dc)
                jac = len(qset & set(dtok[di])) / (len(qset | set(dtok[di])) + 1e-9)
                ov = len(qset & set(dtok[di])) / (len(qtk) + 1e-9)
                base = [float(sims[qi, di]), ce_map[(qi, di)], bm, jac, ov, math.log(1 + dl), math.log(1 + len(qtk))]
                inter = ((qv * D[di]) @ R).tolist()
                rows.append(base + inter)
                labs.append(qrels[q].get(docids[di], 0))
            feats_all.append(np.array(rows, np.float32)); labels_all.append(np.array(labs, np.float32))
            qptr.append(qptr[-1] + len(cand))
        F = feats_all[0].shape[1]
        X = np.concatenate(feats_all); y = np.concatenate(labels_all)
        np.savez_compressed(out, X=X, y=y, qptr=np.array(qptr), F=F,
                            fnames=np.array(["dense", "ce", "bm25", "jac", "ov", "logdl", "logql"] + [f"proj{i}" for i in range(PROJ)]))
        npos = sum((labels_all[i] > 0).any() for i in range(len(labels_all)))
        print(f"[{dom}] saved {len(qids)} queries ({npos} with a gold in pool), {X.shape[0]} pairs, F={F}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
