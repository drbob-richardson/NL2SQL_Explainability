"""EXPERIMENT 6 (GPU): cross-benchmark transfer for the verifier.

Leave-one-DB-out (exp1/exp4) is a soft transfer test; this is a harder one. We train the verifier on
one benchmark and test on the other, both directions, using the bundled diverse dataset:
  BIRD -> Spider : train on all 8 BIRD schemas, test on all 20 Spider schemas
  Spider -> BIRD : train on all 20 Spider schemas, test on all 8 BIRD schemas
and report in-distribution (held-out split within each source) for reference. Large cross-benchmark
gaps would show the verifier learns benchmark/regime-specific surface form rather than a transferable
notion of correctness.

Self-contained: needs data/verifier_data_diverse.jsonl (bundled).
  python exp6_crossbench.py --smoke
  python exp6_crossbench.py --model answerdotai/ModernBERT-base --max-len 1024 --epochs 3
"""
from __future__ import annotations
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "verifier_data_diverse.jsonl")
RESULTS = os.path.join(HERE, "results")


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.stdout.flush(); self.f.flush()
    def isatty(self):
        return False
    def fileno(self):
        return self.stdout.fileno()
    def __getattr__(self, name):
        return getattr(self.stdout, name)


def auroc(scores, labels):
    import numpy as np
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); ranks = ((cs - cnt + cs + 1) / 2.0)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_rows():
    rows = [json.loads(l) for l in open(DATA)]
    for r in rows:
        r["text"] = (f"SQL: {r['sql']}\nQuestion: {r['question']}\nEvidence: {r.get('evidence','')}\n"
                     f"Schema:\n{r['schema']}")
    return rows


def train_eval(train_rows, test_rows, args):
    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def enc(b):
        return tok([r["text"] for r in b], truncation=True, max_length=args.max_len,
                   padding=True, return_tensors="pt")
    model.train(); steps = 0
    for ep in range(args.epochs):
        perm = np.random.RandomState(ep).permutation(len(train_rows))
        for i in range(0, len(train_rows), args.bs):
            b = [train_rows[j] for j in perm[i:i + args.bs]]
            e = {k: v.to(dev) for k, v in enc(b).items()}
            lab = torch.tensor([r["label"] for r in b]).to(dev)
            out = model(**e, labels=lab); out.loss.backward(); opt.step(); opt.zero_grad()
            steps += 1
            if args.smoke and steps >= 8:
                break
        if args.smoke:
            break
    model.eval(); sc = []
    with torch.no_grad():
        for i in range(0, len(test_rows), args.bs):
            e = {k: v.to(dev) for k, v in enc(test_rows[i:i + args.bs]).items()}
            sc.extend(torch.softmax(model(**e).logits, -1)[:, 1].cpu().tolist())
    au = auroc(sc, [r["label"] for r in test_rows])
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return au


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    tag = args.model.split("/")[-1] + ("_smoke" if args.smoke else "")
    sys.stdout = Tee(os.path.join(RESULTS, f"exp6_crossbench_{tag}.log"))
    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} model={args.model}")

    rows = load_rows()
    if args.smoke:
        import random
        random.Random(0).shuffle(rows); rows = rows[:600]
    bird = [r for r in rows if r["source"] == "bird"]
    spider = [r for r in rows if r["source"] == "spider"]
    print(f"BIRD rows {len(bird)}, Spider rows {len(spider)}")

    def half(rows_):
        qids = sorted({(r["source"], r["db_id"], r["question_id"]) for r in rows_})
        import numpy as np
        rng = np.random.RandomState(0); rng.shuffle(qids); test = set(qids[:len(qids) // 5])
        key = lambda r: (r["source"], r["db_id"], r["question_id"])
        return [r for r in rows_ if key(r) not in test], [r for r in rows_ if key(r) in test]

    res = {"model": args.model}
    t0 = time.time()
    btr, bte = half(bird); spr, spe = half(spider)
    res["bird_in_dist"] = train_eval(btr, bte, args)
    res["spider_in_dist"] = train_eval(spr, spe, args)
    res["bird_to_spider"] = train_eval(bird, spider, args)
    res["spider_to_bird"] = train_eval(spider, bird, args)
    print(f"\n  in-distribution: BIRD {res['bird_in_dist']:.3f}   Spider {res['spider_in_dist']:.3f}")
    print(f"  cross-benchmark: BIRD->Spider {res['bird_to_spider']:.3f}   "
          f"Spider->BIRD {res['spider_to_bird']:.3f}   ({time.time()-t0:.0f}s)")
    print("  Read: cross-benchmark << in-distribution means the verifier learns benchmark-specific")
    print("  surface form; cross-benchmark ~ in-distribution means it learns transferable correctness.")
    json.dump(res, open(os.path.join(RESULTS, f"exp6_crossbench_{tag}.json"), "w"), indent=2)
    print(f"\n  wrote results/exp6_crossbench_{tag}.json")


if __name__ == "__main__":
    main()
