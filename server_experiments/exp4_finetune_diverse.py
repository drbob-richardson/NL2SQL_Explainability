"""EXPERIMENT 4 (GPU): does SCHEMA DIVERSITY in training fix the verifier's transfer failure?

exp1 trained the encoder verifier on 8 BIRD schemas and it did not transfer (leave-one-DB-out
AUROC ~0.67). Hypothesis: it overfit because it saw too few schemas. Here we add Spider's 20
schemas to the training pool and re-test transfer to held-out BIRD databases. For each held-out
BIRD db we train two models and compare:
  (a) BIRD-only  : train on the other 7 BIRD dbs        (reproduces exp1)
  (b) diverse    : train on the other 7 BIRD dbs + all 20 Spider dbs
both evaluated on the held-out BIRD db. If (b) > (a), schema diversity improves cross-schema transfer
and a universal trained verifier is within reach; if (b) ~ (a), the transfer wall is not a
data-diversity problem.

Self-contained: needs data/verifier_data_diverse.jsonl (bundled; built by build_diverse_verifier_data.py).

  python exp4_finetune_diverse.py --smoke
  python exp4_finetune_diverse.py --model answerdotai/ModernBERT-base --max-len 1024 --epochs 3
Output: results/exp4_diverse_<tag>.json and .log
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


def train_eval(train_rows, test_rows, args, tag):
    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def enc(batch):
        return tok([r["text"] for r in batch], truncation=True, max_length=args.max_len,
                   padding=True, return_tensors="pt")
    model.train(); steps = 0
    for ep in range(args.epochs):
        perm = np.random.RandomState(ep).permutation(len(train_rows))
        for i in range(0, len(train_rows), args.bs):
            batch = [train_rows[j] for j in perm[i:i + args.bs]]
            e = {k: v.to(dev) for k, v in enc(batch).items()}
            lab = torch.tensor([r["label"] for r in batch]).to(dev)
            out = model(**e, labels=lab); out.loss.backward(); opt.step(); opt.zero_grad()
            steps += 1
            if args.smoke and steps >= 8:
                break
        if args.smoke:
            break
    model.eval(); scores = []
    with torch.no_grad():
        for i in range(0, len(test_rows), args.bs):
            e = {k: v.to(dev) for k, v in enc(test_rows[i:i + args.bs]).items()}
            scores.extend(torch.softmax(model(**e).logits, -1)[:, 1].cpu().tolist())
    au = auroc(scores, [r["label"] for r in test_rows])
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
    ap.add_argument("--max-dbs", type=int, default=99)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    tag = args.model.split("/")[-1] + ("_smoke" if args.smoke else "")
    sys.stdout = Tee(os.path.join(RESULTS, f"exp4_diverse_{tag}.log"))
    import numpy as np
    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} model={args.model}")

    rows = load_rows()
    if args.smoke:
        # keep 2 bird dbs + a little spider for a fast end-to-end check
        bird = sorted({r["db_id"] for r in rows if r["source"] == "bird"})[:2]
        rows = [r for r in rows if (r["source"] == "bird" and r["db_id"] in bird) or r["source"] == "spider"][:600]
    bird_dbs = sorted({r["db_id"] for r in rows if r["source"] == "bird"})[:args.max_dbs]
    nsp = len({r["db_id"] for r in rows if r["source"] == "spider"})
    print(f"{len(rows)} rows; {len(bird_dbs)} BIRD held-out dbs; {nsp} Spider dbs in pool")

    res = {"model": args.model, "baselines": {"exp1_bird_only_lodo": 0.670}, "per_db": {}}
    a_only, a_div = [], []
    for held in bird_dbs:
        test = [r for r in rows if r["source"] == "bird" and r["db_id"] == held]
        if len(set(r["label"] for r in test)) < 2:
            continue
        tr_bird = [r for r in rows if r["source"] == "bird" and r["db_id"] != held]
        tr_div = [r for r in rows if not (r["source"] == "bird" and r["db_id"] == held)]
        t0 = time.time()
        au_b = train_eval(tr_bird, test, args, f"bird:{held}")
        au_d = train_eval(tr_div, test, args, f"div:{held}")
        res["per_db"][held] = {"bird_only": au_b, "diverse": au_d}
        a_only.append(au_b); a_div.append(au_d)
        print(f"  {held:<26} bird-only {au_b:.3f}   diverse {au_d:.3f}   ({time.time()-t0:.0f}s)")
    res["mean_bird_only"] = float(np.mean(a_only)) if a_only else None
    res["mean_diverse"] = float(np.mean(a_div)) if a_div else None
    print(f"\n  MEAN transfer to held-out BIRD db:  bird-only {res['mean_bird_only']:.3f}   "
          f"diverse {res['mean_diverse']:.3f}")
    print("  VERDICT: diverse > bird-only means schema diversity improves transfer (universal verifier")
    print("  within reach); diverse ~ bird-only means the transfer wall is not about data diversity.")
    json.dump(res, open(os.path.join(RESULTS, f"exp4_diverse_{tag}.json"), "w"), indent=2)
    print(f"\n  wrote results/exp4_diverse_{tag}.json")


if __name__ == "__main__":
    main()
