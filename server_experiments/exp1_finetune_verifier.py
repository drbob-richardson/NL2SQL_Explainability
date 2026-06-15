"""EXPERIMENT 1 (GPU): fine-tune a transformer verifier for SQL execution-correctness, and test
whether it TRANSFERS to unseen schemas (the open question the cheap-feature probe could not answer).

Input text per example: question + evidence + schema + candidate SQL  ->  binary {correct?}.
We fit a sequence-classification model and report:
  - IN-DISTRIBUTION AUROC (random split, grouped by question)
  - LEAVE-ONE-DB-OUT AUROC (train on other dbs, test on held-out db) == cross-schema transfer
Baselines for reference (BIRD, gpt-4o-mini generations): frozen gpt-4o-mini verifier 0.724,
frozen gpt-4o verifier 0.770 (both zero-shot/transfer), self-consistency 0.616, cheap-feature
trained classifier 0.768 in-dist / 0.661 transfer. The bet succeeds if this fine-tune beats ~0.77
ON THE LODO (transfer) metric.

Self-contained: needs only data/verifier_data.jsonl (bundled). No DB, embeddings, or API.

  python exp1_finetune_verifier.py --smoke              # ~1 min, verifies it runs end-to-end
  python exp1_finetune_verifier.py --mode indist        # in-distribution only
  python exp1_finetune_verifier.py --mode lodo          # transfer (one training per db)
  python exp1_finetune_verifier.py --mode both --model roberta-base --epochs 3

Output: prints to stdout AND writes results/exp1_verifier_<tag>.json and results/exp1_verifier_<tag>.log
"""
from __future__ import annotations
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "verifier_data.jsonl")
RESULTS = os.path.join(HERE, "results")


class Tee:
    """Mirror stdout to a log file so output is easy to bring back."""
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
        # delegate anything else (encoding, etc.) to the real stdout
        return getattr(self.stdout, name)


def auroc(scores, labels):
    import numpy as np
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_rows():
    rows = [json.loads(l) for l in open(DATA)]
    for r in rows:
        # SQL + question FIRST so they survive truncation; schema (long) is truncated from the tail.
        r["text"] = (f"SQL: {r['sql']}\nQuestion: {r['question']}\nEvidence: {r['evidence']}\n"
                     f"Schema:\n{r['schema']}")
    return rows


def train_eval(train_rows, test_rows, args, tag):
    """Fine-tune on train_rows, return (auroc, test_scores). Manual torch loop for portability."""
    import numpy as np, torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(dev)

    def encode(batch):
        return tok([r["text"] for r in batch], truncation=True, max_length=args.max_len,
                   padding=True, return_tensors="pt")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    steps = 0
    for ep in range(args.epochs):
        perm = np.random.RandomState(ep).permutation(len(train_rows))
        for i in range(0, len(train_rows), args.bs):
            batch = [train_rows[j] for j in perm[i:i + args.bs]]
            enc = {k: v.to(dev) for k, v in encode(batch).items()}
            labels = torch.tensor([r["label"] for r in batch]).to(dev)
            out = model(**enc, labels=labels)
            out.loss.backward(); opt.step(); opt.zero_grad()
            steps += 1
            if args.smoke and steps >= 10:
                break
        if args.smoke:
            break
        print(f"    [{tag}] epoch {ep+1}/{args.epochs} done ({steps} steps)")

    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(test_rows), args.bs):
            batch = test_rows[i:i + args.bs]
            enc = {k: v.to(dev) for k, v in encode(batch).items()}
            p = torch.softmax(model(**enc).logits, -1)[:, 1].cpu().numpy()
            scores.extend(p.tolist())
    y = [r["label"] for r in test_rows]
    return auroc(scores, y), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base",
                    help="HF model id; long-context options: answerdotai/ModernBERT-base")
    ap.add_argument("--mode", default="both", choices=["indist", "lodo", "both"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-dbs", type=int, default=99, help="cap #held-out dbs in lodo (for speed)")
    ap.add_argument("--smoke", action="store_true", help="tiny/fast run to verify it works")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    tag = (args.model.split("/")[-1] + ("_smoke" if args.smoke else ""))
    sys.stdout = Tee(os.path.join(RESULTS, f"exp1_verifier_{tag}.log"))
    import numpy as np
    try:
        import torch
        print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    except Exception as e:
        print("FATAL: torch/transformers not installed. pip install -r requirements.txt"); raise

    rows = load_rows()
    if args.smoke:
        rows = rows[:400]
    dbs = sorted({r["db_id"] for r in rows})
    print(f"loaded {len(rows)} rows across {len(dbs)} dbs; model={args.model} epochs={args.epochs}")
    result = {"model": args.model, "epochs": args.epochs, "n_rows": len(rows),
              "baselines": {"frozen_verifier_mini": 0.724, "frozen_verifier_gpt4o": 0.770,
                            "self_consistency": 0.616, "cheap_clf_indist": 0.768,
                            "cheap_clf_lodo": 0.661}}

    if args.mode in ("indist", "both"):
        # split by QUESTION (no question across train/test)
        qids = sorted({(r["db_id"], r["question_id"]) for r in rows})
        rng = np.random.RandomState(0); rng.shuffle(qids)
        test_q = set(qids[:len(qids) // 5])
        tr = [r for r in rows if (r["db_id"], r["question_id"]) not in test_q]
        te = [r for r in rows if (r["db_id"], r["question_id"]) in test_q]
        t0 = time.time()
        au, _ = train_eval(tr, te, args, "indist")
        result["indist_auroc"] = au
        print(f"\n  IN-DISTRIBUTION AUROC = {au:.3f}   ({time.time()-t0:.0f}s)")

    if args.mode in ("lodo", "both"):
        per_db = {}
        for held in dbs[:args.max_dbs]:
            tr = [r for r in rows if r["db_id"] != held]
            te = [r for r in rows if r["db_id"] == held]
            if not te or len(set(r["label"] for r in te)) < 2:
                continue
            au, _ = train_eval(tr, te, args, f"lodo:{held}")
            per_db[held] = au
            print(f"    LODO held-out {held}: AUROC {au:.3f}")
        result["lodo_per_db"] = per_db
        result["lodo_mean_auroc"] = float(np.mean(list(per_db.values()))) if per_db else None
        print(f"\n  LEAVE-ONE-DB-OUT (transfer) mean AUROC = {result['lodo_mean_auroc']}")

    print("\n  VERDICT: bet succeeds if LODO mean AUROC > 0.77 (beats frozen gpt-4o verifier on")
    print("  transfer). If LODO << indist, the fine-tune overfits schemas like the cheap classifier.")
    json.dump(result, open(os.path.join(RESULTS, f"exp1_verifier_{tag}.json"), "w"), indent=2)
    print(f"\n  wrote results/exp1_verifier_{tag}.json")


if __name__ == "__main__":
    main()
