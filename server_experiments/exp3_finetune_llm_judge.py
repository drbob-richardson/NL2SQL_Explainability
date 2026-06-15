"""EXPERIMENT 3 (GPU): fine-tune a small GENERATIVE LLM as a SQL-correctness judge, and test
whether its reasoning TRANSFERS across schemas where the fine-tuned encoder (exp1) overfit.

exp1 finding: a fine-tuned ENCODER verifier is strong in-distribution (~0.78) but collapses on
unseen schemas (LODO ~0.67) -- it memorizes schema surface patterns. The frozen LLM judge's 0.77
is already a *transfer* number because it reasons. Hypothesis: fine-tuning a generative LLM judge
(LoRA) keeps that reasoning/generalization and beats the frozen judge on transfer.

Method: format (question + evidence + schema + SQL) as a prompt ending "Answer:", LoRA-fine-tune
the model to emit "Yes"/"No" (loss on the answer tokens only), and at eval read
P(Yes) = softmax over the " Yes"/" No" first-token logits. Report IN-DISTRIBUTION and
LEAVE-ONE-DB-OUT (transfer) AUROC -- same splits as exp1.

Self-contained: needs only data/verifier_data.jsonl (bundled). Extra deps: peft.
  pip install peft
  python exp3_finetune_llm_judge.py --smoke                                   # ~2-3 min sanity
  python exp3_finetune_llm_judge.py --mode indist --model Qwen/Qwen2.5-1.5B-Instruct
  python exp3_finetune_llm_judge.py --mode both --model Qwen/Qwen2.5-1.5B-Instruct --epochs 2

Output: results/exp3_judge_<tag>.json and .log
"""
from __future__ import annotations
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "verifier_data.jsonl")
RESULTS = os.path.join(HERE, "results")
INSTR = "You are a strict SQL reviewer. Decide if the SQL correctly answers the question."


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
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]; rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_rows():
    return [json.loads(l) for l in open(DATA)]


def head_tail(r):
    head = (f"{INSTR}\nSQL: {r['sql']}\nQuestion: {r['question']}\n"
            f"Evidence: {r['evidence']}\nSchema:\n")
    tail = "\nIs the SQL correct? Answer Yes or No.\nAnswer:"
    return head, r["schema"], tail


def build_ids(tok, r, max_len, answer_word=None):
    """Tokenize head + (truncated) schema + tail [+ answer]. SQL/question (head) and the
    'Answer:' (tail) are always preserved; only the schema tail is truncated."""
    head, schema, tail = head_tail(r)
    head_ids = tok(head, add_special_tokens=True).input_ids
    tail_ids = tok(tail, add_special_tokens=False).input_ids
    schema_ids = tok(schema, add_special_tokens=False).input_ids
    ans_ids = tok(" " + answer_word, add_special_tokens=False).input_ids if answer_word else []
    budget = max_len - len(head_ids) - len(tail_ids) - len(ans_ids) - 1
    if budget < 0:
        budget = 0
    schema_ids = schema_ids[:budget]
    prompt_ids = head_ids + schema_ids + tail_ids
    return prompt_ids, ans_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mode", default="both", choices=["indist", "lodo", "both"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-dbs", type=int, default=99)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    tag = args.model.split("/")[-1] + ("_smoke" if args.smoke else "")
    sys.stdout = Tee(os.path.join(RESULTS, f"exp3_judge_{tag}.log"))
    import numpy as np, torch
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from peft import LoraConfig, get_peft_model
    except Exception:
        print("FATAL: peft not installed -> pip install peft"); raise
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]

    rows = load_rows()
    if args.smoke:
        rows = rows[:200]
    dbs = sorted({r["db_id"] for r in rows})
    print(f"loaded {len(rows)} rows across {len(dbs)} dbs; yes_id={yes_id} no_id={no_id}")

    def fresh_model():
        m = AutoModelForCausalLM.from_pretrained(args.model)
        m = m.to(dev)
        if dev == "cuda":
            m = m.to(torch.bfloat16)
        lora = LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=2 * args.lora_r,
                          lora_dropout=0.05, target_modules="all-linear")
        return get_peft_model(m, lora)

    def train_eval(train_rows, test_rows, tagx):
        model = fresh_model()
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        steps = 0
        for ep in range(args.epochs):
            perm = np.random.RandomState(ep).permutation(len(train_rows))
            opt.zero_grad()
            for bi in range(0, len(train_rows), args.bs):
                batch = [train_rows[j] for j in perm[bi:bi + args.bs]]
                seqs, labs = [], []
                for r in batch:
                    word = "Yes" if r["label"] == 1 else "No"
                    pids, aids = build_ids(tok, r, args.max_len, word)
                    ids = pids + aids
                    lab = [-100] * len(pids) + aids
                    seqs.append(torch.tensor(ids)); labs.append(torch.tensor(lab))
                inp = pad_sequence(seqs, batch_first=True, padding_value=tok.pad_token_id).to(dev)
                lab = pad_sequence(labs, batch_first=True, padding_value=-100).to(dev)
                att = (inp != tok.pad_token_id).long()
                out = model(input_ids=inp, attention_mask=att, labels=lab)
                (out.loss / args.grad_accum).backward()
                steps += 1
                if steps % args.grad_accum == 0:
                    opt.step(); opt.zero_grad()
                if args.smoke and steps >= 6:
                    break
            if args.smoke:
                break
            print(f"    [{tagx}] epoch {ep+1}/{args.epochs} done ({steps} steps)")
        opt.step(); opt.zero_grad()

        model.eval()
        scores, ys = [], []
        with torch.no_grad():
            for r in test_rows:
                pids, _ = build_ids(tok, r, args.max_len, None)
                inp = torch.tensor([pids]).to(dev)
                logits = model(input_ids=inp).logits[0, -1]
                ly, ln = logits[yes_id].float().item(), logits[no_id].float().item()
                m = max(ly, ln)
                p = np.exp(ly - m) / (np.exp(ly - m) + np.exp(ln - m))
                scores.append(p); ys.append(r["label"])
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        return auroc(scores, ys)

    result = {"model": args.model, "epochs": args.epochs, "n_rows": len(rows),
              "baselines": {"frozen_verifier_gpt4o": 0.770, "encoder_indist": 0.785,
                            "encoder_lodo": 0.670}}

    if args.mode in ("indist", "both"):
        qids = sorted({(r["db_id"], r["question_id"]) for r in rows})
        rng = np.random.RandomState(0); rng.shuffle(qids)
        test_q = set(qids[:len(qids) // 5])
        tr = [r for r in rows if (r["db_id"], r["question_id"]) not in test_q]
        te = [r for r in rows if (r["db_id"], r["question_id"]) in test_q]
        t0 = time.time()
        result["indist_auroc"] = train_eval(tr, te, "indist")
        print(f"\n  IN-DISTRIBUTION AUROC = {result['indist_auroc']:.3f}   ({time.time()-t0:.0f}s)")

    if args.mode in ("lodo", "both"):
        per = {}
        for held in dbs[:args.max_dbs]:
            tr = [r for r in rows if r["db_id"] != held]
            te = [r for r in rows if r["db_id"] == held]
            if not te or len(set(r["label"] for r in te)) < 2:
                continue
            per[held] = train_eval(tr, te, f"lodo:{held}")
            print(f"    LODO held-out {held}: AUROC {per[held]:.3f}")
        result["lodo_per_db"] = per
        result["lodo_mean_auroc"] = float(np.mean(list(per.values()))) if per else None
        print(f"\n  LEAVE-ONE-DB-OUT (transfer) mean AUROC = {result['lodo_mean_auroc']}")

    print("\n  VERDICT: universal-verifier bet wins if LODO > 0.77 (beats frozen gpt-4o judge AND")
    print("  the fine-tuned encoder's 0.67 transfer). If LODO ~ 0.67 too, generative judges also")
    print("  overfit schemas and the universal verifier is just the frozen LLM.")
    json.dump(result, open(os.path.join(RESULTS, f"exp3_judge_{tag}.json"), "w"), indent=2)
    print(f"\n  wrote results/exp3_judge_{tag}.json")


if __name__ == "__main__":
    main()
