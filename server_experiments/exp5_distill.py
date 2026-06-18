"""EXPERIMENT 5 (GPU): reasoning distillation. Does training a small verifier to REASON (not just
emit a verdict) improve cross-schema transfer?

Two LoRA fine-tunes of the same small model on the same pairs (data/distill_data.jsonl, teacher
rationales + ground-truth verdicts), evaluated leave-one-DB-out:
  verdict-only : target "Verdict: Yes/No"
  reasoning    : target "Reasoning: <teacher rationale>\\nVerdict: Yes/No"
At eval, the reasoning model GENERATES its own reasoning, then we read P(Yes) at the verdict token;
the verdict-only model is scored directly. If reasoning > verdict-only on transfer, the verifier's
generalization comes from reasoning (distillable), not scale alone. Compare to exp3 (verdict-only
LoRA, LODO 0.659) and the frozen judge (0.77).

NOTE: this is the most complex script and its eval uses generation. Run --smoke first.
  python exp5_distill.py --smoke
  python exp5_distill.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 2 --eval-cap 200
"""
from __future__ import annotations
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "distill_data.jsonl")
RESULTS = os.path.join(HERE, "results")
INSTR = "Decide whether the SQL correctly answers the question."


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
    def __getattr__(self, n):
        return getattr(self.stdout, n)


def auroc(s, y):
    import numpy as np
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def prompt_text(r, tok, max_len):
    head = f"{INSTR}\nSQL: {r['sql']}\nQuestion: {r['question']}\nEvidence: {r.get('evidence','')}\nSchema:\n"
    tail = "\nAnswer:"
    hid = tok(head, add_special_tokens=True).input_ids
    tid = tok(tail, add_special_tokens=False).input_ids
    sid = tok(r["schema"], add_special_tokens=False).input_ids
    budget = max(0, max_len - len(hid) - len(tid) - 48)
    return hid + sid[:budget] + tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-cap", type=int, default=200, help="max eval examples per held-out db")
    ap.add_argument("--max-dbs", type=int, default=99)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    tag = args.model.split("/")[-1] + ("_smoke" if args.smoke else "")
    sys.stdout = Tee(os.path.join(RESULTS, f"exp5_distill_{tag}.log"))
    import numpy as np, torch
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    vtail = tok(" Verdict:", add_special_tokens=False).input_ids

    rows = [json.loads(l) for l in open(DATA)]
    if args.smoke:
        rows = rows[:120]
    dbs = sorted({r["db_id"] for r in rows})[:args.max_dbs]
    print(f"{len(rows)} rationale rows across {len(dbs)} dbs")

    def target_ids(r, reasoning):
        if reasoning:
            s = f" Reasoning: {r['rationale']}\nVerdict: " + ("Yes" if r["label"] else "No")
        else:
            s = " Verdict: " + ("Yes" if r["label"] else "No")
        return tok(s, add_special_tokens=False).input_ids

    def fresh():
        m = AutoModelForCausalLM.from_pretrained(args.model).to(dev)
        if dev == "cuda":
            m = m.to(torch.bfloat16)
        return get_peft_model(m, LoraConfig(task_type="CAUSAL_LM", r=args.lora_r,
                              lora_alpha=2 * args.lora_r, lora_dropout=0.05, target_modules="all-linear"))

    def train(train_rows, reasoning):
        model = fresh(); model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        steps = 0
        for ep in range(args.epochs):
            perm = np.random.RandomState(ep).permutation(len(train_rows)); opt.zero_grad()
            for bi in range(0, len(train_rows), args.bs):
                seqs, labs = [], []
                for r in (train_rows[j] for j in perm[bi:bi + args.bs]):
                    pid = prompt_text(r, tok, args.max_len); aid = target_ids(r, reasoning)
                    seqs.append(torch.tensor(pid + aid)); labs.append(torch.tensor([-100] * len(pid) + aid))
                inp = pad_sequence(seqs, batch_first=True, padding_value=tok.pad_token_id).to(dev)
                lab = pad_sequence(labs, batch_first=True, padding_value=-100).to(dev)
                out = model(input_ids=inp, attention_mask=(inp != tok.pad_token_id).long(), labels=lab)
                (out.loss / args.grad_accum).backward(); steps += 1
                if steps % args.grad_accum == 0:
                    opt.step(); opt.zero_grad()
                if args.smoke and steps >= 6:
                    break
            if args.smoke:
                break
        opt.step(); opt.zero_grad(); model.eval()
        return model

    def score_verdict(model, ids):
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]).to(dev)).logits[0, -1]
        ly, ln = logits[yes_id].float().item(), logits[no_id].float().item()
        m = max(ly, ln)
        return float(np.exp(ly - m) / (np.exp(ly - m) + np.exp(ln - m)))

    def evaluate(model, test_rows, reasoning):
        scores, ys = [], []
        for r in test_rows:
            pid = prompt_text(r, tok, args.max_len)
            if not reasoning:
                scores.append(score_verdict(model, pid + vtail))
            else:
                with torch.no_grad():
                    gen = model.generate(input_ids=torch.tensor([pid]).to(dev), max_new_tokens=64,
                                         do_sample=False, pad_token_id=tok.pad_token_id)[0].tolist()
                new = gen[len(pid):]
                # cut generated reasoning at "Verdict:" if present, else use all of it
                txt = tok.decode(new)
                reason = txt.split("Verdict:")[0]
                rid = tok(" Reasoning: " + reason.strip(), add_special_tokens=False).input_ids
                scores.append(score_verdict(model, pid + rid + vtail))
            ys.append(r["label"])
        return auroc(scores, ys)

    res = {"model": args.model, "baselines": {"exp3_verdict_only_lodo": 0.659, "frozen_gpt4o": 0.770}, "per_db": {}}
    vo, rr = [], []
    for held in dbs:
        te = [r for r in rows if r["db_id"] == held]
        if len(set(r["label"] for r in te)) < 2:
            continue
        te = te[:args.eval_cap]
        tr = [r for r in rows if r["db_id"] != held]
        t0 = time.time()
        m1 = train(tr, False); a_vo = evaluate(m1, te, False); del m1
        if dev == "cuda":
            torch.cuda.empty_cache()
        m2 = train(tr, True); a_rr = evaluate(m2, te, True); del m2
        if dev == "cuda":
            torch.cuda.empty_cache()
        res["per_db"][held] = {"verdict_only": a_vo, "reasoning": a_rr}
        vo.append(a_vo); rr.append(a_rr)
        print(f"  {held:<26} verdict-only {a_vo:.3f}   reasoning {a_rr:.3f}   ({time.time()-t0:.0f}s)")
    res["mean_verdict_only"] = float(np.mean(vo)) if vo else None
    res["mean_reasoning"] = float(np.mean(rr)) if rr else None
    print(f"\n  MEAN transfer:  verdict-only {res['mean_verdict_only']:.3f}   "
          f"reasoning {res['mean_reasoning']:.3f}")
    print("  Read: reasoning > verdict-only means distilled reasoning improves transfer.")
    json.dump(res, open(os.path.join(RESULTS, f"exp5_distill_{tag}.json"), "w"), indent=2)
    print(f"\n  wrote results/exp5_distill_{tag}.json")


if __name__ == "__main__":
    main()
