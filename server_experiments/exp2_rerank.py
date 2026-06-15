"""EXPERIMENT 2 (no GPU, no API): verification-guided reranking for ACCURACY.

From the K cached samples per question, pick ONE answer by different selection rules and measure
end execution accuracy. Tests whether reranking beats plain self-consistency (modal selection):
  modal        : most frequent SQL (self-consistency baseline)
  first        : first sample (greedy-ish baseline)
  best_logprob : sample with highest mean token logprob (white-box rerank)
  oracle       : is ANY sample correct (best-of-N upper bound / headroom)
This is the cheap, GPU-free precursor to verifier-guided generation: if even logprob reranking
lifts accuracy toward the oracle, a trained verifier as the reranker should lift it further.

Self-contained: needs only data/bird_samples.json (bundled).
  python exp2_rerank.py
Writes results/exp2_rerank.json and results/exp2_rerank.log
"""
from __future__ import annotations
import json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "data", "bird_samples.json")
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


def main():
    os.makedirs(RESULTS, exist_ok=True)
    sys.stdout = Tee(os.path.join(RESULTS, "exp2_rerank.log"))
    data = json.load(open(SAMPLES))
    items = list(data.values())
    n = len(items)

    def acc(selector):
        c = 0
        for e in items:
            i = selector(e)
            c += 1 if e["ok"][i] else 0
        return c / n

    def modal(e):
        mq = Counter(e["samples"]).most_common(1)[0][0]
        return e["samples"].index(mq)

    def first(e):
        return 0

    def best_logprob(e):
        lp = e.get("logp") or [0.0] * len(e["samples"])
        return max(range(len(lp)), key=lambda i: lp[i])

    def oracle(e):
        for i, ok in enumerate(e["ok"]):
            if ok:
                return i
        return 0

    res = {
        "n": n,
        "first": acc(first),
        "modal_self_consistency": acc(modal),
        "best_logprob": acc(best_logprob),
        "oracle_best_of_N": acc(oracle),
    }
    print(f"BIRD rerank (n={n}, K={len(items[0]['samples'])}):")
    print(f"  first sample            : {res['first']:.3f}")
    print(f"  modal (self-consistency): {res['modal_self_consistency']:.3f}")
    print(f"  best logprob (rerank)   : {res['best_logprob']:.3f}")
    print(f"  ORACLE best-of-N (ceil) : {res['oracle_best_of_N']:.3f}")
    lift = res["best_logprob"] - res["modal_self_consistency"]
    headroom = res["oracle_best_of_N"] - res["modal_self_consistency"]
    print(f"\n  logprob-rerank lift over self-consistency: {lift:+.3f}")
    print(f"  oracle headroom (max possible from rerank): {headroom:+.3f}")
    print("\nReading: headroom>0 means a perfect reranker would raise accuracy by that much; if even")
    print("logprob captures part of it, a trained verifier reranker (Exp 1 / verifier-guided gen)")
    print("is the lever to close the rest -- the accuracy payoff top venues reward.")
    json.dump(res, open(os.path.join(RESULTS, "exp2_rerank.json"), "w"), indent=2)
    print("\n  wrote results/exp2_rerank.json")


if __name__ == "__main__":
    main()
