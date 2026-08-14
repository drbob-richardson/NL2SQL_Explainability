"""TMLR revision #4: verifier vs self-consistency on the GPT-4o generator's outputs.

Uses the GPT-4o generations (data/bird_samples_gpt_4o.json) and the three judge caches produced by
bird_verify.py --samples data/bird_samples_gpt_4o.json (gpt-4o-mini, gpt-4o, Claude-Sonnet-4.6).
Finding: same-provider OpenAI judges do NOT clear self-consistency on GPT-4o's harder-to-verify
generations, but the independent-provider Claude judge does -- confirming that a strong generator
needs a judge that is independent of, or stronger than, it.

  ./.venv/bin/python scripts/paper1_gen4o_verify.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot(base, full, y, nb=2000):
    rng = np.random.RandomState(0); b = np.array(base); f = np.array(full); yy = np.array(y); n = len(yy); d = []
    for _ in range(nb):
        idx = rng.randint(0, n, n)
        if len(set(yy[idx])) > 1:
            d.append(auroc(f[idx], yy[idx]) - auroc(b[idx], yy[idx]))
    return np.mean(d), np.percentile(d, [2.5, 97.5])


def main():
    samp = json.load(open(os.path.join(ROOT, "data", "bird_samples_gpt_4o.json")))
    keys = list(samp.keys())
    y = np.array([samp[k]["ok"][samp[k]["samples"].index(Counter(samp[k]["samples"]).most_common(1)[0][0])]
                  for k in keys], int)
    strsc = np.array([Counter(samp[k]["samples"]).most_common(1)[0][1] / len(samp[k]["samples"]) for k in keys])
    J = {"gpt-4o-mini (same provider)": "data/bird_verify_gen-gpt_4o.json",
         "gpt-4o (same provider)": "data/bird_verify_gen-gpt_4o_gpt_4o.json",
         "Claude-Sonnet-4.6 (independent)": "data/bird_verify_gen-gpt_4o_anthropic_claude_sonnet_4_6_verbal.json"}
    print(f"GPT-4o generator, n={len(y)}, modal accuracy {y.mean():.3f}, string-SC AUROC {auroc(strsc, y):.3f}\n")
    for name, path in J.items():
        j = json.load(open(os.path.join(ROOT, path))); v = np.array([j.get(k, 0.5) for k in keys])
        m, (lo, hi) = boot(strsc, v, y); sig = "SIGNIFICANT" if lo > 0 else "n.s."
        print(f"  {name:<32} judge AUROC {auroc(v, y):.3f}   dAUROC vs SC {m:+.3f} [{lo:+.3f},{hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()
