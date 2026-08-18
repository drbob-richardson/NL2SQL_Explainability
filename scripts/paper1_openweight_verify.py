"""TMLR revision #4 (broaden generators): does the verification-beats-black-box pattern hold on an OPEN-WEIGHT,
non-OpenAI generator? Uses the open-weight generations (data/bird_samples_qwen_coder.json, from
bird_openweight_finish.py) and the three judge caches (bird_verify.py --samples ...). $0.

  ./.venv/bin/python scripts/paper1_openweight_verify.py
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
from bnp_nl2sql.uq_baselines import structural_top_prob

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
    rng = np.random.RandomState(0); b = np.array(base, float); f = np.array(full, float); yy = np.array(y, int)
    n = len(yy); d = []
    for _ in range(nb):
        idx = rng.randint(0, n, n)
        if len(set(yy[idx])) > 1:
            d.append(auroc(f[idx], yy[idx]) - auroc(b[idx], yy[idx]))
    return np.mean(d), np.percentile(d, [2.5, 97.5])


def main():
    samp = json.load(open(os.path.join(ROOT, "data", "bird_samples_qwen_coder.json")))
    keys = list(samp)
    def modal_idx(e):
        return e["samples"].index(Counter(e["samples"]).most_common(1)[0][0])
    y = np.array([samp[k]["ok"][modal_idx(samp[k])] for k in keys], int)
    strsc = np.array([Counter(samp[k]["samples"]).most_common(1)[0][1] / len(samp[k]["samples"]) for k in keys])
    logp = np.array([samp[k]["logp"][modal_idx(samp[k])] for k in keys])
    struc = np.array([structural_top_prob(samp[k]["samples"]) for k in keys])

    print(f"OPEN-WEIGHT generator, n={len(y)}, modal accuracy {y.mean():.3f}")
    print(f"  black-box band:  string-SC {auroc(strsc, y):.3f}   structural-SC {auroc(struc, y):.3f}   "
          f"log-prob {auroc(logp, y):.3f}   (verifier must clear the strongest of these)\n")
    J = {"gpt-4o-mini (same-family-ish)": "data/bird_verify_gen-qwen_coder.json",
         "gpt-4o (stronger)": "data/bird_verify_gen-qwen_coder_gpt_4o.json",
         "Claude-Sonnet-4.6 (independent)": "data/bird_verify_gen-qwen_coder_anthropic_claude_sonnet_4_6_verbal.json"}
    for name, path in J.items():
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            print(f"  {name:<32} [judge cache missing: run bird_verify.py --samples ...]"); continue
        j = json.load(open(p)); v = np.array([j.get(k, 0.5) for k in keys])
        m, (lo, hi) = boot(strsc, v, y); sig = "SIGNIFICANT" if lo > 0 else "n.s."
        print(f"  {name:<32} judge AUROC {auroc(v, y):.3f}   dAUROC vs string-SC {m:+.3f} [{lo:+.3f},{hi:+.3f}]  {sig}")
    print("\nReading: if a verifier's dAUROC-vs-SC interval excludes 0 on this non-OpenAI generator, the "
          "verification-beats-black-box finding generalizes beyond the OpenAI family (R1/hFAr's main ask).")


if __name__ == "__main__":
    main()
