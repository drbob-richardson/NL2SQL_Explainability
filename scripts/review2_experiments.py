"""Reviewer-2 decisive experiments (no API), reported with TIE-ROBUST metrics.

#4 Unanimous-only risk separation: among top_prob=1.0 questions (no self-consistency signal),
   does the structural prior separate risk? (error rate by PY-confidence half).
#6 Is the PY posterior needed for ranking, or is simple prior-weighting enough?
   compare tie-robust AUROC(correct) for: top_prob, H(MAP), top_prob*H(MAP), PY-confidence.
#9 Permuted-H placebo: shuffle H across structures -> should collapse to baseline.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from bnp_nl2sql import sql_to_graph, model_a_posterior
from bnp_nl2sql.execeval import exec_match
from bnp_nl2sql.fit import empirical_base, fit_pyp_partitions
from bnp_nl2sql.uq_baselines import structural_top_prob
from compare_baselines import spider_records

def ck(s):
    try: return sql_to_graph(s).canonical_key()
    except Exception: return "<u>"
def auroc(sc, lab):
    pos=[s for s,y in zip(sc,lab) if y]; neg=[s for s,y in zip(sc,lab) if not y]
    if not pos or not neg: return float("nan")
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))

def main():
    recs = spider_records()
    Hf = empirical_base([ck(g) for _,g,_ in recs])
    fit = fit_pyp_partitions([[ck(s) for s in samp] for samp,_,_ in recs])
    # permuted H: same values, shuffled across keys (deterministic shuffle, no RNG dependence)
    keys=list(Hf); vals=[Hf[k] for k in keys]; vals_rot=vals[7:]+vals[:7]
    Hperm={k:v for k,v in zip(keys, vals_rot)}

    rows=[]
    for samp,gold,conn in recs:
        pH=model_a_posterior(samp,discount=fit.discount,concentration=fit.concentration,
            full_discount=fit.discount,full_concentration=fit.concentration,full_base=lambda s:Hf.get(s,0.0))
        pP=model_a_posterior(samp,discount=fit.discount,concentration=fit.concentration,
            full_discount=fit.discount,full_concentration=fit.concentration,full_base=lambda s:Hperm.get(s,0.0))
        mq=pH.map_query()
        try: ok=exec_match(mq,gold,conn) if mq else False
        except Exception: ok=False
        rows.append(dict(correct=ok, top=structural_top_prob(samp), Hmap=Hf.get(ck(mq),0.0),
            py=pH.confidence()*(1-pH.full_discovery_probability),
            py_perm=pP.confidence()*(1-pP.full_discovery_probability),
            unanimous=(pH.pyp_full.K==1)))
    c=[r["correct"] for r in rows]

    print(f"n={len(rows)}, exec acc={sum(c)/len(c):.3f}\n")
    print("#4 UNANIMOUS-ONLY (top_prob=1.0; self-consistency has no signal):")
    uni=[r for r in rows if r["unanimous"]]
    uni.sort(key=lambda r:-r["py"])
    half=len(uni)//2
    top_err=1-sum(r["correct"] for r in uni[:half])/half
    bot_err=1-sum(r["correct"] for r in uni[half:])/(len(uni)-half)
    print(f"  {len(uni)} unanimous questions, overall error {1-sum(r['correct'] for r in uni)/len(uni):.3f}")
    print(f"  top-half by PY/H confidence:    error {top_err:.3f}")
    print(f"  bottom-half by PY/H confidence: error {bot_err:.3f}")
    print(f"  => structural prior separates risk within the saturated bucket: {bot_err-top_err:+.3f}")

    print("\n#6 IS THE PY POSTERIOR NEEDED FOR RANKING? (tie-robust AUROC for correctness):")
    print(f"  top_prob alone            : {auroc([r['top'] for r in rows],c):.3f}")
    print(f"  H(MAP) alone              : {auroc([r['Hmap'] for r in rows],c):.3f}")
    print(f"  top_prob * H(MAP)         : {auroc([r['top']*r['Hmap'] for r in rows],c):.3f}")
    print(f"  PY posterior (full)       : {auroc([r['py'] for r in rows],c):.3f}")

    print("\n#9 PERMUTED-H PLACEBO (shuffle H across structures):")
    print(f"  PY with true H   : {auroc([r['py'] for r in rows],c):.3f}")
    print(f"  PY with permuted H: {auroc([r['py_perm'] for r in rows],c):.3f}  (should collapse to ~baseline 0.609)")


if __name__ == "__main__":
    main()
