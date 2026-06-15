"""Evidence for two reviewer points (no API):
  (1) Correct the open-world label: open-world FAILURE = no execution-correct query exists
      among the K samples (airtight under execution accuracy), not merely "gold canonical
      structure unseen". Re-measure discovery AUROC for this corrected label.
  (2) H provenance: build H from Spider TRAIN single-table golds (DISJOINT databases from
      dev) and evaluate on dev -- does the structural prior generalize across databases, or
      just memorize the benchmark's dev query frequencies?
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from bnp_nl2sql import model_a_posterior, sql_to_graph
from bnp_nl2sql.calibrate import aurc
from bnp_nl2sql.execeval import exec_match
from bnp_nl2sql.fit import empirical_base, fit_pyp_partitions
from bnp_nl2sql.uq_baselines import structural_top_prob
from compare_baselines import spider_records
from spider_benchmark import is_single_table


def sk(s):
    try: return sql_to_graph(s).skeleton_key()
    except Exception: return "<u>"
def ck(s):
    try: return sql_to_graph(s).canonical_key()
    except Exception: return "<u>"
def auroc(scores, labels):
    pos=[s for s,y in zip(scores,labels) if y]; neg=[s for s,y in zip(scores,labels) if not y]
    if not pos or not neg: return float("nan")
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))


def train_H():
    from datasets import load_dataset
    ds = load_dataset("xlangai/spider", split="train")
    cks, sks, dbs = [], [], set()
    for r in ds:
        if is_single_table(r["query"]):
            cks.append(ck(r["query"])); sks.append(sk(r["query"])); dbs.add(r["db_id"])
    return empirical_base(cks), empirical_base(sks), len(cks), dbs


def main():
    recs = spider_records()
    fit = fit_pyp_partitions([[sk(s) for s in samp] for samp,_,_ in recs])
    fitf = fit_pyp_partitions([[ck(s) for s in samp] for samp,_,_ in recs])

    # dev base measures (in-sample)
    Hf = empirical_base([ck(g) for _,g,_ in recs]); Hs = empirical_base([sk(g) for _,g,_ in recs])
    # train base measures (disjoint databases)
    Hf_tr, Hs_tr, n_tr, train_dbs = train_H()

    def eval_with(Hf_b, Hs_b, tag):
        ours, base, disc, correct, gold_unseen, ow_fail = [],[],[],[],[],[]
        for samples, gold, conn in recs:
            p = model_a_posterior(samples, discount=fit.discount, concentration=fit.concentration,
                                  skeleton_base=lambda s: Hs_b.get(s,0.0), full_discount=fitf.discount,
                                  full_concentration=fitf.concentration, full_base=lambda s: Hf_b.get(s,0.0))
            mq = p.map_query()
            try: ok = exec_match(mq, gold, conn) if mq else False
            except Exception: ok = False
            # corrected open-world label: NO execution-correct query among the K samples
            any_correct = False
            for s in samples:
                try:
                    if exec_match(s, gold, conn): any_correct = True; break
                except Exception: pass
            correct.append(ok)
            ours.append(p.confidence()*(1-p.full_discovery_probability))
            base.append(structural_top_prob(samples))
            disc.append(p.full_discovery_probability)
            gold_unseen.append(ck(gold) not in {ck(s) for s in samples})
            ow_fail.append(not any_correct)
        return dict(ours=ours, base=base, disc=disc, correct=correct,
                    gold_unseen=gold_unseen, ow_fail=ow_fail)

    print(f"Spider dev single-table: n={len(recs)}")
    r = eval_with(Hf, Hs, "in-sample")
    n_gu = sum(r["gold_unseen"]); n_ow = sum(r["ow_fail"])
    print(f"\n(1) Open-world label correction:")
    print(f"  gold-canonical-unseen:                 {n_gu} ({n_gu/len(recs):.0%})")
    print(f"  NO execution-correct sample (correct): {n_ow} ({n_ow/len(recs):.0%})  <- airtight label")
    print(f"  discovery AUROC for OLD label (gold_unseen):       {auroc(r['disc'], r['gold_unseen']):.3f}")
    print(f"  discovery AUROC for CORRECTED label (ow_fail):     {auroc(r['disc'], r['ow_fail']):.3f}")
    print(f"  baseline 1-top_prob AUROC for corrected label:     {auroc([1-x for x in r['base']], r['ow_fail']):.3f}")

    print(f"\n(2) H provenance (train: {n_tr} single-table queries over {len(train_dbs)} DBs,")
    print(f"    DISJOINT from dev databases):")
    # overlap of dev structures with train H
    dev_ck = [ck(g) for _,g,_ in recs]
    cover = sum(Hf_tr.get(k,0)>0 for k in dev_ck)/len(dev_ck)
    print(f"  fraction of dev canonical structures present in TRAIN H: {cover:.2f}")
    rt = eval_with(Hf_tr, Hs_tr, "train-only")
    print(f"  {'H source':22s} {'AURC ours':>10} {'open-world AUROC (corrected)':>28}")
    print(f"  {'uniform H (no base)':22s} {'0.172':>10} {'(chance)':>28}")
    print(f"  {'in-sample dev H':22s} {aurc(r['ours'],r['correct']):>10.3f} {auroc(r['disc'],r['ow_fail']):>28.3f}")
    print(f"  {'train-only H (xDB)':22s} {aurc(rt['ours'],rt['correct']):>10.3f} {auroc(rt['disc'],rt['ow_fail']):>28.3f}")


if __name__ == "__main__":
    main()
