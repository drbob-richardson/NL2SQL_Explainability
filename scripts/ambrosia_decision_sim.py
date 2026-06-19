"""Experiment 5: decision simulation (execute / clarify / abstain) on AMBROSIA. No API.

Demonstrates the *payoff* of the decision layer -- the white space (unified execute/clarify/abstain
Bayes objective) -- independent of the generation bottleneck, by comparing posteriors of different
quality as the clarification cost c varies.

Model. Each question has K valid gold interpretations (ambiguous: >=2 with materially different
results; control: 1). The user's intended reading is uniform over the K. Wrong answer costs 1,
abstain costs a, clarify costs c (then we must still *realize* the intended reading -- if it is not
reachable, the clarify fails and costs 1). A policy picks the action minimizing its BELIEVED expected
loss; we score ACTUAL expected loss under the true uniform intended.

Posteriors compared:
  - always-execute (modal) / always-clarify           : fixed baselines
  - Bayes-oracle      : knows it is ambiguous AND can realize every reading (ceiling)
  - Bayes-discovery   : knows it is ambiguous (right clarify decision) but realization coverage is
                        the *actual* sampled coverage (Exp 2b reality) -> clarify can still fail
  - Bayes-realistic   : uses the collapsed sampled posterior we actually get (probe 4b) -> often
                        overconfident-executes on ambiguous questions

  ./.venv/bin/python scripts/ambrosia_decision_sim.py
"""
from __future__ import annotations
import ast, csv, json, os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
csv.field_size_limit(10**7)
from bnp_nl2sql.execeval import open_db, run_sql

ROOT = os.path.join(os.path.dirname(__file__), "..")
ADIR = os.path.join(ROOT, "data", "ambrosia")
SAMPLES = os.path.join(ROOT, "data", "ambrosia_samples.json")
ABSTAIN_COST = 0.5


def db_path(db_file):
    rel = db_file[5:] if db_file.startswith("data/") else db_file
    return os.path.join(ADIR, rel)


def result_sig(conn, sql, timeout=4.0):
    timer = threading.Timer(timeout, conn.interrupt); timer.start()
    try:
        rows = run_sql(conn, sql)
        return tuple(sorted(repr(r) for r in rows)) if rows else ("<empty>",)
    except Exception:
        return None
    finally:
        timer.cancel()


def build():
    """Per-question primitives: K, sampled posterior over readings, sampled coverage."""
    samples = json.load(open(SAMPLES))
    rows = list(csv.DictReader(open(os.path.join(ADIR, "ambrosia.csv"))))
    meta = {f"{r['db_file']}||{r['question']}": r for r in rows}
    conns = {}
    out = []
    for key, e in samples.items():
        r = meta.get(key)
        if r is None:
            continue
        p = db_path(e["db_file"])
        if p not in conns:
            conns[p] = open_db(p)
        conn = conns[p]
        if e["is_ambiguous"]:
            try:
                golds = [g for g in ast.literal_eval(r["ambig_queries"]) if isinstance(g, str)]
            except Exception:
                continue
        else:
            golds = [r["gold_queries"]]
        gsigs = []
        for g in golds:
            s = result_sig(conn, g)
            if s is not None and s not in gsigs:
                gsigs.append(s)
        K = len(gsigs)
        if K == 0:
            continue
        # sampled posterior over readings + invalid mass
        ssigs = [result_sig(conn, s) for s in e["samples"]]
        ssigs = [s for s in ssigs if s is not None]
        n = len(ssigs) or 1
        cnt = Counter(ssigs)
        reach = [j for j, gs in enumerate(gsigs) if gs in cnt]            # reachable gold readings
        w = [cnt.get(gsigs[j], 0) / n for j in range(K)]                  # mass on each gold reading
        invalid = 1.0 - sum(w)
        modal_is_valid = (max(w) >= invalid) if w else False
        modal_w = max(w + [invalid])
        out.append(dict(K=K, is_amb=e["is_ambiguous"], reach=reach,
                        cov=len(reach) / K, modal_w=modal_w, modal_valid=modal_is_valid))
    return out


def actual_loss(action, K, c, r):
    """All policies share the SAME realization ability r (a property of the SQL generator) and the
    same uniform-intended truth. Policies differ ONLY in the action they choose -> isolates the
    value of the *decision*. execute: commit one reading, right w.p. (1/K) and realized w.p. r.
    clarify: pay c, realize the revealed reading w.p. r. abstain: fixed cost."""
    if action == "execute":
        return 1.0 - r / K
    if action == "clarify":
        return c + (1.0 - r)
    return ABSTAIN_COST


def choose(belief, q, c, r):
    K = q["K"]
    if belief == "oracle":            # knows the true ambiguity (K) -> optimal action given r, c
        losses = {"execute": 1 - r / K, "clarify": c + (1 - r), "abstain": ABSTAIN_COST}
    else:                             # realistic: collapsed sampled posterior -> overconfident execute
        losses = {"execute": 1.0 - q["modal_w"], "clarify": c + (1 - r), "abstain": ABSTAIN_COST}
    return min(losses, key=losses.get)


def main():
    Q = build()
    amb = [q for q in Q if q["is_amb"]]
    ctl = [q for q in Q if not q["is_amb"]]
    print(f"=== Experiment 5: execute/clarify/abstain decision simulation (n={len(Q)}; "
          f"{len(amb)} ambiguous, {len(ctl)} control) ===")
    print(f"  wrong-answer loss 1.0, abstain cost {ABSTAIN_COST}. r = realization ability (prob a")
    print(f"  targeted reading's SQL is produced correctly); policies share r and differ only in action.")
    print(f"  realistic-policy mean modal confidence (sampled): amb {np.mean([q['modal_w'] for q in amb]):.2f}, "
          f"ctl {np.mean([q['modal_w'] for q in ctl]):.2f}\n")

    for r in (1.0, 0.3):
        tag = "perfect realization (value-of-discovery ceiling)" if r == 1.0 else "today's realization (Exp 2b ~0.3)"
        print(f"################ r = {r}  — {tag} ################")
        for c in (0.1, 0.3, 0.5):
            print(f"--- clarification cost c = {c} ---")
            print(f"  {'policy':<20}{'avg loss':>10}{'clar% amb':>11}{'clar% ctl':>11}")
            for name, act in (("always-execute", "execute"), ("always-clarify", "clarify")):
                loss = np.mean([actual_loss(act, q["K"], c, r) for q in Q])
                ca = 100.0 if act == "clarify" else 0.0
                print(f"  {name:<20}{loss:>10.3f}{ca:>11.0f}{ca:>11.0f}")
            for belief in ("oracle", "realistic"):
                acts = [(q, choose(belief, q, c, r)) for q in Q]
                loss = np.mean([actual_loss(a, q["K"], c, r) for q, a in acts])
                ca = 100 * np.mean([a == "clarify" for q, a in acts if q["is_amb"]])
                cc = 100 * np.mean([a == "clarify" for q, a in acts if not q["is_amb"]])
                print(f"  Bayes-{belief:<14}{loss:>10.3f}{ca:>11.0f}{cc:>11.0f}")
            print()
    print("Reading: at r=1.0, Bayes-oracle clarifies ambiguous / executes control and beats both")
    print("baselines -> the decision layer has real value IF discovery works. Bayes-realistic ~ always-")
    print("execute (collapsed posterior never flags ambiguity). At r=0.3, clarify is too expensive to")
    print("realize, so even the oracle stops clarifying -> realization gates the payoff. oracle@r=1 minus")
    print("realistic = value of interpretation discovery; r=1 vs r=0.3 = value of fixing realization.")


if __name__ == "__main__":
    main()
