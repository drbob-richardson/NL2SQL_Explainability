"""UQ-coverage experiment (make-or-break for Option C).

Question: does a PYP-reserve / elicited-interpretation-count score control executed-error (here:
executing on an AMBIGUOUS question) at a target risk with HIGHER coverage than the divergence and
confidence baselines that already failed (divergence AUROC 0.475, probe 4b)?

Scores per question (oriented so higher => more likely ambiguous => prefer clarify):
  - divergence   : # distinct result-set clusters among the 8 samples (baseline; known to fail)
  - uncertainty  : 1 - modal sample cluster weight (confidence baseline)
  - K_elicited   : # distinct interpretations from interpretation-first elicitation (the candidate)
  - pyp_reserve  : (theta + d*K_elicited)/(theta + m_elicited), canon-level (d=0.49, theta=30)
                   [monotone in K at these params -> same ranking as K_elicited; PYP adds calibration]

Target: is_ambiguous. Population: 300 ambiguous (cached elicitation) + 300 controls (elicited here).
Reports AUROC per score + LTT-style risk-coverage (execute low-score first; risk = frac ambiguous
among executed). Safe-by-default: dry-run unless --run (only the control elicitation costs API).

  ./.venv/bin/python scripts/ambrosia_uq_coverage.py            # dry-run / analyze
  ./.venv/bin/python scripts/ambrosia_uq_coverage.py --run
"""
from __future__ import annotations
import argparse, csv, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from collections import Counter
import numpy as np
csv.field_size_limit(10**7)
from bnp_nl2sql.execeval import open_db, run_sql

ROOT = os.path.join(os.path.dirname(__file__), "..")
ADIR = os.path.join(ROOT, "data", "ambrosia")
SAMPLES = os.path.join(ROOT, "data", "ambrosia_samples.json")
AMB_ELICIT = os.path.join(ROOT, "data", "ambrosia_interp_gen_gpt_4o.json")
CTL_ELICIT = os.path.join(ROOT, "data", "ambrosia_interp_gen_ctl_gpt_4o.json")
D, THETA = 0.49, 30.3   # canon-level PYP (findings §11)

GEN_SYS = ("You are a careful data analyst. A user's question about a database may be AMBIGUOUS: it "
           "can have more than one valid interpretation given the schema. List EVERY distinct valid "
           "interpretation in plain English -- do NOT write SQL. If the question is unambiguous, give "
           'one. Respond as JSON: {"interpretations": ["...", "..."]}.')


def db_path(db_file):
    rel = db_file[5:] if db_file.startswith("data/") else db_file
    return os.path.join(ADIR, rel)


def schema_str(conn):
    out = []
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        cols = conn.execute(f"PRAGMA table_info(`{t}`)").fetchall()
        out.append(f"{t}(" + ", ".join(c[1] for c in cols) + ")")
    return "\n".join(out)


def result_sig(conn, sql, timeout=4.0):
    t = threading.Timer(timeout, conn.interrupt); t.start()
    try:
        r = run_sql(conn, sql); return tuple(sorted(repr(x) for x in r)) if r else ("e",)
    except Exception:
        return None
    finally:
        t.cancel()


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(c)
    r = ((cs - c + cs + 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def coverage_at_risk(score, y, targets=(0.1, 0.2, 0.3)):
    """Execute (answer) the lowest-score questions first; risk = frac ambiguous among executed.
    Return max coverage with risk <= target."""
    order = np.argsort(score)  # low score first = execute
    y = np.asarray(y)[order]
    n = len(y); out = {}
    cum_amb = np.cumsum(y); cov = np.arange(1, n + 1)
    risk = cum_amb / cov
    for a in targets:
        ok = np.where(risk <= a)[0]
        out[a] = (ok.max() + 1) / n if len(ok) else 0.0
    return out


def elicit(rows, schemas, cache, cache_p, run, workers=12):
    todo = [r for r in rows if r["question"] not in cache]
    if not run or not todo:
        return cache, len(todo)
    from openai import OpenAI
    client = OpenAI()
    def do(r):
        usr = f"Schema:\n{schemas[db_path(r['db_file'])]}\n\nQuestion: {r['question']}"
        for a in range(5):
            try:
                resp = client.chat.completions.create(model="gpt-4o", temperature=0, max_tokens=400,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": GEN_SYS}, {"role": "user", "content": usr}])
                xs = json.loads(resp.choices[0].message.content).get("interpretations", [])
                return r["question"], [x for x in xs if isinstance(x, str)]
            except Exception:
                if a == 4: return r["question"], []
                time.sleep(min(2 ** a, 15))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(do, r) for r in todo]):
            k, v = f.result(); cache[k] = v
    json.dump(cache, open(cache_p, "w"))
    return cache, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-control", type=int, default=300)
    args = ap.parse_args()

    samples = json.load(open(SAMPLES))
    rows = list(csv.DictReader(open(os.path.join(ADIR, "ambrosia.csv"))))
    meta = {f"{r['db_file']}||{r['question']}": r for r in rows}
    amb_elicit = json.load(open(AMB_ELICIT))
    ctl_elicit = json.load(open(CTL_ELICIT)) if os.path.exists(CTL_ELICIT) else {}

    # ambiguous questions that have cached interpretation-first elicitation
    amb_rows = [meta[k] for k, e in samples.items() if e["is_ambiguous"] and meta.get(k) and meta[k]["question"] in amb_elicit]
    ctl_rows = [meta[k] for k, e in samples.items() if meta.get(k) and not e["is_ambiguous"]][:args.n_control]

    conns, schemas = {}, {}
    for r in ctl_rows:
        p = db_path(r["db_file"])
        if p not in conns:
            conns[p] = open_db(p); schemas[p] = schema_str(conns[p])
    ctl_elicit, todo = elicit(ctl_rows, schemas, ctl_elicit, CTL_ELICIT, args.run)
    print(f"control elicitation: {len(ctl_rows)} controls; to generate {todo}  (est ${todo*0.0011:.2f})")
    if todo and not args.run:
        print("  DRY RUN -- pass --run to elicit controls."); return

    # assemble population with all scores
    samp_by_q = {meta[k]["question"]: e for k, e in samples.items() if meta.get(k)}
    pop = []
    for r, is_amb, elic in ([(r, 1, amb_elicit) for r in amb_rows] + [(r, 0, ctl_elicit) for r in ctl_rows]):
        q = r["question"]
        if q not in elic or q not in samp_by_q:
            continue
        e = samp_by_q[q]
        p = db_path(r["db_file"])
        if p not in conns:
            conns[p] = open_db(p)
        sigs = [s for s in (result_sig(conns[p], s) for s in e["samples"]) if s is not None]
        if not sigs:
            continue
        cnt = Counter(sigs)
        divergence = len(cnt)
        uncertainty = 1.0 - cnt.most_common(1)[0][1] / len(sigs)
        K_el = len(set(elic[q])); m_el = max(1, len(elic[q]))
        reserve = (THETA + D * K_el) / (THETA + m_el)
        pop.append((is_amb, dict(divergence=divergence, uncertainty=uncertainty,
                                 K_elicited=K_el, pyp_reserve=reserve)))
    y = [a for a, _ in pop]
    print(f"\npopulation: {len(pop)} questions ({sum(y)} ambiguous, {len(y)-sum(y)} control)\n")
    print(f"  {'score':<14}{'AUROC(amb)':>11}{'cov@risk.1':>12}{'cov@risk.2':>12}{'cov@risk.3':>12}")
    for s in ("divergence", "uncertainty", "K_elicited", "pyp_reserve"):
        vals = [d[s] for _, d in pop]
        a = auroc(vals, y)
        cov = coverage_at_risk(np.array(vals), y)
        print(f"  {s:<14}{a:>11.3f}{cov[0.1]:>12.0%}{cov[0.2]:>12.0%}{cov[0.3]:>12.0%}")
    print("\nReading: AUROC(amb) = ranking ambiguous above control. cov@risk = fraction of questions we")
    print("can safely EXECUTE (lowest-score first) while keeping the ambiguous-among-executed rate <=")
    print("risk. If K_elicited/pyp_reserve >> divergence (0.475) and >> uncertainty, the ask-then-count")
    print("detector + PYP/LTT calibration is the real signal; if not, Option C collapses to Option B.")


if __name__ == "__main__":
    main()
