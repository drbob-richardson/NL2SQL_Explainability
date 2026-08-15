"""GraphRAG downstream multi-hop QA -- does the retrieval win become an ANSWER win?

For chained (multi-hop) questions, at each judgment budget B, retrieve the top-k passages by each
method (passive verify-top-B / cosine-GP=BAGEL-lite / graph-GP=ours), feed ONLY those passages to
gpt-4o-mini, and score the answer (EM + token-F1) against gold. The money plot: answer-accuracy gain
(graph-GP - passive) vs budget -- the recall win from structure-as-covariance turned into an end-task win.

Safe by default: prints a dry-run cost estimate and exits unless --run; caps calls with --max-calls;
caches every answer (keyed by the passage SET, so identical retrievals across methods/budgets and the
shared B=0 baseline collapse to a single call).

  ./.venv/bin/python scripts/graphrag_downstream_qa.py --subset 150         # dry-run estimate
  ./.venv/bin/python scripts/graphrag_downstream_qa.py --subset 150 --run   # execute
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, string, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pyarrow.parquet as pq
from graphrag_active_scale import calib, kern_graph, kern_cos, post, title_graph, CHAINED

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = "gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.150, 0.600
CACHE = os.path.join(ROOT, "data", "graphrag_qa_answers.json")
BUDGETS = [0, 1, 2, 3]
METHODS = [("passive", None, False), ("cosine-GP", kern_cos, True), ("graph-GP", kern_graph, True)]
DATASETS = [("HotpotQA", "data/hotpot/dev_distractor.parquet", False, "data/hotpot_emb.json"),
            ("2WikiMultiHopQA", "data/twowiki/dev.parquet", True, "data/twowiki_emb.json")]
SYS = ("Answer the question using ONLY the passages provided. Reply with the short answer only "
       "(a name, entity, number, or yes/no), nothing else. If the passages seem insufficient, give "
       "your best guess from them.")


def build_qa(rows, cache, twowiki):
    def vec(s):
        v = np.array(cache[s]); return v / (np.linalg.norm(v) + 1e-9)
    P = []
    for r in rows:
        if twowiki:
            ctx = json.loads(r["context"]); titles = [c[0] for c in ctx]; sents = [c[1] for c in ctx]
            gold = set(sf[0] for sf in json.loads(r["supporting_facts"])) & set(titles)
        else:
            titles = list(r["context"]["title"]); sents = list(r["context"]["sentences"])
            gold = set(r["supporting_facts"]["title"]) & set(titles)
        texts = [t + ". " + " ".join(s) for t, s in zip(titles, sents)]
        if len(gold) < 2 or not (4 <= len(titles) <= 16):
            continue
        if r["question"] not in cache or any(tx not in cache for tx in texts):
            continue
        n = len(titles); qv = vec(r["question"]); V = np.array([vec(tx) for tx in texts])
        gi = np.array([1.0 if titles[i] in gold else 0.0 for i in range(n)])
        P.append(dict(q=r["question"], answer=str(r["answer"]), titles=titles, texts=texts,
                      cos=V @ qv, V=V, A=title_graph(titles, texts), gi=gi, n=n,
                      k=int(gi.sum()), type=r["type"]))
    return P


def retrieve_at_budget(p, prior, kernel, active, B, beta=0.7):
    """Top-k passage indices after B judgments -- mirrors graphrag_active_scale.run scoring."""
    m = prior(p["cos"]); y = p["gi"]; n = p["n"]; K = kernel(p) if kernel else None
    judged, prior_order = [], list(np.argsort(-m))
    for step in range(B + 1):
        if step == B:
            mean = post(m, K, judged, y)[0] if K is not None else m.copy()
            sc = mean.copy()
            for j in judged:
                sc[j] = 1e6 if y[j] > 0 else -1e6          # judged relevant -> top, non-rel -> sink
            return list(np.argsort(-sc)[:p["k"]])
        rem = [i for i in range(n) if i not in set(judged)]
        if active:
            mean, var = post(m, K, judged, y); acq = mean + beta * np.sqrt(var)
            judged.append(rem[int(np.argmax(acq[rem]))])
        else:
            judged.append(next(i for i in prior_order if i not in set(judged)))


PUNC = str.maketrans("", "", string.punctuation)
def norm(s):
    s = str(s).lower().translate(PUNC); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def em(pred, gold):
    return float(norm(pred) == norm(gold))
def f1(pred, gold):
    pt, gt = norm(pred).split(), norm(gold).split()
    if not pt or not gt: return float(pt == gt)
    ns = sum((Counter(pt) & Counter(gt)).values())
    if ns == 0: return 0.0
    prec, rec = ns / len(pt), ns / len(gt); return 2 * prec * rec / (prec + rec)


def context(p, idxs):
    return "\n\n".join(f"[{i+1}] {p['texts'][j]}" for i, j in enumerate(idxs))
def ans_key(p, idxs):
    t = "|".join(sorted(p["titles"][j] for j in idxs)); return hashlib.md5(f"{p['q']}||{t}".encode()).hexdigest()
def ntok(s):
    try:
        import tiktoken; return len(tiktoken.get_encoding("o200k_base").encode(s))
    except Exception:
        return len(s) // 4


def ci(a, b, nb=3000):
    rng = np.random.RandomState(0); a, b = np.asarray(a), np.asarray(b)
    d = [(a[s] - b[s]).mean() for s in (rng.randint(0, len(a), len(a)) for _ in range(nb))]
    return (a - b).mean(), np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="rows scanned per dataset")
    ap.add_argument("--subset", type=int, default=150, help="chained questions per dataset")
    ap.add_argument("--run", action="store_true"); ap.add_argument("--max-calls", type=int, default=6000)
    args = ap.parse_args()

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    cells = []   # (dataset, method, B, p, idxs, recall, key)
    need = {}    # key -> (context text, question) for uncached answers
    for ds, path, tw, emb in DATASETS:
        rows = pq.read_table(os.path.join(ROOT, path)).slice(0, args.n).to_pylist()
        embc = json.load(open(os.path.join(ROOT, emb)))
        P = build_qa(rows, embc, tw); del embc
        prior = calib(P)
        chained = [p for p in P if p["type"] in CHAINED][:args.subset]
        print(f"{ds}: {len(chained)} chained questions")
        for p in chained:
            for mname, kern, act in METHODS:
                for B in BUDGETS:
                    idxs = retrieve_at_budget(p, prior, kern, act, B)
                    k = ans_key(p, idxs)
                    cells.append((ds, mname, B, p, idxs, float(p["gi"][idxs].sum()) / p["k"], k))
                    if k not in cache:
                        need[k] = (context(p, idxs), p["q"])

    in_tok = sum(ntok(SYS) + ntok(f"Question: {q}\n\nPassages:\n{c}") + 12 for c, q in need.values())
    cost = in_tok / 1e6 * PRICE_IN + len(need) * 12 / 1e6 * PRICE_OUT
    print(f"\ncells: {len(cells)};  unique answers needed: {len(need)} uncached "
          f"(cache has {len(cache)});  ~{in_tok/1000:.0f}K in;  est cost ${cost:.4f}")

    if need and not args.run:
        print("[dry run] re-run with --run to generate answers."); return
    if len(need) > args.max_calls:
        print(f"REFUSING: {len(need)} > --max-calls {args.max_calls}"); sys.exit(1)
    if need:
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not set"); sys.exit(1)
        from openai import OpenAI
        client = OpenAI()
        for i, (k, (ctx, q)) in enumerate(need.items(), 1):
            resp = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=30,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": f"Question: {q}\n\nPassages:\n{ctx}"}])
            cache[k] = resp.choices[0].message.content.strip()
            if i % 100 == 0:
                json.dump(cache, open(CACHE, "w")); print(f"  {i}/{len(need)}")
        json.dump(cache, open(CACHE, "w"))

    # ---- score: EM / F1 / recall by (method, B); graph-GP - passive gain with paired bootstrap ----
    agg = {(m, B): {"em": [], "f1": [], "rec": []} for m, _, _ in METHODS for B in BUDGETS}
    for ds, mname, B, p, idxs, rec, k in cells:
        a = cache.get(k, "")
        agg[(mname, B)]["em"].append(em(a, p["answer"]))
        agg[(mname, B)]["f1"].append(f1(a, p["answer"]))
        agg[(mname, B)]["rec"].append(rec)
    print("\n=== DOWNSTREAM MULTI-HOP QA (chained, both datasets pooled) ===")
    print("  " + "method".ljust(11) + "".join(f"B={B}:EM/F1/rec".ljust(20) for B in BUDGETS))
    for mname, _, _ in METHODS:
        row = "  " + mname.ljust(11)
        for B in BUDGETS:
            d = agg[(mname, B)]
            row += f"{np.mean(d['em']):.3f}/{np.mean(d['f1']):.3f}/{np.mean(d['rec']):.3f}".ljust(20)
        print(row)
    print("\n  graph-GP - passive gain (paired bootstrap 95% CI):")
    for metric in ("em", "f1", "rec"):
        line = f"    {metric.upper():<4}"
        for B in BUDGETS:
            m, c = ci(agg[("graph-GP", B)][metric], agg[("passive", B)][metric])
            line += f"  B={B}: {m:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]"
        print(line)
    print("\n  => if the EM/F1 gain tracks the recall gain at low B, structure-as-covariance pays off end-to-end.")


if __name__ == "__main__":
    main()
