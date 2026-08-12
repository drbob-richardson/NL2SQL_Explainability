r"""Candidate-restricted approximate inference (JASA A&CS reframing, item #4).

Exact evaluation of the joint posterior enumerates all 2^|V| table subsets, which is only feasible for the
small schemas here (|V| <= 14). This script shows the model scales past that boundary WITHOUT losing fidelity:

  Candidate restriction. Rank tables by the cheap unary score a_t, keep a pool C of the top-K plus their
  foreign-key neighbours, and enumerate the joint posterior exactly over the induced subgraph on C
  (2^|C| configurations), forcing the tables outside C to be excluded.

Part A (this script): on BIRD and Spider, where full exact inference IS feasible, we validate that
candidate-restricted inference reproduces the full-exact marginals, recall, MAP subset, and predictive-set
coverage as the pool grows -- i.e. the approximation is faithful, at a fraction of the configurations.
Part B (scripts/s2_beaver.py): the same restriction runs on the BEAVER 'dw' warehouse (97 tables, 2^97
infeasible), where the top-15 pool recovers gold tables better than cosine and every structural baseline.

  ./.venv/bin/python scripts/bayes_subgraph_scale.py [bird|spider]
Writes data/scale_{DS}.json and prints a fidelity-vs-pool-size table.
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np
import torch

DS = sys.argv[1] if len(sys.argv) > 1 else "bird"
sys.argv = ["x", DS, "xfit"]
from bayes_subgraph_hbayes import build_features, db_tensors, NFEAT  # noqa: E402

torch.set_default_dtype(torch.float64)
softplus = torch.nn.functional.softplus
K_DRAWS = int(os.environ.get("HB_K", "200"))
ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    out, sch, fold = build_features()
    n = len(out)
    dbs = sorted(set(o["db"] for o in out)); dbi = {d: i for i, d in enumerate(dbs)}
    T = db_tensors(out)
    for o in out:
        o["phi_t"] = torch.tensor(o["phi"]); gm = 0
        for t in o["gold"]:
            gm |= (1 << o["tbls"].index(t))
        o["gidx"] = gm
    P_DIM = NFEAT + 4 + 2 * len(dbs)
    print(f"=== Candidate-restricted fidelity ({DS.upper()}, {n} queries, {len(dbs)} DBs, K={K_DRAWS}) ===")
    tabmax = max(len(o["tbls"]) for o in out)
    print(f"    table counts: max {tabmax}, mean {np.mean([len(o['tbls']) for o in out]):.1f}")

    def unpack(P):
        theta = P[:NFEAT]; alpha0, b0, lsa, lsb = P[NFEAT], P[NFEAT + 1], P[NFEAT + 2], P[NFEAT + 3]
        z = P[NFEAT + 4:].reshape(len(dbs), 2); sa, sb = torch.exp(lsa), torch.exp(lsb)
        return theta, alpha0, b0, sa, sb, z

    def neg_log_joint(P, idx):
        theta, alpha0, b0, sa, sb, z = unpack(P)
        alpha_d = alpha0 + sa * z[:, 0]; beta_d = softplus(b0 + sb * z[:, 1]); by = {}
        for i in idx:
            by.setdefault(out[i]["db"], []).append(i)
        ll = 0.0
        for db, idxs in by.items():
            t = T[db]; di = dbi[db]; phi = torch.stack([out[i]["phi_t"] for i in idxs]); a = phi @ theta
            score = alpha_d[di] * t["Nsel"][None, :] + a @ t["bits"].T + beta_d[di] * t["ec"][None, :]
            logZ = torch.logsumexp(score, dim=1); gidx = torch.tensor([out[i]["gidx"] for i in idxs])
            ll = ll + (score.gather(1, gidx[:, None]).squeeze(1) - logZ).sum()
        lp = -(theta ** 2).sum() / 8.0 - alpha0 ** 2 / 18.0 - b0 ** 2 / 18.0
        lp = lp - (sa ** 2) / 2.0 - (sb ** 2) / 2.0 + torch.log(sa) + torch.log(sb) - (z ** 2).sum() / 2.0
        return -(ll + lp)

    def fit_map(idx, steps=400):
        P = torch.zeros(P_DIM, requires_grad=True)
        with torch.no_grad():
            P[NFEAT + 2] = -0.5; P[NFEAT + 3] = -0.5
        opt = torch.optim.Adam([P], lr=0.05)
        for _ in range(steps):
            opt.zero_grad(); loss = neg_log_joint(P, idx); loss.backward(); opt.step()
        return P.detach()

    def laplace_L(P, idx):
        P = P.clone().requires_grad_(True)
        g = torch.autograd.grad(neg_log_joint(P, idx), P, create_graph=True)[0]
        H = torch.zeros(P_DIM, P_DIM)
        for i in range(P_DIM):
            H[i] = torch.autograd.grad(g[i], P, retain_graph=True)[0]
        H = 0.5 * (H + H.T) + 1e-4 * torch.eye(P_DIM); cov = torch.linalg.inv(H)
        return torch.linalg.cholesky((cov + cov.T) / 2 + 1e-8 * torch.eye(P_DIM))

    def enum_bits(m):
        masks = np.arange(1 << m)
        return ((masks[:, None] >> np.arange(m)) & 1).astype(np.float64)

    def full_predict(Ps, o, di):
        t = T[o["db"]]; pacc = torch.zeros(len(t["Nsel"]))
        for P in Ps:
            theta, alpha0, b0, sa, sb, z = unpack(P)
            alpha_d = alpha0 + sa * z[:, 0]; beta_d = softplus(b0 + sb * z[:, 1]); a = o["phi_t"] @ theta
            pacc = pacc + torch.softmax(alpha_d[di] * t["Nsel"] + t["bits"] @ a + beta_d[di] * t["ec"], dim=0)
        pavg = (pacc / len(Ps)).numpy(); marg = (pavg[:, None] * t["bits"].numpy()).sum(0)
        return pavg, marg

    def pool_select(o, amean, K):
        order = np.argsort(-amean); pool = set(order[:K].tolist())
        nb = set()
        for (i, j) in o["edges"]:
            if i in pool:
                nb.add(j)
            if j in pool:
                nb.add(i)
        pool |= nb
        return sorted(pool)

    def pool_predict(Ps, o, di, pool):
        m = len(pool); bits = enum_bits(m); Nsel = bits.sum(1); pidx = {p: j for j, p in enumerate(pool)}
        ec = np.zeros(1 << m)
        for (i, j) in o["edges"]:
            if i in pidx and j in pidx:
                ec += bits[:, pidx[i]] * bits[:, pidx[j]]
        bt = torch.tensor(bits); Nt = torch.tensor(Nsel); et = torch.tensor(ec); phi_pool = o["phi_t"][pool]
        pacc = torch.zeros(1 << m)
        for P in Ps:
            theta, alpha0, b0, sa, sb, z = unpack(P)
            alpha_d = alpha0 + sa * z[:, 0]; beta_d = softplus(b0 + sb * z[:, 1]); a = phi_pool @ theta
            pacc = pacc + torch.softmax(alpha_d[di] * Nt + bt @ a + beta_d[di] * et, dim=0)
        pavg = (pacc / len(Ps)).numpy(); marg_pool = (pavg[:, None] * bits).sum(0)
        return pavg, marg_pool, bits

    def recall_at(marg_full_space, gold_idx, gsz, amean):
        # rank by marginal, break ties / rank the zeros by unary score
        key = marg_full_space + 1e-6 * (amean - amean.min()) / (np.ptp(amean) + 1e-9)
        top = set(np.argsort(-key)[:gsz].tolist())
        return len(top & gold_idx) / len(gold_idx)

    Ks = [4, 6, 8, 10]
    gen = torch.Generator().manual_seed(0)
    agg = {K: dict(dm=[], rr=[], rf=[], mapok=[], ginp=[], psz=[], cfg=[], cov=[], covf=[]) for K in Ks}
    for te in (0, 1):
        tr = [i for i in range(n) if fold[i] != te]; teq = [i for i in range(n) if fold[i] == te]
        Pmap = fit_map(tr); L = laplace_L(Pmap, tr)
        Ps = [Pmap + L @ torch.randn(P_DIM, generator=gen) for _ in range(K_DRAWS)]
        amean_all = {}
        for qi in teq:
            o = out[qi]
            amean_all[qi] = torch.stack([o["phi_t"] @ unpack(P)[0] for P in Ps]).mean(0).detach().numpy()
        for qi in teq:
            o = out[qi]; di = dbi[o["db"]]; nn = len(o["tbls"]); gidx = o["gidx"]
            gset = {i for i in range(nn) if (gidx >> i) & 1}; gsz = len(gset)
            pavg_f, m_f = full_predict(Ps, o, di)
            # full-exact reference metrics
            order_f = np.argsort(-pavg_f); cf = np.cumsum(pavg_f[order_f]); kf = int(np.searchsorted(cf, 0.90) + 1)
            cov_f = 1.0 if gidx in order_f[:kf] else 0.0
            rf = recall_at(m_f, gset, gsz, amean_all[qi])
            for K in Ks:
                if K >= nn:
                    # pool would be the whole schema -> identical to exact; record as exact
                    agg[K]["dm"].append(0.0); agg[K]["rr"].append(rf); agg[K]["rf"].append(rf)
                    agg[K]["mapok"].append(1.0); agg[K]["ginp"].append(1.0)
                    agg[K]["psz"].append(nn); agg[K]["cfg"].append(1 << nn); agg[K]["cov"].append(cov_f)
                    agg[K]["covf"].append(cov_f); continue
                pool = pool_select(o, amean_all[qi], K)
                pavg_p, marg_p, bits = pool_predict(Ps, o, di, pool)
                m_rest = np.zeros(nn)
                for j, p in enumerate(pool):
                    m_rest[p] = marg_p[j]
                pool_set = set(pool)
                dm = np.mean(np.abs(m_rest[list(pool_set)] - m_f[list(pool_set)]))
                rr = recall_at(m_rest, gset, gsz, amean_all[qi])
                # restricted MAP subset (map pool bitmask to full-space table set)
                best = int(np.argmax(pavg_p)); mapset = {pool[j] for j in range(len(pool)) if (best >> j) & 1}
                map_full = {i for i in range(nn) if (int(np.argmax(pavg_f)) >> i) & 1}
                # restricted 90% predictive set coverage (gold must be in pool AND in the HPD set)
                gold_in_pool = gset <= pool_set
                gold_local = 0
                if gold_in_pool:
                    for j, p in enumerate(pool):
                        if p in gset:
                            gold_local |= (1 << j)
                    order_p = np.argsort(-pavg_p); cp = np.cumsum(pavg_p[order_p]); kp = int(np.searchsorted(cp, 0.90) + 1)
                    cov_p = 1.0 if gold_local in order_p[:kp] else 0.0
                else:
                    cov_p = 0.0
                agg[K]["dm"].append(dm); agg[K]["rr"].append(rr); agg[K]["rf"].append(rf)
                agg[K]["mapok"].append(1.0 if mapset == map_full else 0.0)
                agg[K]["ginp"].append(1.0 if gold_in_pool else 0.0)
                agg[K]["psz"].append(len(pool)); agg[K]["cfg"].append(1 << len(pool))
                agg[K]["cov"].append(cov_p); agg[K]["covf"].append(cov_f)

    print(f"\n  pool = top-K unary + FK neighbours; full-exact recall = {np.mean(agg[Ks[0]]['rf']):.3f}, "
          f"coverage = {np.mean(agg[Ks[0]]['covf']):.3f}")
    print(f"  {'K':>3} {'pool_sz':>8} {'#configs':>9} {'gold_in_pool':>13} {'marg_L1':>9} "
          f"{'recall':>7} {'MAP_agree':>10} {'set_cov':>8}")
    res = {}
    for K in Ks:
        a = agg[K]
        row = dict(pool_sz=float(np.mean(a["psz"])), configs=float(np.mean(a["cfg"])),
                   gold_in_pool=float(np.mean(a["ginp"])), marg_L1=float(np.mean(a["dm"])),
                   recall=float(np.mean(a["rr"])), recall_full=float(np.mean(a["rf"])),
                   map_agree=float(np.mean(a["mapok"])), set_cov=float(np.mean(a["cov"])),
                   set_cov_full=float(np.mean(a["covf"])))
        res[str(K)] = row
        print(f"  {K:>3} {row['pool_sz']:>8.1f} {row['configs']:>9.0f} {row['gold_in_pool']:>13.3f} "
              f"{row['marg_L1']:>9.4f} {row['recall']:>7.3f} {row['map_agree']:>10.3f} {row['set_cov']:>8.3f}")
    print(f"  full 2^|V| mean configs = {np.mean([1 << len(o['tbls']) for o in out]):.0f}")

    js = dict(ds=DS, n=n, K_draws=K_DRAWS, max_tables=int(tabmax), full=res)
    path = os.path.join(ROOT, "data", f"scale_{DS}.json")
    json.dump(js, open(path, "w"), indent=1); print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
