r"""Decision analysis on the joint subgraph posterior (JASA A&CS reframing, item #1).

The object of inference is a calibrated joint distribution p(x | D) over table subsets (the length-2^n
`pavg` from the hierarchical-Bayes predictive). This script turns that posterior into a *decision*: which
schema set S to pass to the LLM, under asymmetric costs.

Realistic loss (NON-separable, so the joint matters): a query's SQL fails if ANY required table is dropped,
while each extra table costs context.
    L(S, R) = c_miss * 1{ R not subset of S } + c_include * |S \ R|.
Under the posterior over the required set R ~ p(. | D),
    E[L(S)] = c_miss * (1 - P(R subset of S)) + c_include * sum_{t in S} (1 - m_t),
where P(R subset of S) = sum_{c subset of S} pavg[c]  (a JOINT quantity; marginals cannot produce it).

We compare, out of sample under the 2-fold cross-fit:
  * joint Bayes action   S*(rho) = argmin_S E[L(S)],  rho = c_miss / c_include   (uses the joint)
  * marginal threshold   S_tau = { t : m_t >= tau }                              (uses only marginals)
  * MAP subset           argmax_c pavg[c]
  * top-k marginal       the round(E[|S|]) tables of highest m_t
  * containment set      S_eta = smallest S with P(R subset of S) >= eta         (uses the joint)
Reported per rule: miss rate (fraction with R not subset of S), mean extra tables |S\R|, mean |S|.
The headline is the (miss rate, extra tables) frontier: the joint rule should dominate marginal thresholding.

  ./.venv/bin/python scripts/bayes_subgraph_decision.py [bird|spider]
Writes data/decision_{DS}.json and prints a summary.
"""
from __future__ import annotations
import sys, os, json
import numpy as np
import torch

DS = sys.argv[1] if len(sys.argv) > 1 else "bird"
sys.argv = ["x", DS, "xfit"]
from bayes_subgraph_hbayes import build_features, db_tensors, NFEAT  # noqa: E402

torch.set_default_dtype(torch.float64)
softplus = torch.nn.functional.softplus
K = int(os.environ.get("HB_K", "200"))       # Laplace draws in the predictive
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
    print(f"=== Decision analysis ({DS.upper()}, {n} queries, {len(dbs)} DBs, K={K}) ===")

    def unpack(P):
        theta = P[:NFEAT]; alpha0, b0, lsa, lsb = P[NFEAT], P[NFEAT + 1], P[NFEAT + 2], P[NFEAT + 3]
        z = P[NFEAT + 4:].reshape(len(dbs), 2); sa, sb = torch.exp(lsa), torch.exp(lsb)
        return theta, alpha0, b0, sa, sb, z

    def neg_log_joint(P, idx):
        theta, alpha0, b0, sa, sb, z = unpack(P)
        alpha_d = alpha0 + sa * z[:, 0]; beta_d = softplus(b0 + sb * z[:, 1])
        by = {}
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

    def predict_pavg(P_samples, qi, beta_zero=False):
        o = out[qi]; t = T[o["db"]]; di = dbi[o["db"]]; pacc = torch.zeros(len(t["Nsel"]))
        for P in P_samples:
            theta, alpha0, b0, sa, sb, z = unpack(P)
            alpha_d = alpha0 + sa * z[:, 0]; beta_d = softplus(b0 + sb * z[:, 1])
            a = o["phi_t"] @ theta; bd = torch.zeros(()) if beta_zero else beta_d[di]
            score = alpha_d[di] * t["Nsel"] + t["bits"] @ a + bd * t["ec"]
            pacc = pacc + torch.softmax(score, dim=0)
        pavg = (pacc / len(P_samples)).numpy()
        marg = (pavg[:, None] * t["bits"].numpy()).sum(0)
        return pavg, marg

    # ---- cross-fit: predict every query out of sample ----
    gen = torch.Generator().manual_seed(0)
    Q = []  # per-query decision inputs
    for te in (0, 1):
        tr = [i for i in range(n) if fold[i] != te]; teq = [i for i in range(n) if fold[i] == te]
        Pmap = fit_map(tr); L = laplace_L(Pmap, tr)
        Ps = [Pmap + L @ torch.randn(P_DIM, generator=gen) for _ in range(K)]
        for qi in teq:
            o = out[qi]; nn = len(o["tbls"])
            pavg, marg = predict_pavg(Ps, qi, beta_zero=False)      # coupled, calibrated
            pavg0, marg0 = predict_pavg(Ps, qi, beta_zero=True)     # independent beta=0 (jointly overconfident)
            unary = torch.stack([o["phi_t"] @ unpack(P)[0] for P in Ps]).mean(0).detach().numpy()
            Q.append(dict(nn=nn, gidx=o["gidx"], pavg=pavg, m=marg, pavg0=pavg0, m0=marg0,
                          unary=unary, gsize=len(o["gold"]), db=o["db"]))
    print(f"predicted {len(Q)} held-out queries")

    # ---- helpers on a single query ----
    def subset_sums(p, nn):
        """f[S] = sum_{c subset of S} p[c]  (zeta transform over nn bits)."""
        f = p.astype(np.float64).copy(); idx = np.arange(1 << nn)
        for i in range(nn):
            b = 1 << i; mask = (idx & b) > 0
            f[mask] += f[idx[mask] ^ b]
        return f

    def bits_matrix(nn):
        return ((np.arange(1 << nn)[:, None] >> np.arange(nn)) & 1).astype(np.float64)

    def eval_S(Smask, gidx):
        miss = 1.0 if (gidx & ~Smask) != 0 else 0.0          # some required table dropped
        extra = bin(Smask & ~gidx).count("1")                # |S \ R|
        return miss, extra, bin(Smask).count("1")

    # precompute per query: f = P(R subset of S), extra_cost = E|S\R|, size
    for q in Q:
        nn = q["nn"]; B = bits_matrix(nn)
        q["f"] = subset_sums(q["pavg"], nn)                  # P(R subset S) under coupled posterior
        q["f0"] = subset_sums(q["pavg0"], nn)               # ... under independent beta=0 model
        q["extra_cost"] = B @ (1.0 - q["m"])                 # E[|S\R|] for every S
        q["size"] = B.sum(1)
        q["B"] = B

    def aggregate(masks):
        mi = ex = sz = 0.0
        for q, Sm in zip(Q, masks):
            m_, e_, s_ = eval_S(int(Sm), q["gidx"]); mi += m_; ex += e_; sz += s_
        k = len(Q); return mi / k, ex / k, sz / k

    # ---- rules ----
    def joint_bayes(rho):
        masks = []
        for q in Q:
            EL = rho * (1.0 - q["f"]) + q["extra_cost"]      # E[L]/c_include
            masks.append(int(np.argmin(EL)))
        return masks

    def marginal_threshold(tau):
        masks = []
        for q in Q:
            sel = np.where(q["m"] >= tau)[0]
            Sm = 0
            for t in sel:
                Sm |= (1 << int(t))
            if Sm == 0:
                Sm = 1 << int(np.argmax(q["m"]))
            masks.append(Sm)
        return masks

    def map_rule():
        return [int(np.argmax(q["pavg"])) for q in Q]

    def topk_rule():
        masks = []
        for q in Q:
            k = max(1, int(round(q["m"].sum())))
            top = np.argsort(-q["m"])[:k]; Sm = 0
            for t in top:
                Sm |= (1 << int(t))
            masks.append(Sm)
        return masks

    def containment(eta, key="f"):
        masks = []
        for q in Q:
            feas = np.where(q[key] >= eta)[0]
            if len(feas) == 0:
                masks.append(int(np.argmax(q[key]))); continue
            # smallest cardinality feasible S, break ties by fewest expected extra tables
            sizes = q["size"][feas]
            best = feas[np.lexsort((q["extra_cost"][feas], sizes))[0]]
            masks.append(int(best))
        return masks

    # ---- independence surrogate: f_indep[S] = P_indep(R subset of S) = prod_{t not in S}(1 - m_t) ----
    # Same asymmetric-loss decision framework as the joint; only the correlation structure differs.
    for q in Q:
        mc = np.clip(q["m"], 1e-6, 1 - 1e-6); logc = np.log(1 - mc)
        q["f_indep"] = np.exp(logc.sum() - q["B"] @ logc)

    def bayes_action(rho, key):
        return [int(np.argmin(rho * (1.0 - q[key]) + q["extra_cost"])) for q in Q]

    rhos = np.round(np.exp(np.linspace(np.log(1.0), np.log(80.0), 24)), 3)
    joint_fr = [dict(rho=float(r), **dict(zip(("miss", "extra", "size"), aggregate(bayes_action(r, "f"))))) for r in rhos]
    indep_fr = [dict(rho=float(r), **dict(zip(("miss", "extra", "size"), aggregate(bayes_action(r, "f_indep"))))) for r in rhos]
    taus = np.round(np.linspace(0.02, 0.9, 30), 3)
    marg_fr = [dict(tau=float(t), **dict(zip(("miss", "extra", "size"), aggregate(marginal_threshold(t))))) for t in taus]
    map_m = dict(zip(("miss", "extra", "size"), aggregate(map_rule())))
    topk_m = dict(zip(("miss", "extra", "size"), aggregate(topk_rule())))
    cont = {}; cont_b0 = {}
    for eta in (0.80, 0.90, 0.95):
        mi, ex, sz = aggregate(containment(eta, "f")); cont[f"{eta:.2f}"] = dict(cover=1.0 - mi, extra=ex, size=sz)
        mi2, ex2, sz2 = aggregate(containment(eta, "f0")); cont_b0[f"{eta:.2f}"] = dict(cover=1.0 - mi2, extra=ex2, size=sz2)

    loss_cmp = []
    for j, ii in zip(joint_fr, indep_fr):
        lj = j["rho"] * j["miss"] + j["extra"]; li = ii["rho"] * ii["miss"] + ii["extra"]
        loss_cmp.append(dict(rho=j["rho"], loss_joint=lj, loss_indep=li, gain=(li - lj) / max(li, 1e-9)))
    wins = sum(1 for d in loss_cmp if d["loss_joint"] <= d["loss_indep"] + 1e-9)

    def min_extra_at(fr, target):
        ok = [d for d in fr if (1.0 - d["miss"]) >= target]; return min(ok, key=lambda d: d["extra"]) if ok else None
    matched = {}
    for tg in (0.85, 0.90, 0.95):
        matched[f"{tg:.2f}"] = dict(joint=min_extra_at(joint_fr, tg), indep=min_extra_at(indep_fr, tg),
                                    marg=min_extra_at(marg_fr, tg))

    print(f"\n  Bayes action, same loss: TRUE JOINT vs INDEPENDENCE surrogate (isolates the joint's value)")
    print(f"  joint realized loss <= independence at {wins}/{len(loss_cmp)} cost ratios")
    print(f"  {'rho':>6} {'loss_joint':>11} {'loss_indep':>11} {'reduction':>10} {'miss_j':>8} {'miss_i':>8}")
    for k, (j, ii, d) in enumerate(zip(joint_fr, indep_fr, loss_cmp)):
        if k in (0, 6, 12, 18, 23):
            print(f"  {j['rho']:>6.1f} {d['loss_joint']:>11.3f} {d['loss_indep']:>11.3f} {d['gain']*100:>9.1f}% {j['miss']:>8.3f} {ii['miss']:>8.3f}")
    print(f"\n  min extra tables at matched realized retain-all coverage (lower is better):")
    print(f"  {'target':>7} {'joint':>7} {'indep':>7} {'marg':>7}")
    for tg, d in matched.items():
        g = lambda x: f"{x['extra']:.2f}" if x else "  -"
        print(f"  {tg:>7} {g(d['joint']):>7} {g(d['indep']):>7} {g(d['marg']):>7}")
    print(f"\n  containment sets S_eta fed by CALIBRATED coupled posterior vs INDEPENDENT beta=0 (target=eta)")
    print(f"  {'eta':>5} {'coupled_cov':>12} {'coupled_sz':>11} {'indep0_cov':>11} {'indep0_sz':>10}")
    for e in cont:
        j = cont[e]; b = cont_b0[e]
        print(f"  {e:>5} {j['cover']:>12.3f} {j['size']:>11.2f} {b['cover']:>11.3f} {b['size']:>10.2f}")
    print(f"  MAP subset: retain-all {1 - map_m['miss']:.3f}, extra {map_m['extra']:.2f}, size {map_m['size']:.2f}")
    print(f"  top-k marg: retain-all {1 - topk_m['miss']:.3f}, extra {topk_m['extra']:.2f}, size {topk_m['size']:.2f}")

    # ---- budget-matched comparison: mean context size to reach a target retain-all ----
    maxn = max(q["nn"] for q in Q)

    def topk_frontier(scorekey):
        pts = []
        for k in range(1, maxn + 1):
            masks = []
            for q in Q:
                top = np.argsort(-q[scorekey])[:k]; Sm = 0
                for t in top:
                    Sm |= (1 << int(t))
                masks.append(Sm)
            mi, ex, sz = aggregate(masks); pts.append((1.0 - mi, sz))
        return pts

    def cont_frontier(key):
        pts = []
        for eta in np.linspace(0.50, 0.999, 40):
            mi, ex, sz = aggregate(containment(eta, key)); pts.append((1.0 - mi, sz))
        return pts

    def size_to_reach(pts, target):
        ok = [s for (r, s) in pts if r >= target - 1e-9]
        return float(min(ok)) if ok else None

    fr = {"containment S_eta (joint)": cont_frontier("f"),
          "containment (indep beta=0)": cont_frontier("f0"),
          "top-k (coupled marginal)": topk_frontier("m"),
          "top-k (unary score)": topk_frontier("unary")}
    budget = {}
    print(f"\n  budget-matched: mean context size (tables passed) to reach a target retain-all rate")
    print(f"  {'method':<30}{'@0.90':>7}{'@0.95':>8}{'@0.975':>8}")
    for name, pts in fr.items():
        row = {f"{t}": size_to_reach(pts, t) for t in (0.90, 0.95, 0.975)}
        budget[name] = row
        g = lambda x: f"{x:.2f}" if x is not None else "  -"
        print(f"  {name:<30}{g(row['0.9']):>7}{g(row['0.95']):>8}{g(row['0.975']):>8}")

    # ---- split-conformal containment: distribution-free retain-all guarantee (does not assume calibration) ----
    # Nested family: greedy-pi chain S_0 subset S_1 subset ...; S(eta) = first prefix with pi >= eta.
    # Score s_i = pi at which the gold set first enters the chain. Calibrate eta_hat on held-out queries.
    for q in Q:
        nn = q["nn"]; f = q["f"]; gidx = q["gidx"]; S = 0; chain = []; rem = set(range(nn)); score = None
        while rem:
            bt = max(rem, key=lambda t: f[S | (1 << t)]); S |= (1 << bt); rem.discard(bt)
            chain.append((float(f[S]), int(S), bin(S).count("1")))
            if score is None and (gidx & ~S) == 0:
                score = float(f[S])
        q["conf_score"] = score if score is not None else float(f[(1 << nn) - 1]); q["chain"] = chain

    def S_at(q, eta):
        for pi, mask, sz in q["chain"]:
            if pi >= eta:
                return pi, mask, sz
        return q["chain"][-1]

    def conformal(target, reps=200):
        nq = len(Q); scores = np.array([q["conf_score"] for q in Q]); rng = np.random.RandomState(0)
        covs, szs = [], []
        for _ in range(reps):
            perm = rng.permutation(nq); cal, test = perm[:nq // 2], perm[nq // 2:]
            ncal = len(cal); kk = min(ncal, int(np.ceil((ncal + 1) * target)))
            eta_hat = np.sort(scores[cal])[kk - 1]; cov = sz = 0.0
            for qi in test:
                _, mask, s = S_at(Q[qi], eta_hat)
                cov += 1.0 if (Q[qi]["gidx"] & ~mask) == 0 else 0.0; sz += s
            covs.append(cov / len(test)); szs.append(sz / len(test))
        return float(np.mean(covs)), float(np.std(covs)), float(np.mean(szs))

    conf = {}
    print(f"\n  split-conformal containment (distribution-free; random 50/50 cal/test, 200 reps):")
    print(f"  {'target':>7}{'realized cov':>14}{'(sd)':>8}{'mean size':>11}")
    for target in (0.90, 0.95):
        c, sd, z = conformal(target); conf[f"{target}"] = dict(cov=c, sd=sd, size=z)
        print(f"  {target:>7}{c:>14.3f}{sd:>8.3f}{z:>11.2f}")

    # ---- retention calibration: is the reported pi(S) calibrated vs the empirical event 1{R subset S}? ----
    # This is exactly the condition Proposition 4 assumes (reliability of the retention probability).
    rel_pi, rel_ret = [], []
    for q in Q:
        for eta in np.linspace(0.5, 0.99, 20):
            pi, mask, sz = S_at(q, eta)
            rel_pi.append(pi); rel_ret.append(1.0 if (q["gidx"] & ~mask) == 0 else 0.0)
    rel_pi = np.array(rel_pi); rel_ret = np.array(rel_ret)
    edges = np.linspace(float(rel_pi.min()), 1.0, 9); relbins = []; ece = 0.0
    for b in range(len(edges) - 1):
        hi = edges[b + 1] + (1e-9 if b == len(edges) - 2 else 0.0)
        m = (rel_pi >= edges[b]) & (rel_pi < hi)
        if m.sum() > 0:
            mp, mr, nb = float(rel_pi[m].mean()), float(rel_ret[m].mean()), int(m.sum())
            relbins.append(dict(stated=mp, empirical=mr, n=nb)); ece += nb * abs(mp - mr)
    ret_ece = float(ece / len(rel_pi))
    print(f"\n  retention calibration of pi(S_eta) vs empirical retain-all (Prop 4 assumption): ECE = {ret_ece:.3f}")
    print(f"  {'stated':>8}{'empirical':>11}{'n':>7}")
    for bd in relbins:
        print(f"  {bd['stated']:>8.3f}{bd['empirical']:>11.3f}{bd['n']:>7}")

    js = dict(ds=DS, n=len(Q), K=K, budget=budget, conformal=conf,
              retention_ece=ret_ece, retention_bins=relbins, frontiers={k: v for k, v in fr.items()},
              joint_frontier=joint_fr, indep_frontier=indep_fr, marg_frontier=marg_fr,
              loss_cmp=loss_cmp, joint_wins=wins, n_rho=len(loss_cmp), map=map_m, topk=topk_m,
              containment=cont, containment_b0=cont_b0, matched=matched)
    path = os.path.join(ROOT, "data", f"decision_{DS}.json")
    json.dump(js, open(path, "w"), indent=1); print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
