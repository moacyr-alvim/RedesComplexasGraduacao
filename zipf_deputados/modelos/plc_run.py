"""Uso: python plc_run.py votos_por_candidato.csv saida_prefixo n_boot_scan n_boot_fixo"""
import sys, json, time, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import optimize, stats
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CSV, OUT = sys.argv[1], sys.argv[2]
NB_SCAN, NB_FIX = int(sys.argv[3]), int(sys.argv[4])
d = pd.read_csv(CSV); fed = d[d.DS_CARGO == "Deputado Federal"]
SETS = [("Brasil", fed), ("SP", fed[fed.SG_UF == "SP"]), ("BA", fed[fed.SG_UF == "BA"])]

def boot_rep(args):
    x, xmin, a, th, seed, mode = args
    rng = np.random.default_rng(seed)
    n = len(x); tail = x >= xmin; nt = tail.sum(); body = x[~tail]
    m = rng.binomial(n, nt/n)
    xs = np.concatenate([plc.sample_tail(m, a, th, xmin, rng), rng.choice(body, n-m) if len(body) else np.array([])])
    if mode == "fix":
        a2, th2, _ = plc.fit(xs, xmin); return plc.ks(xs, xmin, a2, th2)
    return plc.scan(xs, plc.candidates(xs))[3]

def pointwise(x, xmin):
    x = x[x >= xmin]; n = len(x); out = {}
    a, th, ll = plc.fit(x, xmin)
    lx, g = plc._grid(a, th, xmin); lC = np.log(plc.integrate.simpson(g, x=lx))
    out["cutoff"] = -a*np.log(x) - th*x - lC
    ap = 1 + n/np.log(x/xmin).sum(); out["powerlaw"] = np.log((ap-1)/xmin) - ap*np.log(x/xmin)
    lam = 1/np.mean(x-xmin); out["exponential"] = np.log(lam) - lam*(x-xmin)
    L = np.log(x); zmin = lambda m, s: (np.log(xmin)-m)/s
    def lnl(p):
        m, s = p[0], np.exp(p[1])
        return -(stats.norm.logpdf((L-m)/s) - L - np.log(s) - stats.norm.logsf(zmin(m, s))).sum()
    r = optimize.minimize(lnl, [L.mean(), np.log(L.std())], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
    m, s = r.x[0], np.exp(r.x[1])
    out["lognormal"] = stats.norm.logpdf((L-m)/s) - L - np.log(s) - stats.norm.logsf(zmin(m, s))
    # alpha=1 fixo (serie logaritmica contínua) : 1 parametro
    sx, slx = x.sum(), L.sum()
    f1 = lambda lth: plc.nll([1.0, lth], sx, slx, n, xmin)
    r1 = optimize.minimize_scalar(f1, bounds=(-25, -3), method="bounded", options=dict(xatol=1e-10))
    th1 = np.exp(r1.x); lx1, g1 = plc._grid(1.0, th1, xmin); lC1 = np.log(plc.integrate.simpson(g1, x=lx1))
    out["alpha1"] = -L - th1*x - lC1
    return dict(n=n, alpha=a, theta=th, logL=ll, alpha_pl=ap, theta1=th1, lam=lam, lnmu=m, lnsigma=s), out

def compare(pw):
    res = {}
    for k in ("powerlaw", "alpha1"):    # aninhados: razao de verossimilhanças ~ chi2(1)
        R = (pw["cutoff"] - pw[k]).sum(); res[k] = dict(R=R, p=float(stats.chi2.sf(max(2*R, 0), 1)))
    for k in ("exponential", "lognormal"):   # nao aninhados: Vuong
        diff = pw["cutoff"] - pw[k]; R = diff.sum(); z = R/(np.sqrt(len(diff))*diff.std())
        res[k] = dict(R=R, z=z, p=float(2*stats.norm.sf(abs(z))))
    return res

if __name__ == "__main__":
    results = {}; pool = Pool(4)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, dd) in zip(axs, SETS):
        x = dd.votos.values.astype(float); t0 = time.time()
        res = {}
        for mode, xm in (("csn", None), ("fix5000", 5000.0)):
            if mode == "csn":
                xm, a, th, D, ll = plc.scan(x, plc.candidates(x)); nb = NB_SCAN; bm = "scan"
            else:
                a, th, ll = plc.fit(x, xm); D = plc.ks(x, xm, a, th); nb = NB_FIX; bm = "fix"
            Ds = np.array(pool.map(boot_rep, [(x, xm, a, th, 1000+i, bm) for i in range(nb)]))
            p = float((Ds >= D).mean())
            par, pw = pointwise(x, xm)
            res[mode] = dict(xmin=float(xm), ntail=int((x >= xm).sum()), alpha=float(a), theta=float(th), D=float(D),
                             p_gof=p, n_boot=nb, params=par, compare=compare(pw))
            print(f"{name} [{mode}] xmin={xm:.0f} ntail={res[mode]['ntail']} alpha={a:.3f} theta={th:.3g} (1/theta={1/th:.3g}) D={D:.4f} p_gof={p:.3f} ({nb} reps)", flush=True)
            for k, v in res[mode]["compare"].items():
                print(f"     cutoff vs {k}: R={v['R']:.1f} p={v['p']:.3g}", flush=True)
            if mode == "csn":
                xt = np.sort(x[x >= xm])[::-1]; nt = len(xt); rk = np.arange(1, len(x)+1)
                xs_all = np.sort(x)[::-1]
                ax.loglog(rk, xs_all, ".", ms=3, color="C0", label="observado")
                q = (np.arange(1, nt+1)-.5)/nt
                lx, S = plc.surv_fn(a, th, xm); curve = np.exp(np.interp(1-q, 1-S, lx))
                ax.loglog(np.arange(1, nt+1), curve, "r-", lw=1.5, label=f"PL + corte (α={a:.2f}, 1/θ={1/th:.2g})")
                ap = par["alpha_pl"]; ax.loglog(np.arange(1, nt+1), xm*q**(-1/(ap-1)), "g--", label=f"PL puro (α={ap:.2f})")
                lx1, S1 = plc.surv_fn(1.0, par["theta1"], xm); ax.loglog(np.arange(1, nt+1), np.exp(np.interp(1-q, 1-S1, lx1)), "k:", lw=2, label="α=1 (logarítmica)")
                ax.axhline(xm, color="gray", lw=.5); ax.set(title=f"{name} (x_min={xm:.0f}, cauda n={nt}, p={p:.2f})", xlabel="Ranking", ylabel="Votos")
                ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
        results[name] = res; print(f"   ({time.time()-t0:.0f}s)", flush=True)
    fig.tight_layout(); fig.savefig(OUT + ".png", dpi=130)
    json.dump(results, open(OUT + ".json", "w"), indent=1, default=float)
