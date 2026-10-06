import sys, numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T, RATIO = 5000, 1.6
df = pd.read_csv(sys.argv[1]); out = sys.argv[2]
fed = df[(df.DS_CARGO == "Deputado Federal") & (df.votos >= T)]

def nb_cdf(k, r, mu): return stats.nbinom.cdf(k, r, r/(r+mu))
def ln_cdf(k, m, s): return stats.norm.cdf((np.log(np.maximum(k, 1e-9))-m)/s)
def nb_nll(p, x):
    r, mu = np.exp(p); pr = r/(r+mu)
    return -(stats.nbinom.logpmf(x, r, pr) - stats.nbinom.logsf(T-1, r, pr)).sum()
def ln_nll(p, x):
    m, s = p[0], np.exp(p[1])
    pm = ln_cdf(x+.5, m, s) - ln_cdf(x-.5, m, s)
    return -(np.log(np.maximum(pm, 1e-300)) - np.log(1-ln_cdf(T-.5, m, s))).sum()


from scipy import integrate
def ls_G(a, th):
    """G(a) = sum_{k>=a} exp(-k th)/k = int_th^inf exp(-a s)/(1-exp(-s)) ds  (a inteiro >= 1)."""
    f = lambda y: np.exp(-y)/(-np.expm1(-(th + y/a)))
    return np.exp(-a*th)/a*integrate.quad(f, 0, 60, limit=200)[0]
def ls_nll(lth, x):
    th = np.exp(lth)
    return -(-x*th - np.log(x) - np.log(ls_G(T, th))).sum()
def fit_ls(x):
    a = optimize.minimize_scalar(ls_nll, bounds=(-16, 0), args=(x,), method="bounded", options=dict(xatol=1e-9))
    grid = np.linspace(-16, 0, 80); prof = min((ls_nll(g, x), g) for g in grid)
    print(f"   LS perfil: melhor theta na grade={np.exp(prof[1]):.3g} logL={-prof[0]:.1f} | bounded: theta={np.exp(a.x):.3g} logL={-a.fun:.1f}")
    return a

def fit_nb(x):  # Nelder-Mead com varios chutes; r em [1e-3, 1e3] via penalidade; checa convergencia
    def f(p):
        if not (np.log(1e-3) <= p[0] <= np.log(1e3)) or not np.isfinite(p).all(): return 1e12
        v = nb_nll(p, x); return v if np.isfinite(v) else 1e12
    best = None
    for r0 in (0.05, 0.3, 1, 5):
        a = optimize.minimize(f, [np.log(r0), np.log(x.mean())], method="Nelder-Mead",
                              options=dict(xatol=1e-8, fatol=1e-8, maxiter=4000, maxfev=8000))
        if best is None or a.fun < best.fun: best = a
    # perfil: varre r numa grade e otimiza mu, para conferir
    grid = np.exp(np.linspace(np.log(1e-3), np.log(5), 60))
    prof = [(optimize.minimize_scalar(lambda lm: nb_nll([np.log(g), lm], x), bounds=(0, 16), method="bounded").fun, g) for g in grid]
    pbest = min(prof)
    print(f"   NB perfil: melhor r na grade={pbest[1]:.3g} logL={-pbest[0]:.1f} | Nelder-Mead: r={np.exp(best.x[0]):.3g} logL={-best.fun:.1f}")
    return best

def expected_probs(edges, model, par):
    """Probabilidade (condicionada a >=T) de cada bin [a,b), votos inteiros."""
    if model == "ls":
        th = par[0]; G = np.array([ls_G(int(e), th) for e in edges])
        return (G[:-1]-G[1:])/ls_G(T, th)
    if model == "nb":
        r, mu = par; F = lambda k: nb_cdf(k-1, r, mu); z = 1-F(T)
    else:
        m, s = par; F = lambda k: ln_cdf(k-.5, m, s); z = 1-F(T)
    return np.diff(F(edges))/z   # P(a<=X<b) = F(b)-F(a), com F(k)=P(X<k)

def merge_tail(obs, exp, minexp=5):
    """Junta bins a partir do topo ate que cada um tenha esperado >= minexp (em ambos os modelos)."""
    o, e = list(obs), [list(v) for v in exp]
    idx = list(range(len(o)))
    groups = [[i] for i in idx]
    def ok(g): return all(sum(ev[i] for i in g) >= minexp for ev in e)
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for j in range(len(groups)-1, -1, -1):
            if not ok(groups[j]):
                k = j-1 if j > 0 else j+1
                lo, hi = sorted((j, k))
                groups[lo] = groups[lo]+groups[hi]; del groups[hi]; changed = True; break
    return groups

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
rows = []
for ax, (t, d) in zip(axs, [("Brasil", fed), ("SP", fed[fed.SG_UF == "SP"]), ("BA", fed[fed.SG_UF == "BA"])]):
    x = d.votos.values.astype(int); n = len(x)
    a = fit_nb(x); r, mu = np.exp(a.x)
    b = optimize.minimize(ln_nll, [np.log(x).mean(), np.log(np.log(x).std())], args=(x,), method="Nelder-Mead")
    m, s = b.x[0], np.exp(b.x[1])
    c = fit_ls(x); th = np.exp(c.x)
    print(f"   logL: NB={-a.fun:.1f} (2 par) | LN={-b.fun:.1f} (2 par) | LS={-c.fun:.1f} (1 par)  AIC: NB={2*2+2*a.fun:.1f} LN={2*2+2*b.fun:.1f} LS={2*1+2*c.fun:.1f}")
    k = int(np.ceil(np.log(x.max()/T)/np.log(RATIO)))+1
    edges = np.unique(np.round(T*RATIO**np.arange(k+1)).astype(int))
    obs = np.histogram(x, bins=edges)[0]
    pn, pl = expected_probs(edges, "nb", (r, mu)), expected_probs(edges, "ln", (m, s))
    en, el = n*pn, n*pl
    es = n*expected_probs(edges, "ls", (th,))
    groups = merge_tail(obs, (en, el, es))
    chi = {}
    for nm, e, npar in (("NB", en, 2), ("LN", el, 2), ("LS", es, 1)):
        cc = sum((obs[g].sum()-e[g].sum())**2/e[g].sum() for g in groups)
        dof = len(groups)-1-npar
        chi[nm] = (cc, dof, stats.chi2.sf(cc, dof))
    print(f"{t}: n={n} bins={len(obs)} (apos juntar: {len(groups)}) | NB r={r:.3g} mu={mu:.0f} | LN m={m:.2f} s={s:.2f}")
    for nm in chi: print(f"   {nm}: chi2={chi[nm][0]:.1f} gl={chi[nm][1]} p={chi[nm][2]:.3g}")
    w = np.diff(edges); mid = np.sqrt(edges[:-1]*edges[1:])
    # densidade (candidatos por voto) com erro de Poisson
    nz = obs > 0
    ax.errorbar(mid[nz], obs[nz]/w[nz], yerr=np.sqrt(obs[nz])/w[nz], fmt="o", ms=4, capsize=2, label="observado")
    ax.plot(mid, en/w, "r-", label="binomial negativa"); ax.plot(mid, el/w, "g--", label="lognormal"); ax.plot(mid, es/w, "k:", lw=2, label="logarítmica (1 parâmetro)")
    ax.set(xscale="log", yscale="log", xlabel="Votos (bins geométricos, razão 1,6)", ylabel="Candidatos por voto",
           title=f"{t} (≥{T} votos, n={n})")
    ax.text(.03, .05, "χ² (p): " + "\n        ".join(f"{k_} {v[0]:.0f} ({v[2]:.2g})" for k_, v in chi.items()),
            transform=ax.transAxes, fontsize=9, va="bottom")
    ax.legend(); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(out, dpi=130)
