"""Lei de potencia com corte exponencial, p(x) ~ x^-alpha exp(-theta x), x >= xmin (Clauset, Shalizi & Newman 2009).
Aproximacao continua (votos >= xmin sao inteiros grandes)."""
import numpy as np
from scipy import optimize, stats, integrate

NG = 6000
def _grid(alpha, theta, xmin):
    xmax = xmin + 60.0/theta
    lx = np.linspace(np.log(xmin), np.log(xmax), NG)
    return lx, np.exp((1-alpha)*lx - theta*np.exp(lx))      # densidade nao normalizada, em d ln x

def nll(p, sx, slx, n, xmin):
    alpha, lth = p
    if not (-3 < alpha < 10 and -25 < lth < 0): return 1e18
    th = np.exp(lth)
    if th*xmin > 500: return 1e18
    lx, g = _grid(alpha, th, xmin)
    C = integrate.simpson(g, x=lx)
    if not (C > 0 and np.isfinite(C)): return 1e18
    return alpha*slx + th*sx + n*np.log(C)

def fit(x, xmin, starts=None):
    x = x[x >= xmin]; n = len(x); sx, slx = x.sum(), np.log(x).sum()
    a0 = 1 + n/np.log(x/xmin).sum()
    best = None
    for s in (starts or [(1.0, -11.0), (0.5, -12.0), (1.5, -10.0), (min(a0, 9), -15.0)]):
        r = optimize.minimize(nll, s, args=(sx, slx, n, xmin), method="Nelder-Mead",
                              options=dict(xatol=1e-7, fatol=1e-9, maxiter=3000))
        if best is None or r.fun < best.fun: best = r
    return best.x[0], np.exp(best.x[1]), -best.fun   # alpha, theta, logL

def surv_fn(alpha, theta, xmin):
    lx, g = _grid(alpha, theta, xmin)
    cum = integrate.cumulative_trapezoid(g, lx, initial=0.0)
    return lx, 1 - cum/cum[-1]

def ks(x, xmin, alpha, theta):
    x = np.sort(x[x >= xmin]); n = len(x)
    lx, S = surv_fn(alpha, theta, xmin)
    Sm = np.interp(np.log(x), lx, S)
    i = np.arange(1, n+1)
    return max(np.max(np.abs((n-i+1)/n - Sm)), np.max(np.abs((n-i)/n - Sm)))   # empirica antes/depois do degrau

def scan(x, cands):
    best = None
    for xm in cands:
        a, th, ll = fit(x, xm)
        D = ks(x, xm, a, th)
        if best is None or D < best[3]: best = (xm, a, th, D, ll)
    return best

def candidates(x, nmin=30, k=25):
    xs = np.sort(x)[::-1]
    sizes = np.unique(np.geomspace(nmin, len(xs)*0.9, k).astype(int))
    return np.unique(xs[sizes-1])          # xmin tal que cauda tem ~size pontos

def sample_tail(m, alpha, theta, xmin, rng):
    lx, S = surv_fn(alpha, theta, xmin)
    u = rng.random(m)                         # S(x)=u  -> x
    return np.exp(np.interp(1-u, 1-S, lx))     # 1-S crescente
