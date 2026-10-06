import sys, numpy as np, pandas as pd
from scipy import stats, optimize, special
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df = pd.read_csv(sys.argv[1]); out = sys.argv[2]
fed = df[df.DS_CARGO == "Deputado Federal"]
def nb_nll(p, x):
    r, mu = np.exp(p); pr = r/(r+mu)
    return -stats.nbinom.logpmf(x, r, pr).sum()
def fit(x):
    x = np.asarray(x, int)
    res = optimize.minimize(nb_nll, [0, np.log(x.mean())], args=(x,), method="Nelder-Mead")
    r, mu = np.exp(res.x); ll_nb = -res.fun
    # lognormal discretizada: P(k)=F(k+.5)-F(k-.5), k>=1 (log-espaço)
    lx = np.log(x[x>0]); m, s = lx.mean(), lx.std()
    cdf = lambda k: stats.norm.cdf((np.log(np.maximum(k,1e-9))-m)/s)
    ll_ln = np.log(np.maximum(cdf(x+.5)-cdf(x-.5),1e-300)).sum()
    return r, mu, ll_nb, ll_ln, m, s
fig, axs = plt.subplots(1,3, figsize=(16,4.8))
for ax,(t,d) in zip(axs,[("Brasil",fed),("SP",fed[fed.SG_UF=="SP"]),("AC",fed[fed.SG_UF=="AC"])]):
    x = np.sort(d.votos.values)[::-1]; n = len(x)
    r, mu, lnb, lln, m, s = fit(x)
    print(t, "n=%d media=%.0f var/media=%.0f  NB: r=%.3f mu=%.0f logL=%.0f | lognormal logL=%.0f"%(n,x.mean(),x.var()/x.mean(),r,mu,lnb,lln))
    q = (np.arange(1,n+1)-.5)/n
    nbq = stats.nbinom.isf(q, r, r/(r+mu)); lnq = np.exp(m+s*stats.norm.isf(q))
    rk = np.arange(1,n+1)
    ax.loglog(rk,x,".",ms=3,label="observado"); ax.loglog(rk,np.maximum(nbq,.5),"r-",label="binomial negativa"); ax.loglog(rk,lnq,"g--",label="lognormal")
    ax.set(title=t,xlabel="Ranking",ylabel="Votos"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(out,dpi=130)
