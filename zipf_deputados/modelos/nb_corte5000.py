import sys, numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = 5000
df = pd.read_csv(sys.argv[1]); out = sys.argv[2]
fed = df[(df.DS_CARGO == "Deputado Federal") & (df.votos >= T)]
def nb_nll(p, x):
    r, mu = np.exp(p); pr = r/(r+mu)
    return -(stats.nbinom.logpmf(x, r, pr) - stats.nbinom.logsf(T-1, r, pr)).sum()
def ln_cdf(k, m, s): return stats.norm.cdf((np.log(k)-m)/s)
def ln_nll(p, x):
    m, s = p[0], np.exp(p[1])
    pm = ln_cdf(x+.5, m, s) - ln_cdf(x-.5, m, s)
    return -(np.log(np.maximum(pm, 1e-300)) - np.log(1-ln_cdf(T-.5, m, s))).sum()
fig, axs = plt.subplots(1,3, figsize=(16,4.8))
for ax,(t,d) in zip(axs,[("Brasil",fed),("SP",fed[fed.SG_UF=="SP"]),("BA",fed[fed.SG_UF=="BA"])]):
    x = np.sort(d.votos.values)[::-1].astype(int); n = len(x)
    a = optimize.minimize(nb_nll,[0,np.log(x.mean())],args=(x,),method="Nelder-Mead")
    r, mu = np.exp(a.x); pr = r/(r+mu)
    b = optimize.minimize(ln_nll,[np.log(x).mean(),np.log(np.log(x).std())],args=(x,),method="Nelder-Mead")
    m, s = b.x[0], np.exp(b.x[1])
    print("%s n=%d mediana=%.0f max=%d | NB r=%.2f mu=%.0f logL=%.0f | lognormal m=%.2f s=%.2f logL=%.0f"%(t,n,np.median(x),x.max(),r,mu,-a.fun,m,s,-b.fun))
    q = (np.arange(1,n+1)-.5)/n
    nbq = stats.nbinom.isf(q*stats.nbinom.sf(T-1,r,pr), r, pr)
    lnq = np.exp(m+s*stats.norm.isf(q*(1-ln_cdf(T-.5,m,s))))
    rk = np.arange(1,n+1)
    ax.loglog(rk,x,".",ms=3,label="observado"); ax.loglog(rk,nbq,"r-",label="binomial negativa"); ax.loglog(rk,lnq,"g--",label="lognormal")
    ax.set(title=f"{t} (≥{T} votos, n={n})",xlabel="Ranking",ylabel="Votos"); ax.legend(); ax.grid(alpha=.3,which="both")
fig.tight_layout(); fig.savefig(out,dpi=130)
