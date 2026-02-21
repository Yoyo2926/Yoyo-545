# VaR/ES on 2 levels from simulated values - Copula
# Input: test9_1_portfolio.csv, test9_1_returns.csv
# Output: testout_9.1.csv

import numpy as np
import pandas as pd
from scipy.stats import t, norm

IN_PORT = "test9_1_portfolio.csv"
IN_RET  = "test9_1_returns.csv"
OUT = "testout_9.1.csv"

ALPHA = 0.95
NSIM = 200000
SEED = 42
USE_T_COPULA = True

pf = pd.read_csv(IN_PORT)
rets_df = pd.read_csv(IN_RET)

if not np.issubdtype(rets_df.iloc[:,0].dtype, np.number):
    rets_df = rets_df.iloc[:,1:]

rets_df = rets_df.select_dtypes(include=[np.number])

assets = pf["Stock"].astype(str).values
holdings = pf["Holding"].astype(float).values
prices0 = pf["Starting Price"].astype(float).values
dists = pf["Distribution"].astype(str).values

if list(assets) != list(rets_df.columns):
    raise ValueError("Asset order mismatch.")

pos_values = holdings * prices0
n_assets = len(assets)
p = 1 - ALPHA

per_rows = []

for i in range(n_assets):
    x = rets_df[assets[i]].values
    pos = pos_values[i]

    if dists[i].strip().lower().startswith("t"):
        df_i, loc_i, scale_i = t.fit(x)
        z = t.ppf(p, df_i)
        VaR_money = -(loc_i + scale_i * z) * pos
        nu = df_i
        fz = t.pdf(z, nu)
        ET = - (nu/(nu-1.0)) * (1.0 + z*z/nu) * fz / (1.0 - ALPHA)
        ES_money = -(loc_i + scale_i * ET) * pos
    else:
        mu = x.mean()
        sigma = x.std(ddof=1)
        z = norm.ppf(p)
        VaR_money = -(mu + sigma * z) * pos
        phi = norm.pdf(z)
        ES_money = -(mu - sigma * phi / (1.0 - ALPHA)) * pos

    per_rows.append({
        "Stock": assets[i],
        "VaR95": float(VaR_money),
        "ES95": float(ES_money),
        "VaR95_Pct": float(VaR_money / pos),
        "ES95_Pct": float(ES_money / pos)
    })

params = []

for i in range(n_assets):
    x = rets_df[assets[i]].values
    if dists[i].strip().lower().startswith("t"):
        df_i, loc_i, scale_i = t.fit(x)
        params.append(("t", float(df_i), float(loc_i), float(scale_i)))
    else:
        mu = x.mean()
        sigma = x.std(ddof=1)
        params.append(("norm", float(mu), float(sigma)))

U = np.zeros_like(rets_df.values)

for i in range(n_assets):
    if params[i][0] == "t":
        _, df_i, loc_i, scale_i = params[i]
        U[:, i] = t.cdf(rets_df.iloc[:, i].values, df_i, loc=loc_i, scale=scale_i)
    else:
        _, mu, sigma = params[i]
        U[:, i] = norm.cdf(rets_df.iloc[:, i].values, loc=mu, scale=sigma)

U = np.clip(U, 1e-10, 1-1e-10)
Z = norm.ppf(U)
corr = np.corrcoef(Z.T)

eps = 1e-10
k = 0

while True:
    try:
        L = np.linalg.cholesky(corr + np.eye(n_assets)*(eps*k))
        break
    except np.linalg.LinAlgError:
        k += 1

rng = np.random.RandomState(SEED)

if USE_T_COPULA:
    t_dfs = [pinfo[1] for pinfo in params if pinfo[0] == "t"]
    df_cop = float(min(t_dfs)) if len(t_dfs) > 0 else 5.0
    Z_sim = rng.normal(size=(NSIM, n_assets)).dot(L.T)
    chi2 = rng.chisquare(df_cop, size=NSIM)
    T_sim = Z_sim / np.sqrt(chi2[:, None] / df_cop)
    U_sim = t.cdf(T_sim, df_cop)
else:
    Z_sim = rng.normal(size=(NSIM, n_assets)).dot(L.T)
    U_sim = norm.cdf(Z_sim)

U_sim = np.clip(U_sim, 1e-10, 1-1e-10)

sim_returns = np.zeros_like(U_sim)

for i in range(n_assets):
    if params[i][0] == "t":
        _, df_i, loc_i, scale_i = params[i]
        sim_returns[:, i] = t.ppf(U_sim[:, i], df_i, loc=loc_i, scale=scale_i)
    else:
        _, mu, sigma = params[i]
        sim_returns[:, i] = norm.ppf(U_sim[:, i], loc=mu, scale=sigma)

loss_sim = (-sim_returns) * pos_values
loss_total = loss_sim.sum(axis=1)

VaR95_tot = np.percentile(loss_total, ALPHA*100)
ES95_tot = loss_total[loss_total >= VaR95_tot].mean()

per_rows.append({
    "Stock": "Total",
    "VaR95": float(VaR95_tot),
    "ES95": float(ES95_tot),
    "VaR95_Pct": float(VaR95_tot / pos_values.sum()),
    "ES95_Pct": float(ES95_tot / pos_values.sum())
})

out = pd.DataFrame(per_rows, columns=["Stock","VaR95","ES95","VaR95_Pct","ES95_Pct"])
out.to_csv(OUT, index=False, float_format="%.15f")