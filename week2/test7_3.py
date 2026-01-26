import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize

data = pd.read_csv("test7_3.csv")

y = data["y"].values
X = data[["x1", "x2", "x3"]].values
X = np.column_stack([np.ones(len(X)), X])

def neg_ll(params):
    beta = params[:-2]
    nu = params[-2]
    sigma = params[-1]
    resid = y - X @ beta
    return -np.sum(t.logpdf(resid / sigma, nu) - np.log(sigma))

init = np.r_[np.zeros(X.shape[1]), 5, 1]

res = minimize(
    neg_ll,
    init,
    method="L-BFGS-B",
    bounds=[(None, None)] * X.shape[1] + [(2.1, None), (1e-6, None)]
)

beta = res.x[:-2]
nu, sigma = res.x[-2:]

pd.DataFrame(
    [list(beta) + [nu, sigma]],
    columns=["alpha", "beta1", "beta2", "beta3", "nu", "sigma"]
).to_csv("testout_7.3.csv", index=False)
