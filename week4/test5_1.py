#Normal Simulation PD Input 0 mean - 100,000 simulations, compare input vs output covariance
#Input: test5_1.csv
#Output: testout_5.1.csv
import numpy as np
import pandas as pd

Sigma = pd.read_csv("test5_1.csv").values
n = Sigma.shape[0]
NSIM = 100000

np.random.seed(42)   

L = np.linalg.cholesky(Sigma)

Z = np.random.normal(size=(NSIM, n))
X = Z @ L.T

Sigma_sim = np.cov(X, rowvar=False, ddof=0)

pd.DataFrame(Sigma_sim, columns=['x1','x2','x3','x4','x5']).to_csv(
    "testout_5.1.csv", index=False
)
