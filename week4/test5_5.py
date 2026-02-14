#PCA Simulation, 99% explained, 0 mean - 100,000 simulations compare input vs output covariance
#Input: test5_2.csv
#Output: testout_5.5.csv
import numpy as np
import pandas as pd

INFILE = "test5_2.csv"
OUTFILE = "testout_5.5.csv"
NSIM = 100000
EXPLAIN = 0.99

df = pd.read_csv(INFILE, header=0)
labels = list(df.columns)
Sigma = df.values
Sigma = (Sigma + Sigma.T) / 2.0

eigvals, eigvecs = np.linalg.eigh(Sigma)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

total = eigvals.sum()
cum = np.cumsum(eigvals) / total
m = int(np.searchsorted(cum, EXPLAIN) + 1)
m = max(1, min(m, Sigma.shape[0]))

L = eigvecs[:, :m] * np.sqrt(eigvals[:m])

np.random.seed(42)
Z = np.random.normal(size=(NSIM, m))
X = Z @ L.T

Sigma_out = (X.T @ X) / NSIM
Sigma_out = (Sigma_out + Sigma_out.T) / 2.0

pd.DataFrame(Sigma_out, columns=labels).to_csv(
    OUTFILE, index=False, header=True, float_format="%.15f"
)
