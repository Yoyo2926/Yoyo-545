#Normal Simulation PSD Input 0 mean - 100,000 simulations, compare input vs output covariance
#Input: test5_2.csv
#Output: testout_5.2.csv
import numpy as np
import pandas as pd

INFILE = "test5_2.csv"
OUTFILE = "testout_5.2.csv"
NSIM = 100000

df = pd.read_csv(INFILE, header=0)
Sigma_in = df.values
Sigma_in = (Sigma_in + Sigma_in.T) / 2.0
labels = list(df.columns)

eigvals, eigvecs = np.linalg.eigh(Sigma_in)
eigvals = np.clip(eigvals, 0.0, None)
L = eigvecs * np.sqrt(eigvals)

np.random.seed(42)

Z = np.random.normal(size=(NSIM, Sigma_in.shape[0]))
X = Z @ L.T

Sigma_out = np.cov(X, rowvar=False, ddof=0)
Sigma_out = (Sigma_out + Sigma_out.T) / 2.0

pd.DataFrame(Sigma_out, columns=labels).to_csv(
    OUTFILE, index=False, header=True, float_format="%.15f"
)
