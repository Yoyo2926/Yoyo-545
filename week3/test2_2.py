#EW Correlation, lambd=0.94
#Input: test2.csv
#Output: testout_2.2.csv
import numpy as np
import pandas as pd

lambda_ = 0.94
infile = "test2.csv"
outfile = "testout_2.2.csv"

data = pd.read_csv(infile)
data = data.dropna(axis=0)
cols = data.columns.tolist()
X = data.values 

T, p = X.shape
if T == 0:
    raise ValueError("No usable observations after dropna().")

raw_weights = (1 - lambda_) * (lambda_ ** np.arange(T-1, -1, -1))
weights = raw_weights / raw_weights.sum() 

mu = np.sum(weights[:, None] * X, axis=0)  

X_centered = X - mu[None, :]
weighted_cov = (X_centered * weights[:, None]).T @ X_centered 

stds = np.sqrt(np.diag(weighted_cov))

denom = np.outer(stds, stds)
with np.errstate(divide='ignore', invalid='ignore'):
    corr = weighted_cov / denom
    corr[denom == 0] = np.nan

corr_df = pd.DataFrame(corr, index=cols, columns=cols)

corr_df.to_csv(outfile, float_format="%.15f")
