#EW Covariance, lambd=0.97
#Input: test2.csv
#Output: testout_2.1.csv
import numpy as np
import pandas as pd

data = pd.read_csv("test2.csv")
data = data.dropna(axis=0)  


cols = data.columns.tolist()
X = data.values        
T, p = X.shape
if T == 0:
    raise ValueError("invalid file")


lambda_ = 0.97
raw_weights = (1 - lambda_) * (lambda_ ** np.arange(T-1, -1, -1))
weights = raw_weights / raw_weights.sum()  

mu = np.sum(weights[:, None] * X, axis=0)  

X_centered = X - mu[None, :]                

weighted_cov = (X_centered * weights[:, None]).T @ X_centered   


cov_df = pd.DataFrame(weighted_cov, index=cols, columns=cols)


cov_df.to_csv("testout_2.1.csv", float_format="%.15f")