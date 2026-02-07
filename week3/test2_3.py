#Covariance with EW Variance (l=0.97), EW Correlation (l=0.94)
#Input: test2.csv
#Output: testout_2.3.csv
import numpy as np
import pandas as pd


lambda_var = 0.97   
lambda_cor = 0.94  
infile = "test2.csv"
outfile = "testout_2.3.csv"

data = pd.read_csv(infile)
data = data.dropna(axis=0)
cols = data.columns.tolist()
X = data.values   

T, p = X.shape
if T == 0:
    raise ValueError("No usable observations after dropna().")


def ew_weights(T, lam):

    raw = (1 - lam) * (lam ** np.arange(T-1, -1, -1))
    return raw / raw.sum()

w_var = ew_weights(T, lambda_var)                
mu_var = np.sum(w_var[:, None] * X, axis=0)     
Xc_var = X - mu_var[None, :]                     
var_ew = np.sum(w_var[:, None] * (Xc_var ** 2), axis=0)  


w_cor = ew_weights(T, lambda_cor)
mu_cor = np.sum(w_cor[:, None] * X, axis=0)
Xc_cor = X - mu_cor[None, :]
cov_ew_cor = (Xc_cor * w_cor[:, None]).T @ Xc_cor    

stds_cor = np.sqrt(np.diag(cov_ew_cor))
denom = np.outer(stds_cor, stds_cor)
with np.errstate(divide='ignore', invalid='ignore'):
    corr_ew = cov_ew_cor / denom
    corr_ew[denom == 0] = np.nan


var_ew = np.clip(var_ew, 0.0, None)
sqrt_var = np.sqrt(var_ew)
cov_final = corr_ew * np.outer(sqrt_var, sqrt_var)

cov_df = pd.DataFrame(cov_final, index=cols, columns=cols)

cov_df.to_csv(outfile, float_format="%.15f")