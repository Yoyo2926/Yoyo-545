#near_psd covariance
#Input: testout_1.3.csv
#Output: testout_3.1.csv
import numpy as np
import pandas as pd

C = pd.read_csv("testout_1.3.csv", index_col=0).values
C = (C + C.T) / 2.0
vars = np.diag(C).copy()
std = np.sqrt(vars)
std[std == 0] = 1.0
corr = C / np.outer(std, std)
corr = (corr + corr.T) / 2.0
eigvals, eigvecs = np.linalg.eigh(corr)
eigvals_pos = np.clip(eigvals, 0.0, None)
sqrt_L = np.sqrt(eigvals_pos)
S = eigvecs
u = (S**2) @ eigvals_pos
u[u <= 0] = 1e-16
t = 1.0 / np.sqrt(u)
T = np.diag(t)
B = T @ S @ np.diag(sqrt_L)
C_hat_corr = B @ B.T
C_hat_corr = (C_hat_corr + C_hat_corr.T) / 2.0
np.fill_diagonal(C_hat_corr, 1.0)
cov_hat = C_hat_corr * np.outer(std, std)
cols = pd.read_csv("testout_1.3.csv", nrows=0).columns.tolist()[1:]
out = pd.DataFrame(cov_hat, index=cols, columns=cols)
out.to_csv("testout_3.1.csv", float_format="%.15f")
