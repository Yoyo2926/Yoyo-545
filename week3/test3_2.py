#near_psd correlation
#Input: testout_1.4.csv
#Output: testout_3.2.csv
import numpy as np
import pandas as pd

df = pd.read_csv("testout_1.4.csv", index_col=0)
C = df.values
C = (C + C.T) / 2.0
eigvals, eigvecs = np.linalg.eigh(C)
eigvals_pos = np.clip(eigvals, 0.0, None)
sqrt_L = np.sqrt(eigvals_pos)
S = eigvecs
u = (S**2) @ eigvals_pos
u[u <= 0] = 1e-16
t = 1.0 / np.sqrt(u)
T = np.diag(t)
B = T @ S @ np.diag(sqrt_L)
C_hat = B @ B.T
C_hat = (C_hat + C_hat.T) / 2.0
np.fill_diagonal(C_hat, 1.0)
out = pd.DataFrame(C_hat, index=df.index, columns=df.columns)
out.to_csv("testout_3.2.csv", float_format="%.15f")
