#Higham covariance
#Input: testout_1.3.csv
#Output: testout_3.3.csv
import numpy as np
import pandas as pd

A_df = pd.read_csv("testout_1.3.csv", index_col=0)
cols = A_df.columns.tolist()
A = A_df.values
A = (A + A.T) / 2.0
diag_orig = np.diag(A).copy()

Y = A.copy()
deltaS = np.zeros_like(A)
tol = 1e-12
maxiter = 1000

for k in range(maxiter):
    R = Y - deltaS
    R = (R + R.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    X = (eigvecs * eigvals_clipped) @ eigvecs.T
    deltaS = X - R
    Y_old = Y
    Y = X.copy()
    np.fill_diagonal(Y, diag_orig)
    if np.linalg.norm(Y - Y_old, ord='fro') < tol:
        break

C_hat = (Y + Y.T) / 2.0
eig_final = np.linalg.eigvalsh(C_hat)
if eig_final.min() < 0:
    eigvals_f, eigvecs_f = np.linalg.eigh(C_hat)
    eigvals_f = np.clip(eigvals_f, 0.0, None)
    C_hat = (eigvecs_f * eigvals_f) @ eigvecs_f.T
    C_hat = (C_hat + C_hat.T) / 2.0
np.fill_diagonal(C_hat, diag_orig)

out = pd.DataFrame(C_hat, index=cols, columns=cols)
out.to_csv("testout_3.3.csv", float_format="%.15f")
