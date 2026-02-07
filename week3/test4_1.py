#chol_psd
#Input: testout_3.1.csv
#Output: testout_4.1.csv
import numpy as np
import pandas as pd

def near_pd(A, tol=1e-12, max_iter=1000):
    A = (A + A.T) / 2.0
    n = A.shape[0]
    Y = A.copy()
    deltaS = np.zeros_like(A)
    for k in range(max_iter):
        R = Y - deltaS
        R = (R + R.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals_clipped = np.clip(eigvals, 0.0, None)
        X = (eigvecs * eigvals_clipped) @ eigvecs.T
        deltaS = X - R
        Y_old = Y
        Y = X.copy()
        np.fill_diagonal(Y, np.diag(A))
        if np.linalg.norm(Y - Y_old, ord='fro') < tol:
            break
    C = (Y + Y.T) / 2.0
    eig_final = np.linalg.eigvalsh(C)
    if eig_final.min() < 0:
        eigvals_f, eigvecs_f = np.linalg.eigh(C)
        eigvals_f = np.clip(eigvals_f, 0.0, None)
        C = (eigvecs_f * eigvals_f) @ eigvecs_f.T
        C = (C + C.T) / 2.0
    return C

df = pd.read_csv("testout_3.1.csv", index_col=0)
cols = df.index.tolist()
A = df.values
A = (A + A.T) / 2.0

try:
    L = np.linalg.cholesky(A)
except np.linalg.LinAlgError:
    A_pd = near_pd(A)
    try:
        L = np.linalg.cholesky(A_pd)
    except np.linalg.LinAlgError:
        jitter = 1e-12
        max_jitter_iters = 100
        for i in range(max_jitter_iters):
            try:
                L = np.linalg.cholesky(A_pd + np.eye(A_pd.shape[0]) * jitter)
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            raise RuntimeError("Cholesky failed even after near-PD and jittering.")
L_df = pd.DataFrame(L, index=cols, columns=cols)
L_df.to_csv("testout_4.1.csv", float_format="%.15f")

