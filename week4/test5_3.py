#Normal Simulation nonPSD Input, 0 mean, near_psd fix - 100,000 simulations, compare input vs output covariance
#Input: test5_3.csv
#Output: testout_5.3.csv
import numpy as np
import pandas as pd

INFILE = "test5_3.csv"
OUTFILE = "testout_5.3.csv"
NSIM = 100000

df = pd.read_csv(INFILE, header=0)
labels = list(df.columns)
A = df.values
A = (A + A.T) / 2.0

def higham_corr(C, tol=1e-12, maxiter=1000):
    Y = C.copy()
    deltaS = np.zeros_like(C)
    for k in range(maxiter):
        R = Y - deltaS
        R = (R + R.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals_clipped = np.clip(eigvals, 0.0, None)
        X = (eigvecs * eigvals_clipped) @ eigvecs.T
        deltaS = X - R
        Y_old = Y.copy()
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        if np.linalg.norm(Y - Y_old, ord='fro') < tol:
            break
    C_hat = (Y + Y.T) / 2.0
    eig_final = np.linalg.eigvalsh(C_hat)
    if eig_final.min() < 0:
        eigvals_f, eigvecs_f = np.linalg.eigh(C_hat)
        eigvals_f = np.clip(eigvals_f, 0.0, None)
        C_hat = (eigvecs_f * eigvals_f) @ eigvecs_f.T
        C_hat = (C_hat + C_hat.T) / 2.0
    np.fill_diagonal(C_hat, 1.0)
    return C_hat

d = np.sqrt(np.diag(A))
d_safe = np.where(d == 0, 1.0, d)
D_inv = np.diag(1.0 / d_safe)
R = D_inv @ A @ D_inv
R = (R + R.T) / 2.0

R_hat = higham_corr(R)

Sigma_hat = np.diag(d) @ R_hat @ np.diag(d)
Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0

eigvals, eigvecs = np.linalg.eigh(Sigma_hat)
eigvals = np.clip(eigvals, 0.0, None)
L = eigvecs * np.sqrt(eigvals)

np.random.seed(42)
Z = np.random.normal(size=(NSIM, Sigma_hat.shape[0]))
X = Z @ L.T

Sigma_out = (X.T @ X) / NSIM
Sigma_out = (Sigma_out + Sigma_out.T) / 2.0

pd.DataFrame(Sigma_out, columns=labels).to_csv(
    OUTFILE, index=False, header=True, float_format="%.15f"
)
