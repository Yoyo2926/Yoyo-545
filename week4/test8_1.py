#Var from Normal Distribution
#Input: test7_1.csv
#Output: testout_8.1.csv
import numpy as np
import pandas as pd
from scipy.stats import norm

INFILE = "test7_1.csv"
OUTFILE = "testout_8.1.csv"
alpha = 0.05

df = pd.read_csv(INFILE)
x = df.iloc[:, 0].dropna().values
mu = x.mean()
sigma = x.std(ddof=1)
var_abs = - (mu + sigma * norm.ppf(alpha))
var_diff_from_mean = mu + var_abs

pd.DataFrame([[var_abs, var_diff_from_mean]], columns=["VaR Absolute", "VaR Diff from Mean"]).to_csv(
    OUTFILE, index=False, float_format="%.15f"
)
