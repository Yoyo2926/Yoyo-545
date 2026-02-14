#Var from T Distribution
#Input: test7_2.csv
#Output: testout_8.2.csv
import numpy as np
import pandas as pd
from scipy.stats import t

INFILE = "test7_2.csv"
OUTFILE = "testout_8.2.csv"
alpha = 0.05

df = pd.read_csv(INFILE)
x = df.iloc[:, 0].dropna().values

mu = x.mean()
df_fit, loc_fit, scale_fit = t.fit(x, floc=mu)

var_abs = - (loc_fit + scale_fit * t.ppf(alpha, df_fit))
var_diff_from_mean = mu + var_abs

pd.DataFrame([[var_abs, var_diff_from_mean]],
             columns=["VaR Absolute", "VaR Diff from Mean"]).to_csv(
    OUTFILE, index=False, float_format="%.15f"
)
