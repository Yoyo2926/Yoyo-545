#VaR from Simulation -- compare to 8.2 values
#Input: test7_2.csv
#Output: testout_8.3.csv
import numpy as np
import pandas as pd
from scipy.stats import t

INFILE = "test7_2.csv"
OUTFILE = "testout_8.3.csv"

ALPHA = 0.05
NSIM = 100000

df = pd.read_csv(INFILE)
x = df.iloc[:, 0].dropna().values

mu = x.mean()

df_fit, loc_fit, scale_fit = t.fit(x, floc=mu)

rng = np.random.RandomState(42)
sim = t.rvs(df_fit, loc=loc_fit, scale=scale_fit,
            size=NSIM, random_state=rng)

q = np.percentile(sim, ALPHA * 100)

var_abs = -q
var_diff = mu + var_abs

pd.DataFrame([[var_abs, var_diff]],
             columns=["VaR Absolute", "VaR Diff from Mean"]).to_csv(
    OUTFILE, index=False, float_format="%.15f"
)
