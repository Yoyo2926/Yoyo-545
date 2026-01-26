import pandas as pd
from scipy.stats import t

data = pd.read_csv("test7_2.csv")
x = data.iloc[:, 0].values

nu, mu, sigma = t.fit(x)

pd.DataFrame([[nu, mu, sigma]],
             columns=["nu", "mu", "sigma"]) \
  .to_csv("testout_7.2.csv", index=False)
