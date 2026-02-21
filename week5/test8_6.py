#ES from Simulation -- compare to 8.5 values
#Input: test7_2.csv
#Output: testout_8.6.csv
import numpy as np
import pandas as pd
from scipy.stats import t

data = pd.read_csv("test7_2.csv")
x = data.iloc[:, 0].values

n = 100000

nu, mu, sigma = t.fit(x)

np.random.RandomState(42)

sim = t.rvs(nu, mu, sigma, n)

alpha = 0.95

var = np.quantile(sim, (1-alpha))

ES = sim[sim < var].mean()

ES_abs = abs(ES)
diff = mu-ES

pd.DataFrame({"ES Absolute": [ES_abs],"ES Diff from Mean":[diff]}).to_csv("testout_8.6.csv", index=False)
