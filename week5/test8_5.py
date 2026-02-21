#ES from T Distribution
#Input: test7_2.csv
#Output: testout_8.5.csv
import pandas as pd
import numpy as np
from scipy.stats import t

data = pd.read_csv("test7_2.csv")
x = data.iloc[:, 0].values

nu, mu, sigma = t.fit(x)

alpha = 0.95

z = t.ppf(1 - alpha,nu) #CDF's quantile

fz = t.pdf(z, nu) #PDF

ET_z = - (nu / (nu - 1.0)) * (1.0 + z**2 / nu) * fz / (1.0 - alpha)

ES = mu + sigma * ET_z
ES_abs = abs(ES)
diff = mu-ES

pd.DataFrame({"ES Absolute": [ES_abs],"ES Diff from Mean":[diff]}).to_csv("testout_8.5.csv", index=False)
