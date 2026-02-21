#ES From Normal Distribution
#Input: test7_1.csv
#Output: testout_8.4.csv
import pandas as pd
import numpy as np
from scipy.stats import norm

data = pd.read_csv("test7_1.csv")
x = data.iloc[:, 0].values

mu = np.mean(x)
sigma = np.std(x, ddof=1)

alpha = 0.95

z = norm.ppf(1 - alpha)

ES_z = (-norm.pdf(z)) / (1 - alpha)
ES = mu + sigma * ES_z
ES_abs = abs(ES)
diff = mu-ES

pd.DataFrame({"ES Absolute": [ES_abs],"ES Diff from Mean":[diff]}).to_csv("testout_8.4.csv", index=False)
