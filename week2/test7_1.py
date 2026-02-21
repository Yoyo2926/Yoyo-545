#Fit Normal Distribution
#均值和方差
import pandas as pd
import numpy as np

data = pd.read_csv("test7_1.csv")
x = data.iloc[:, 0].values

mu = np.mean(x)
sigma = np.std(x, ddof=1)

pd.DataFrame([[mu, sigma]], columns=["mu", "sigma"]).to_csv("testout_7.1.csv", index=False)
