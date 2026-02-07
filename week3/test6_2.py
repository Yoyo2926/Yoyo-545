#calculate calculate log returns
#Input: test6.csv
#Output: testout_6.2.csv
import pandas as pd
import numpy as np

data = pd.read_csv("test6.csv")

dates = data.iloc[:, 0]         
prices = data.iloc[:, 1:]      

log_returns = np.log(prices / prices.shift(1))
log_returns = log_returns.iloc[1:]

out = pd.concat([dates.iloc[1:].reset_index(drop=True),
                 log_returns.reset_index(drop=True)], axis=1)

out.to_csv("testout_6.2.csv", index=False, float_format="%.15f")
