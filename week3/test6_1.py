#calculate arithmetic returns
#Input: test6.csv
#Output: testout_6.1.csv
import pandas as pd
import numpy as np

data = pd.read_csv("test6.csv")

dates = data.iloc[:, 0]      # 第一列（日期）
prices = data.iloc[:, 1:]    # 后面是价格（数值）

returns = prices.pct_change()
returns = returns.iloc[1:]

out = pd.concat(
    [dates.iloc[1:].reset_index(drop=True),
     returns.reset_index(drop=True)],
    axis=1
)

out.to_csv("testout_6.1.csv", index=False, float_format="%.15f")
