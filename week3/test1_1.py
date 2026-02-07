#Covariance Missing data, skip missing rows
#Input: test1.csv
#Output: testout_1.1.csv
import pandas as pd

data=pd.read_csv("test1.csv")
data=data.dropna()
cov_matrix=data.cov()
cov_matrix.to_csv("testout_1.1.csv")
