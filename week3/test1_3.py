#Covariance Missing data, Pairwise
#Input: test1.csv
#Output: testout_1.3.csv
import pandas as pd

data=pd.read_csv("test1.csv")
cov_matrix=data.cov()
cov_matrix.to_csv("testout_1.3.csv")
