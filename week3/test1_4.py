#Correlation Missing data, pairwise
#Input: test1.csv
#Output: testout_1.4.csv
import pandas as pd

data=pd.read_csv("test1.csv")
corr_matrix=data.corr()
corr_matrix.to_csv("testout_1.4.csv")
