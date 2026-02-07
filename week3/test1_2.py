#Correlation Missing data, skip missing rows
#Input: test1.csv
#Output: testout_1.2.csv
import pandas as pd

data=pd.read_csv("test1.csv")
data=data.dropna()
corr_matrix=data.corr()
corr_matrix.to_csv("testout_1.2.csv")
