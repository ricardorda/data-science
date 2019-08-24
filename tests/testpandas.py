import numpy as np
import pandas as pd

testDict = {
'name': [np.nan,'John','Peter','Charles'],
'heights': [1.9, np.nan, 1.8, 1.7],
'weight': [80,70,np.nan,90],
'age': [15, 20, 25, 40]}

df = pd.DataFrame(testDict)
print(df)

# Identifying null values
print(df.isnull())

# Identifying the columns with null values
print(df.loc[:, df.isnull().any(axis=0)])

# Identifying the rows with null values
print(df[df.isnull().any(axis=1)])

# Identifying the columns/rows with null values
print(df.loc[df.isnull().any(axis=1), df.isnull().any(axis=0)])
