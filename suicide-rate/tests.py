import pandas as pd
import  numpy as np

testDict = {
'name': [np.nan,'John','Peter','Charles'],
'heights': [1.9, np.nan, 1.8, 1.7],
'weight': [80,70,np.nan,90]}

# Identifying null values
df = pd.DataFrame(testDict)
print(df)
print(df.isnull())
print(df.isnull().any(axis=0))
print(df.isnull().any(axis=1))


