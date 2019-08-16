# -*- coding: utf-8 -*-
'''
https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
'''
from scipy.stats import spearmanr
from scipy.stats import pearsonr

import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


def RemoveVirgula(valor):
    try:
        valor = float( valor.replace(",","") )
    except:
        valor = np.NaN

    return valor



df = pd.read_csv('suiciderate.csv',
                 header=0,
                 index_col = 'year',
                 names=['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides_100k', 'country_year', 'HDI_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation'])


df['gdp_for_year'] = df['gdp_for_year'].apply(RemoveVirgula)

df.drop(['country_year','HDI_for_year'], axis=1, inplace=True)

df.to_csv('teste.csv')

mycor = df.corr(method='spearman')

mycor.to_excel('teste2.xlsx')
 
'''
corr1, p_value1 = pearsonr(df['suicides_no'], df['gdp_for_year'])
corr2, p_value2 = pearsonr(df['suicides_no'], df['gdp_per_capita'])
corr3, p_value3 = pearsonr(df['suicides_100k'], df['gdp_for_year'])
corr4, p_value4 = pearsonr(df['suicides_100k'], df['gdp_per_capita'])

print("corr1 = ", corr1)
print("corr2 = ", corr2)
print("corr3 = ", corr3)
print("corr4 = ", corr4)
print("p_value1 = ", p_value1)
print("p_value2 = ", p_value2)
print("p_value3 = ", p_value3)
print("p_value4 = ", p_value4)


corr1, p_value1 = spearmanr(df['suicides_no'], df['gdp_for_year'])
corr2, p_value2 = spearmanr(df['suicides_no'], df['gdp_per_capita'])
corr3, p_value3 = spearmanr(df['suicides_100k'], df['gdp_for_year'])
corr4, p_value4 = spearmanr(df['suicides_100k'], df['gdp_per_capita'])

dts = df.cov()
dts.to_csv
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

print(df.corr())

print(df.info())
print(df.head(10))

df.plot( y='suicides_no', subplots=True, kind='hist')
plt.show()

#print(df[df['year'] > 2014])

#print(nulos)

#print("gdp_for_year = ", df['gdp_for_year'].value_counts(dropna=False))

#print(df['gdp_per_capita'])

#gdp = pd.to_numeric(df['gdp_for_year'])
#print(gdp)
#print(.unique())

print("suicides_no = ", round( scipy.stats.kurtosis(df['suicides_no']),5) )
print("population = ", round( scipy.stats.kurtosis(df['population']),5))
print("suicides_100k = ", round( scipy.stats.kurtosis(df['suicides_100k']),5))
print("gdp_per_capita = ", round( scipy.stats.kurtosis(df['gdp_per_capita']),5))
print("gdp_for_year = ", round( scipy.stats.kurtosis(df['gdp_for_year']),5))


print("suicides_no = ", round( scipy.stats.skew(df['suicides_no']),5) )
print("population = ", round( scipy.stats.skew(df['population']),5))
print("suicides_100k = ", round( scipy.stats.skew(df['suicides_100k']),5))
print("gdp_per_capita = ", round( scipy.stats.skew(df['gdp_per_capita']),5))
print("gdp_for_year = ", round( scipy.stats.skew(df['gdp_for_year']),5))


df['sex'] = df['sex'].astype('category')
df['age'] = df['age'].astype('category')
df['generation'] = df['generation'].astype('category')


print(df.columns)
print('============================')
print(df.info())
print('============================')
print(df.head(15))
print('============================')
'''

#print('============================')
#print(DF.describe())
#print('============================')
#print(DF['HDI_for_year'].value_counts(dropna=False))
#print('============================')
#print(DF['country_year'].value_counts())
#print('============================')
#print(DF.info())
#print('============================')
#print(DF.head())
#print('============================')
#print(DF['HDI_for_year'].value_counts(dropna=True))
#print('============================')

