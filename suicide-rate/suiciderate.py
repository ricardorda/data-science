# -*- coding: utf-8 -*-
'''
https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
'''
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


def ConverterParaFloat(valor):
    try:
        valor = float( valor.replace(",","") )
    except:
        valor = np.NaN

    return valor



df = pd.read_csv('suiciderate.csv',
                 header=0,
                 names=['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides_100k', 'country_year', 'HDI_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation'])


df['gdp_for_year'] = df['gdp_for_year'].apply(ConverterParaFloat)

df.drop(['country_year','HDI_for_year'], axis=1, inplace=True)

df = df[df['year'] != 2016]

df['generation'] = pd.Categorical(df['generation'] )
df['sex'] = pd.Categorical(df['sex'] )
df['age'] = pd.Categorical(df['age'] )


'''
df.corr(method='spearman').to_excel('spearman.xlsx')
df.corr().to_excel('pearson.xlsx')

print(df.info())
print(df.head())

'''
suicidiosPorAno = df.groupby('year')['suicides_no','suicides_100k'].sum()

fig = plt.figure(figsize=(15,6))
#suicidiosPorAno.plot(title='Quantidade de Suicidio/100k Por Ano', grid=True)

#plt.clf()
'''
suicidioPorGdpPerCapita.reset_index().sort_values(by='gdp_per_capita')
print(suicidioPorGdpPerCapita )
plt.scatter(x=suicidioPorGdpPerCapita['gdp_per_capita'], y=suicidioPorGdpPerCapita['suicides_no'])
'''

suicidioPorGdpPerCapita = df.groupby(['gdp_per_capita'])['suicides_no','suicides_100k'].sum()
suicidioPorGdpPerCapita = suicidioPorGdpPerCapita.reset_index().sort_values(by='gdp_per_capita')
print(suicidioPorGdpPerCapita.head(5))
plt.scatter(x=suicidioPorGdpPerCapita['gdp_per_capita'], y=suicidioPorGdpPerCapita['suicides_no'])
plt.xlabel('gdp_per_capita')
plt.ylabel('suicides_no')
plt.title('Suicidios por gdp_per_capita')
plt.show()

'''geracao = df.groupby(['year'])['generation','suicides_no']

geracao.plot.bar()
print(geracao.head(10))
'''
'''

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

