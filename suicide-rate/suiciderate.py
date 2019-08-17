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
import seaborn as sns

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


suicidiosPorAno = df.groupby(['year','age'])['suicides_no','suicides_100k'].sum().reset_index().sort_values(by='year')

fig, ax = plt.subplots(figsize=(16,7))

for key, grp in suicidiosPorAno.groupby(['age']):
    ax = grp.plot(ax=ax, x='year', y='suicides_no', label=key , grid=True)

plt.ylabel('suicides_no') 
plt.title('Suicidios por Ano') 
plt.show()



plt.figure(figsize=(16,7))
sns.heatmap(df.corr(), annot = True)
plt.show()


suicidioPorGdpPerCapita = df.groupby(['gdp_per_capita'])['suicides_no','suicides_100k'].sum().reset_index().sort_values(by='gdp_per_capita')

plt.figure(figsize=(16,7))
plt.scatter(data=suicidioPorGdpPerCapita, x='gdp_per_capita',y='suicides_no',alpha=0.5)
plt.xlabel('gdp_per_capita')
plt.ylabel('suicides_no')
plt.title('Suicidios Por GDP Per Capita')
plt.show()



suicidiosPorPais = df.groupby(['country'])['suicides_no','suicides_100k'].sum().reset_index()
suicidiosPorPaisTotal = suicidiosPorPais.sort_values(by='suicides_no', ascending=False)[:10]
suicidiosPorPais100k = suicidiosPorPais.sort_values(by='suicides_100k', ascending=False)[:10]

suicidiosPorPaisTotal.plot.bar(x='country',y='suicides_no',rot=45, figsize=(16,7))
plt.ylabel('suicides_no')

suicidiosPorPais100k.plot.bar(x='country',y='suicides_100k',rot=45, figsize=(16,7))
plt.ylabel('suicides/100k')


brazil = df[df['country'] == 'Brazil']
suicidiosBrazil = brazil.groupby(['year','age'])['suicides_no','suicides_100k'].sum().reset_index().sort_values(by='year')

fig, ax = plt.subplots(figsize=(16,7))

for key, grp in suicidiosBrazil.groupby(['age']):
    ax = grp.plot(ax=ax, x='year', y='suicides_no', label=key , grid=True)

plt.ylabel('suicides_no') 
plt.title('Suicidios Por Ano No Brazil') 
plt.show()


'''
pais15 = suicidiosPorPais[:5]
#.sort_values(by='suicides_100k', ascending=False)
print(pais15)
print('====')
for key, grp in pais15.groupby('sex'):
    print(key)
    print(grp)
  '''  

'''
fig, ax = plt.subplots(figsize=(16,7))

for key, grp in suicidiosPorPais.groupby(['sex']):
    ax = grp.plot(ax=ax, x='country', y='suicides_no', label=key , grid=True, kind='bar')

plt.ylabel('suicides_no') 
plt.title('Suicidios por Ano') 
plt.show()
'''

'''



suicidiosPorIdade = df.groupby(['country', 'age'])['suicides_no','suicides_100k'].sum().reset_index()
print(suicidiosPorIdade.head())
'''


#Paises Total 
#Paises 100k
#Brasil por idade



'''
df.corr(method='spearman').to_excel('spearman.xlsx')
df.corr().to_excel('pearson.xlsx')

print(df.info())
print(df.head())


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

#suicidiosPorAno.plot(x='year', y='suicides_no', title='Quantidade de Suicidio/100k Por Ano', grid=True)

#plt.plot()
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

