# -*- coding: utf-8 -*-
'''
https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
'''
from scipy.stats import spearmanr
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from sklearn import preprocessing
from scipy import stats
import numpy as np

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
                 names=['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides_100k'
                     , 'country_year', 'HDI_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation'])


df['gdp_for_year'] = df['gdp_for_year'].apply(ConverterParaFloat)

df = df[df['year'] != 2016]

df['generation'] = pd.Categorical(df['generation'])
df['sex'] = pd.Categorical(df['sex'])
df['age'] = pd.Categorical(df['age'])


print(df.info())
print('==============')

print(df.describe())

print('==============')

print(df.median())

print('==============')

print("kurtosis suicides_no = ", round( scipy.stats.kurtosis(df['suicides_no']),5))
print("kurtosis population = ", round( scipy.stats.kurtosis(df['population']),5))
print("kurtosis suicides_100k = ", round( scipy.stats.kurtosis(df['suicides_100k']),5))
print("kurtosis gdp_per_capita = ", round( scipy.stats.kurtosis(df['gdp_per_capita']),5))
print("kurtosis gdp_for_year = ", round( scipy.stats.kurtosis(df['gdp_for_year']),5))

print('==============')

print("kurtosis suicides_no = ", round( scipy.stats.skew(df['suicides_no']),5))
print("kurtosis population = ", round( scipy.stats.skew(df['population']),5))
print("kurtosis suicides_100k = ", round( scipy.stats.skew(df['suicides_100k']),5))
print("kurtosis gdp_per_capita = ", round( scipy.stats.skew(df['gdp_per_capita']),5))
print("kurtosis gdp_for_year = ", round( scipy.stats.skew(df['gdp_for_year']),5))

print('==============')

# A covariância entre os atributos
print(df.cov())

# A correlação entre os atributos /	O coeficiente de correlação de Pearson
df.corr().to_excel('pearson.xlsx')

# O coeficiente de correlação de Spearman
df.corr(method='spearman').to_excel('spearman.xlsx')

#Normalização mín-máx
colunas = ['suicides_no','population','suicides_100k','gdp_for_year','gdp_per_capita']
dfMinMax = df.copy()
dfMinMax[colunas] = dfMinMax[colunas].apply(minmax_scale)
dfMinMax.to_excel('suicide-rate-min-max.xlsx')
print(dfMinMax.head())

#Normalizacao pelo desvio padrao
colunas = ['suicides_no','population','suicides_100k','gdp_per_capita','gdp_for_year']
dfDesvio = df.copy()
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(dfDesvio[colunas])
dfDesvio[colunas] = scaled_df
dfDesvio.to_excel('suicide-rate-standard.xlsx')
print(dfDesvio.head())


# Trabalhando com os valores faltantes
print(df['HDI_for_year'].describe())
# mean 0.776601
# std 0.093367
df['HDI_for_year'].fillna((df['HDI_for_year'].mean()), inplace=True)
print(df['HDI_for_year'].describe())
# mean 0.776601
# std 0.051340
print("Erro padrão = ", stats.sem(df['HDI_for_year']))


# Mantendo o desvio padrão constante
print(df['HDI_for_year'].std())
df['HDI_for_year'].fillna(0.94639, inplace=True)
print("Desvio Padrão = ", df['HDI_for_year'].std())
print("Média = ", df['HDI_for_year'].mean())

# Encontrando os outliers
print(df['suicides_no'].describe())
z = np.abs(stats.zscore(df['suicides_no']))
outliers = df[z > 3]
print(outliers.count())

dfSemOutlier = df[z < 3]
print(dfSemOutlier['suicides_no'].describe())

# Gráficos
suicidiosPorAno = df.groupby(['year','age'])['suicides_no','suicides_100k'].sum().reset_index().sort_values(by='year')

fig, ax = plt.subplots(figsize=(16,7))

for key, grp in suicidiosPorAno.groupby(['age']):
    ax = grp.plot(ax=ax, x='year', y='suicides_no', label=key , grid=True)

plt.ylabel('suicides_no') 
plt.title('Suicidios por Ano') 
plt.show()


suicidiosPorSexo = df.groupby(['year','sex'])['suicides_no','suicides_100k'].sum().reset_index().sort_values(by='year')

fig, ax = plt.subplots(figsize=(16,7))

for key, grp in suicidiosPorSexo.groupby(['sex']):
    ax = grp.plot(ax=ax, x='year', y='suicides_no', label=key , grid=True)

plt.ylabel('suicides_no') 
plt.title('Suicidios por Ano e Sexo') 
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
plt.title('Suicidios Por Ano e Idade No Brasil') 
plt.show()


suicidiosPorSexoBrazil = brazil.groupby(['year','sex'])['suicides_no','suicides_100k'].sum().reset_index().sort_values(by='year')

fig, ax = plt.subplots(figsize=(16,7))

for key, grp in suicidiosPorSexoBrazil.groupby(['sex']):
    ax = grp.plot(ax=ax, x='year', y='suicides_no', label=key , grid=True)

plt.ylabel('suicides_no') 
plt.title('Suicidios Por Ano e Sexo no Brasil') 
plt.show()


dfParallel = df[['year','suicides_no','population','suicides_100k','gdp_for_year','gdp_per_capita']]
pd.plotting.parallel_coordinates(dfParallel, 'year')
plt.show()
