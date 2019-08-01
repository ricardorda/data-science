# -*- coding: utf-8 -*-
'''
https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
'''

import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


DF = pd.read_csv('suiciderate.csv',
                 header=0,
                 names=['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides100k', 'country_year', 'HDI_for_year', 'gdp_for_year', 'gdp_per_capita', 'generation'])

print(DF.columns)
print('============================')
print(DF.info())
print('============================')
print(DF.describe())
print('============================')
print(DF.head())
print('============================')
print(DF['HDI_for_year'].value_counts(dropna=False))
print('============================')
#print(DF['HDI_for_year'].value_counts(dropna=True))
print('============================')
