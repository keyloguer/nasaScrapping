#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
#%%
forestDS = pd.read_csv('API_AG.LND.FRST.K2_DS2_en_csv_v2_316456.csv')

#%%
populationDS = pd.read_csv('countries of the world.csv')

#%%
emissionDS = pd.read_csv('emission data.csv')
#%%
emissionDS = emissionDS[['Country','2005','2006','2007','2008','2009','2010']]
#%%
emissionDS
#%%
temperature = pd.read_csv('GlobalLandTemperaturesByCountry.csv')

#%%
temperature = temperature.loc[(temperature['dt'] > '2005-01-01') & (temperature['dt'] < '2011-01-01')]

#%%
temperature['period'] = temperature['mon'].map(lambda x: 2 if (x == 9 or x == 10 or x == 11 or  x == 12 or x == 1 or x == 2) else 1)

#%%
temperature['period']

#%%
temperature['mon'] = pd.to_numeric(temperature['mon'])
#%%
temperature['mon'].loc[temperature['mon'] == 1]
#%%
temperature['year'] = temperature['dt'].map(lambda x: x[0:4])
#%%
temperature['mon'] = temperature['dt'].map(lambda x: x[5:7]) 
#%%
temperatureAnnual = temperature.groupby(['Country','year', 'period'], as_index=False)['AverageTemperature'].mean()
#%%
temperature.groupby(['Country','year'], as_index=False)['AverageTemperature'].mean()
#%%
temperatureAnnual['AverageTemperature']
#%%
temperatureAnnual = temperatureAnnual.to_frame()
#%%
temperatureAnnual
#%%
tabelaTemp = pd.DataFrame(columns=colunas)

#%%
def transforma(row):
    global tabelaTemp
    if tabelaTemp.loc[(tabelaTemp['Country Name'] == row['Country']) & (tabelaTemp['period'] == row['period'])].empty:
        print(row['Country'])
        tabelaTemp = tabelaTemp.append(pd.Series([row['Country'], row['AverageTemperature'], row['period']], index=['Country Name','2005', 'period'] ), ignore_index=True)
    else:
        if row['year'] == '2006':
            tabelaTemp['2006'].loc[(tabelaTemp['Country Name'] == row['Country']) & (tabelaTemp['period'] == row['period'])] = row['AverageTemperature']
        if row['year'] == '2007':
            tabelaTemp['2007'].loc[(tabelaTemp['Country Name'] == row['Country']) & (tabelaTemp['period'] == row['period'])] = row['AverageTemperature']
        if row['year'] == '2008':
            tabelaTemp['2008'].loc[(tabelaTemp['Country Name'] == row['Country']) & (tabelaTemp['period'] == row['period'])] = row['AverageTemperature']
        if row['year'] == '2009':
            tabelaTemp['2009'].loc[(tabelaTemp['Country Name'] == row['Country']) & (tabelaTemp['period'] == row['period'])] = row['AverageTemperature']
        if row['year'] == '2010':
            tabelaTemp['2010'].loc[(tabelaTemp['Country Name'] == row['Country']) & (tabelaTemp['period'] == row['period'])] = row['AverageTemperature']
    
#%%
temperatureAnnual.apply(transforma, axis=1)
#%%
tabelaTemp = tabelaTemp.append(pd.Series(['teste', 15.000], index=['Country Name','2005'] ), ignore_index=True)
#%%
tabelaTemp = tabelaTemp.drop(columns=['Country'])
#%%
tabelaTemp['2006'].loc[tabelaTemp['Country Name'] == 'teste'] = 16.00
#%%
tabelaTemp['Country Name'] = temperatureAnnual['Country'][0]
#%%
tabelaTemp = pd.DataFrame(columns=colunas)
#%%
tabelaTemp.loc[tabelaTemp['Country Name'] == 'Brazil']

#%%
valores = forestDS.loc[:, forestDS.columns != 'Country Name']
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(valores)

#%%
forestDSS = pd.DataFrame(data=x_scaled,columns=['2005','2006','2007','2008','2009','2010'])
#%%
forestDSS.iloc[27]
#%%
tabelaTemp.loc[tabelaTemp['Country Name'] == 'Brazil']
#%%
forestDS.loc[forestDS['Country Name'] == 'Brazil']
#%%
emissionDS.loc[emissionDS['Country'] == 'Brazil']
#%%
forestDS = forestDS[['Country Name','2005','2006','2007','2008','2009','2010']]

#%%
emissionDS.to_csv('emissao.csv')
#%%
forestDS.to_csv('floresta.csv')
#%%
tabelaTemp.to_csv('temperatura.csv')
#%%
colunas = ['Country Name','2005','2006','2007','2008','2009','2010', 'period']

#%%
forestDS = forestDS.dropna().reset_index(drop=True)

#%%
row = forestDS.loc[forestDS['Country Name'] == 'Brazil']
row.plot(kind='bar')
plt.show()

#%%
row = emissionDS.loc[emissionDS['Country'] == 'Brazil']
row.plot(kind='bar')
plt.show()

#%%
row = tabelaTemp.loc[tabelaTemp['Country Name'] == 'Brazil']
row.plot(kind='bar')
plt.show()

#%%
forestDS.loc[forestDS['Country Name'] == 'Russia']

#%% 

forestDS['Country Name']

#%%
(25.226333 + 26.398500)/2

#%%
0.41*(10^9)/14.9*(10^6)

#%%
emissaoDS = emissionDS[['2008','2009','2010']].loc[emissionDS['Country'] == 'Brazil']
#%%
tempAnnual = pd.DataFrame(tabelaTemp[['2008','2009','2010']].loc[tabelaTemp['Country Name'] == 'Brazil'].mean())
#%%
tempAnnual = pd.DataFrame(data={'2008': '25.410333','2009': '25.600583','2010':'25.812417'}, index=[0])
#%%
tempAnnual
#%%
tempAnnual.index.to_list
#%%
florestaDS = forestDS[['2008','2009','2010']].loc[forestDS['Country Name'] == 'Brazil']
#%%
dfFinal = pd.concat([florestaDS,tempAnnual,emissaoDS])

#%%
dfFinal.to_json('informacoes.js')

#%%
