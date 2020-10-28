###https://www.tutorialspoint.com/time_series/time_series_moving_average.htm
import pandas as pd
import numpy
from sklearn import metrics
from math import sqrt
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

df_all = pd.read_csv('ALL_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True)
df_all = df_all.drop(labels='Unnamed: 0', axis=1)
df_all['TS'] = pd.to_datetime(df_all['TS'], yearfirst=True)
df_all = df_all.set_index('TS', drop=True, verify_integrity=True)

df_average = pd.DataFrame()
df_average['average'] = df_all.mean(axis=1)

split = len(df_average) - int(0.2*len(df_average))
train, test = df_average['average'][0:split], df_average['average'][split:]

plot_pacf(train, lags = 500)
plt.show()