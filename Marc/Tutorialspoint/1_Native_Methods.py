###https://www.tutorialspoint.com/time_series/time_series_naive_methods.htm
import pandas as pd
import numpy
from sklearn import metrics
from math import sqrt

df_all = pd.read_csv('ALL_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True)
df_all = df_all.drop(labels='Unnamed: 0', axis=1)
df_all['TS'] = pd.to_datetime(df_all['TS'], yearfirst=True)
df_all = df_all.set_index('TS', drop=True, verify_integrity=True)

df_average = pd.DataFrame()
df_average['average'] = df_all.mean(axis=1)
df_average['average t-1'] = df_average['average'].shift(1)
df_naive = df_average[['average','average t-1']][1:]

true = df_naive['average']
prediction = df_naive['average t-1']
error = sqrt(metrics.mean_squared_error(true,prediction))
print('RMSE for Naive Method 1:', error)

df_average['average_rm'] = df_average['average'].rolling(3).mean().shift(1)
df_naive = df_average[['average','average_rm']].dropna()

true = df_naive['average']
prediction = df_naive['average_rm']
error = sqrt(metrics.mean_squared_error(true,prediction))
print('RMSE for Naive Method 2:', error)