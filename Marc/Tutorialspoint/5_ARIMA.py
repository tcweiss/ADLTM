###https://www.tutorialspoint.com/time_series/time_series_arima.htm
import pandas as pd
from sklearn import metrics
from math import sqrt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
import hurst
from statsmodels.tsa.arima_model import ARIMA


df_all = pd.read_csv('ALL_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True)
df_all = df_all.drop(labels='Unnamed: 0', axis=1)
df_all['TS'] = pd.to_datetime(df_all['TS'], yearfirst=True)
df_all = df_all.set_index('TS', drop=True, verify_integrity=True)

df_average = pd.DataFrame()
df_average['average'] = df_all.mean(axis=1)
print(df_average)

split = len(df_average) - int(0.2*len(df_average))
train, test = df_average['average'][0:split], df_average['average'][split:]
print(split)

result = adfuller(train)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
   print('\t%s: %.3f' % (key, value))


H, c, data = hurst.compute_Hc(train)
print("H = {:.4f}, c = {:.4f}".format(H,c))


model = ARIMA(train.values, order=(5, 0, 2))
model_fit = model.fit(disp=False)

predictions = model_fit.predict(len(test))
test_ = pd.DataFrame(test)
test_['predictions'] = predictions[0:7802]
print(test_)

plt.plot(df_average['average'])
plt.plot(test_['predictions'])
plt.show()

# df_result = pd.DataFrame()
# df_result['Average'] = df_average['average']
# df_result['Prediction'] = test_['predictions']
# print(df_result)


fig = go.Figure()
fig.add_trace(go.Scatter(x=df_average.index, y=df_average['average'], name='Average'))
fig.add_trace(go.Scatter(x=test_.index,y=test_['predictions'], name='Predictions ARIMA'))
fig.update_yaxes(range=[0,35])
fig.show()


error = sqrt(metrics.mean_squared_error(test.values,predictions[0:7802]))
print ('Test RMSE for ARIMA: ', error)