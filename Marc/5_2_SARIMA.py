###https://www.tutorialspoint.com/time_series/time_series_arima.htm
import pandas as pd
from sklearn import metrics
from math import sqrt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import hurst
from statsmodels.tsa.arima_model import ARIMA


###Get Data
df_average = pd.read_csv('Average_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True, index_col=0)
df_average.index = pd.to_datetime(df_average.index, yearfirst=True)
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


model = SARIMAX(train.values, order=(1, 0, 1), seasonal_order=(1,0,1,2016))
model_fit = model.fit(disp=False)

predictions = model_fit.predict(len(test))
test_ = pd.DataFrame(test)
test_['predictions'] = predictions[0:7802]

plt.plot(df_average['average'])
plt.plot(test_['predictions'])
plt.show()

error = sqrt(metrics.mean_squared_error(test.values,predictions[0:7802]))
print ('Test RMSE for ARIMA: ', error)