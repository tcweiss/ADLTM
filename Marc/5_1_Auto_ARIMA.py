###Source: https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c

import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm


###Get Data
df_average = pd.read_csv('Average_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True, index_col=0)
df_average.index = pd.to_datetime(df_average.index, yearfirst=True)
print(df_average)


# df_average['TS'] = pd.to_datetime(df_average['TS'], yearfirst=True)
# df_average = df_average.set_index('TS', drop=True, verify_integrity=True)



###Statistical Decomposition
result = seasonal_decompose(df_average, model='multiplicative', period=2016)
df_result = pd.DataFrame()

df_result['seasonal'] = result.seasonal
df_result['residual'] = result.resid
df_result['trend'] = result.trend

print(df_result)

fig = px.line(df_result)
fig.show()


stepwise_model = pm.auto_arima(df_average, m=2016, seasonal=True, trace=True, stationary=True, error_action='ignore', suppress_warnings=True, stepwise=True)
print(stepwise_model.aic())