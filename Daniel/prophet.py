import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import datetime
import requests
import base64
import math
import time
import numpy as np
from scipy.stats import multivariate_normal
from plotly.subplots import make_subplots
import statsmodels.api as sm
from plotly import subplots
from fbprophet import Prophet


# Fits Daniel's prophet model
def prophet_fit_model(dataframe, interval_width=0.98, changepoint_range=0.9):
	m = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False,
				seasonality_mode='multiplicative',
				interval_width=interval_width,
				changepoint_range=changepoint_range)
	m = m.fit(dataframe)
	forecast = m.predict(dataframe)
	forecast['fact'] = dataframe['y'].reset_index(drop=True)

	return forecast


# Returns anomalies based on Daniel's model version 2
def prophet_detect_anomalies(forecast):
	forecasted = forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
	# forecast['fact'] = df['y']

	forecasted['anomaly'] = 0
	forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
	forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = 1

	return forecasted


# Creates and returns a plotly figure according to Daniel's model version 2
def prophet_get_graph(pred):
	a = pred.loc[pred['anomaly'] == 1, ['ds', 'fact']]  # anomaly

	trace0 = go.Scatter(
		x=pred['ds'],
		y=pred['fact'],
		mode='lines',
		name='Temperature'
	)

	trace1 = go.Scatter(
		x=pred['ds'],
		y=pred['yhat_lower'],
		mode='lines',
		name='Lower Bound'
	)

	trace2 = go.Scatter(
		x=pred['ds'],
		y=pred['yhat_upper'],
		mode='lines',
		name='Higher Bound'
	)

	trace3 = go.Scatter(
		x=a['ds'],
		y=a['fact'],
		mode='markers',
		name='Anomaly'
	)

	data1 = [trace0, trace1, trace2, trace3]
	fig3 = go.Figure(data=data1)

	return fig3


# Returns list of ts for anomalies from the prophet model
def prophet_get_anomaly_ts(anomalies):
	anomalies_ts = anomalies[anomalies['anomaly']==1]
	anomalies_ts = pd.DataFrame(index=anomalies_ts['ds'].copy())
	anomalies_ts = anomalies_ts.reset_index()

	return anomalies_ts

