import statsmodels.api as sm
from plotly import subplots

# Decompose time series into cycle and trend for HP filter Model
def hp_filter_decomposition(dataframe):
	cycle, trend = sm.tsa.filters.hpfilter(dataframe['y'], 10000)
	dec = pd.DataFrame(dataframe[['ds', 'y']])
	dec["cycle"] = cycle
	dec["trend"] = trend

	return dec


# Calculate error for HP filter Model
def hp_filter_calc_error(dataframe):
	dataframe['err'] = dataframe['y'] - dataframe['trend']

	return dataframe


# Identify anomalies for the HP filter Model
def hp_filter_get_anomalies(dataframe):
	dataframe['up'] = np.zeros(len(dataframe))
	dataframe['low'] = np.zeros(len(dataframe))
	dataframe.loc[dataframe['err'] > 1.2, 'up'] = 1
	dataframe.loc[dataframe['err'] < -1.40, 'low'] = 1

	return dataframe


# DF with anomalies for plot of HP filter Model
def hp_filter_get_anom_list(dataframe):
	anomalies = dataframe.loc[((dataframe['up'] == 1) | (dataframe['low'] == 1)), ['ds', 'err']].reset_index(drop=True)
	anomalies = anomalies.merge(dataframe[['ds','y']], on='ds')

	return anomalies


# Create graph for HP filter model including anomalies
def hp_filter_get_graph(df_general, df_anomalies):
	fig = subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)

	fig.data = []

	fig.add_trace(go.Scatter(x=df_general['ds'], y=df_general['y'], mode='lines', name='Temperature'), row=1, col=1)
	fig.add_trace(go.Scatter(x=df_anomalies['ds'], y=df_anomalies['y'], mode='markers', name='Filter anomaly'), row=1,col=1)
	fig.add_trace(go.Scatter(x=df_general['ds'], y=df_general['err'], mode='lines', name='Filter error'), row=2, col=1)
	fig.add_trace(go.Scatter(x=df_anomalies['ds'], y=df_anomalies['err'], mode='markers', name='Filter anomaly'), row=2,col=1)

	return fig


# Returns list of ts for anomalies from the hp filter model
def hp_filter_get_anomaly_ts(anomalies):
	anomalies_ts = pd.DataFrame(index=anomalies['ds'].copy())
	anomalies_ts = anomalies_ts.reset_index()

	return anomalies_ts

