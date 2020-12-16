###HP Filter Functions:

"""
Summary:

This is the further elaboration of the docstring. Within this section,
you can elaborate further on details as appropriate for the situation.
Notice that the summary and the elaboration is separated by a blank new
line.
"""

#Import Packages:-------------------------------------------------------------------------------------------

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import datetime
import requests
import base64
import math
from fbprophet import Prophet
import time
import numpy as np
from scipy.stats import multivariate_normal
# from plotly.subplots import make_subplots
import statsmodels.api as sm
from plotly import subplots

from query import *
from heatmap import *
from gaussian import *
from Prophet import *

#Fit HP Filter:---------------------------------------------------------------------------------------------

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
    anomalies = anomalies.merge(dataframe[['ds', 'y']], on='ds')

    return anomalies


# Create graph for HP filter model including anomalies
def hp_filter_get_graph(df_general, df_anomalies):
    fig = subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.data = []

    fig.add_trace(go.Scatter(x=df_general['ds'], y=df_general['y'], mode='lines', name='Temperature'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_anomalies['ds'], y=df_anomalies['y'], mode='markers', name='Filter anomaly'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df_general['ds'], y=df_general['err'], mode='lines', name='Filter error'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_anomalies['ds'], y=df_anomalies['err'], mode='markers', name='Filter anomaly'), row=2,
                  col=1)

    return fig


# Returns list of ts for anomalies from the hp filter model
def hp_filter_get_anomaly_ts(anomalies):
    anomalies_ts = pd.DataFrame(index=anomalies['ds'].copy())
    anomalies_ts = anomalies_ts.reset_index()

    return anomalies_ts
