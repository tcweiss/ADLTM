###Gaussian Model Functions:

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

#Fit Gaussian Model:-----------------------------------------------------------------------------------------

# take the difference in temperature between each time stamp for each sensor group
def gaussian_data_manipulation(dataframe):
    npData = np.array(dataframe)
    # get the changes in temperature
    new_npData = npData[1:] - npData[0:-1]
    new_npData = (new_npData - np.mean(new_npData, axis=0)) / np.std(new_npData, axis=0)

    return new_npData


# Hard coded mean parameters for Andri's model - NEVER LOOK AT THIS, IT'S SECRET
def gaussian_return_mean():
    mean = [1.26980624e-17, -8.48664474e-18,  8.32712134e-18, -7.24236224e-18,
        1.23471110e-17,  1.13899706e-17, -1.81856673e-17, -9.89045064e-18,
        1.65904333e-18,  8.03997923e-18, -4.08379897e-18, -1.75475737e-18,
        1.14856846e-18, -2.30351786e-17,  2.90013536e-17,  1.17728267e-17,
        1.20918735e-17, -5.48760487e-18]

    return mean


# Hard coded cov parameters for Andri's model - NEVER LOOK AT THIS, IT'S SECRET
def gaussian_return_cov():
    cov = [[1.00001796, 0.5571847, 0.15087369, 0.03481675, 0.0391908,
         0.0269671, 0.6855736, 0.48312932, 0.18253151, 0.07000051,
         0.02211134, 0.02406162, 0.44811593, 0.37382189, 0.14753482,
         0.09341452, 0.03068674, 0.0202361],
       [0.5571847 ,  1.00001796,  0.47972956,  0.08482414, -0.03509766,
        -0.07656625,  0.39961696,  0.68417961,  0.40955409,  0.05603912,
        -0.07698473, -0.11192427,  0.28889017,  0.45989147,  0.24656116,
         0.02304456, -0.07433647, -0.13866954],
       [0.15087369,  0.47972956,  1.00001796,  0.57776025,  0.28535567,
         0.07320221,  0.15496899,  0.34995314,  0.64657916,  0.41855368,
         0.15914887,  0.09535653,  0.16647924,  0.22992577,  0.29773663,
         0.22493255,  0.13322671,  0.06890058],
       [0.03481675,  0.08482414,  0.57776025,  1.00001796,  0.6770828 ,
         0.27337579,  0.05209916,  0.04376717,  0.42381383,  0.6876802 ,
         0.45295504,  0.3618921 ,  0.10588774,  0.01635582,  0.13640938,
         0.36139864,  0.3642839 ,  0.32872845],
       [0.0391908 , -0.03509766,  0.28535567,  0.6770828 ,  1.00001796,
         0.55500753,  0.05991235, -0.03357928,  0.25958275,  0.65620757,
         0.73323725,  0.72327339,  0.10690153, -0.01087908,  0.18358715,
         0.55169304,  0.67865085,  0.70128162],
       [0.0269671 , -0.07656625,  0.07320221,  0.27337579,  0.55500753,
         1.00001796,  0.0422648 , -0.05302041,  0.09009137,  0.36323727,
         0.56979103,  0.64957743,  0.06674545, -0.01664394,  0.13611794,
         0.41321989,  0.57398105,  0.63998164],
       [0.6855736 ,  0.39961696,  0.15496899,  0.05209916,  0.05991235,
         0.0422648 ,  1.00001796,  0.57308323,  0.20256623,  0.08107672,
         0.03661562,  0.04075687,  0.72328539,  0.48810192,  0.14349716,
         0.09569694,  0.04679102,  0.03989571],
       [0.48312932,  0.68417961,  0.34995314,  0.04376717, -0.03357928,
        -0.05302041,  0.57308323,  1.00001796,  0.48693611,  0.06727837,
        -0.06165379, -0.08799564,  0.43251428,  0.7410337 ,  0.35447721,
         0.04696833, -0.04982985, -0.10995754],
       [0.18253151,  0.40955409,  0.64657916,  0.42381383,  0.25958275,
         0.09009137,  0.20256623,  0.48693611,  1.00001796,  0.55137798,
         0.20494453,  0.12713163,  0.17876925,  0.38140191,  0.54117818,
         0.39213078,  0.1868817 ,  0.10309341],
       [0.07000051,  0.05603912,  0.41855368,  0.6876802 ,  0.65620757,
         0.36323727,  0.08107672,  0.06727837,  0.55137798,  1.00001796,
         0.67039761,  0.52354597,  0.11229263,  0.085138  ,  0.39846412,
         0.71999669,  0.59376078,  0.49403863],
       [0.02211134, -0.07698473,  0.15914887,  0.45295504,  0.73323725,
         0.56979103,  0.03661562, -0.06165379,  0.20494453,  0.67039761,
         1.00001796,  0.86471362,  0.06687654, -0.01627526,  0.27209333,
         0.69425488,  0.88032115,  0.81372421],
       [0.02406162, -0.11192427,  0.09535653,  0.3618921 ,  0.72327339,
         0.64957743,  0.04075687, -0.08799564,  0.12713163,  0.52354597,
         0.86471362,  1.00001796,  0.07529139, -0.03462015,  0.21547961,
         0.61842717,  0.87315206,  0.93809651],
       [0.44811593,  0.28889017,  0.16647924,  0.10588774,  0.10690153,
         0.06674545,  0.72328539,  0.43251428,  0.17876925,  0.11229263,
         0.06687654,  0.07529139,  1.00001796,  0.50076848,  0.1321828 ,
         0.11566958,  0.07959061,  0.07059234],
       [0.37382189,  0.45989147,  0.22992577,  0.01635582, -0.01087908,
        -0.01664394,  0.48810192,  0.7410337 ,  0.38140191,  0.085138  ,
        -0.01627526, -0.03462015,  0.50076848,  1.00001796,  0.49296885,
         0.11455681,  0.01199651, -0.05825793],
       [0.14753482,  0.24656116,  0.29773663,  0.13640938,  0.18358715,
         0.13611794,  0.14349716,  0.35447721,  0.54117818,  0.39846412,
         0.27209333,  0.21547961,  0.1321828 ,  0.49296885,  1.00001796,
         0.51968753,  0.31248081,  0.19135599],
       [0.09341452,  0.02304456,  0.22493255,  0.36139864,  0.55169304,
         0.41321989,  0.09569694,  0.04696833,  0.39213078,  0.71999669,
         0.69425488,  0.61842717,  0.11566958,  0.11455681,  0.51968753,
         1.00001796,  0.74847432,  0.60625895],
       [0.03068674, -0.07433647,  0.13322671,  0.3642839 ,  0.67865085,
         0.57398105,  0.04679102, -0.04982985,  0.1868817 ,  0.59376078,
         0.88032115,  0.87315206,  0.07959061,  0.01199651,  0.31248081,
         0.74847432,  1.00001796,  0.85310562],
       [0.0202361 , -0.13866954,  0.06890058,  0.32872845,  0.70128162,
         0.63998164,  0.03989571, -0.10995754,  0.10309341,  0.49403863,
         0.81372421,  0.93809651,  0.07059234, -0.05825793,  0.19135599,
         0.60625895,  0.85310562,  1.00001796]]

    return cov


# Hard coded threshold parameters for Andri's model - NEVER LOOK AT THIS, IT'S SECRET
def gaussian_return_threshold():
    threshold = 1.8130403839344504e-34

    return threshold


# Hard coded all Gaussian parameters and returns them: mean, cov, thr - STILL SECRET BUT YOU CAN LOOK AT IT
def gaussian_return_params():
    mean = gaussian_return_mean()
    cov = gaussian_return_cov()
    thr = gaussian_return_threshold()

    return mean, cov, thr


# Creating multi-variate gaussian model
def gaussian_fit_model(mean, cov):
    # multivariate gaussian model
    multivar = multivariate_normal(mean=mean, cov=cov)

    return multivar


# Creating the gaussian probability densities
def gaussian_calculate_prob_densities(npMatrix, multivar_Gaussian):
    probabilityDensities = multivar_Gaussian.pdf(npMatrix)

    return probabilityDensities


# Adding three columns to the df, probDensities, Anomaly, and Threshold
def gaussian_add_columns(dataframe, thr, densities):
    pd_dataframe = pd.DataFrame(dataframe)
    new_dataframe = pd_dataframe.iloc[1:, ]
    new_dataframe["probDensities"] = densities
    new_dataframe["Anomaly"] = new_dataframe["probDensities"] < thr
    new_dataframe["Threshold"] = thr

    return new_dataframe


# Creates new dataframe that only includes the anomalies which can be used for the graph
def gaussian_get_anomalies(dataframe):
    new_dataframe = dataframe[dataframe["Anomaly"] == True]
    new_dataframe["rowMean"] = new_dataframe.iloc[:, 0:18].mean(axis=1)

    return new_dataframe


# Creates and returns the gaussian subplot
def gaussian_get_graph(df_general, df_anomalies):
    trace0 = go.Scatter(
        x=df_general.index,
        y=df_general['rA1'],
        mode='lines',
        name='rA1'
    )

    trace1 = go.Scatter(
        x=df_general.index,
        y=df_general['rA2'],
        mode='lines',
        name='rA2'
    )

    trace2 = go.Scatter(
        x=df_general.index,
        y=df_general['rA3'],
        mode='lines',
        name='rA3'
    )

    trace3 = go.Scatter(
        x=df_general.index,
        y=df_general['rA4'],
        mode='lines',
        name='rA4'
    )

    trace4 = go.Scatter(
        x=df_general.index,
        y=df_general['rA5'],
        mode='lines',
        name='rA5',
    )

    trace5 = go.Scatter(
        x=df_general.index,
        y=df_general['rA6'],
        mode='lines',
        name='rA6'
    )

    trace6 = go.Scatter(
        x=df_general.index,
        y=df_general['rB1'],
        mode='lines',
        name='rB1'
    )

    trace7 = go.Scatter(
        x=df_general.index,
        y=df_general['rB2'],
        mode='lines',
        name='rB2'
    )

    trace8 = go.Scatter(
        x=df_general.index,
        y=df_general['rB3'],
        mode='lines',
        name='rB3'
    )

    trace9 = go.Scatter(
        x=df_general.index,
        y=df_general['rB4'],
        mode='lines',
        name='rB4',
    )

    trace10 = go.Scatter(
        x=df_general.index,
        y=df_general['rB5'],
        mode='lines',
        name='rB5'
    )

    trace11 = go.Scatter(
        x=df_general.index,
        y=df_general['rB6'],
        mode='lines',
        name='rB6'
    )

    trace12 = go.Scatter(
        x=df_general.index,
        y=df_general['rC1'],
        mode='lines',
        name='rC1'
    )

    trace13 = go.Scatter(
        x=df_general.index,
        y=df_general['rC2'],
        mode='lines',
        name='rC2'
    )

    trace14 = go.Scatter(
        x=df_general.index,
        y=df_general['rC3'],
        mode='lines',
        name='rC3',
    )

    trace15 = go.Scatter(
        x=df_general.index,
        y=df_general['rC4'],
        mode='lines',
        name='rC4'
    )

    trace16 = go.Scatter(
        x=df_general.index,
        y=df_general['rC5'],
        mode='lines',
        name='rC5'
    )

    trace17 = go.Scatter(
        x=df_general.index,
        y=df_general['rC6'],
        mode='lines',
        name='rC6'
    )

    trace18 = go.Scatter(
        x=df_anomalies.index,
        y=df_anomalies['rowMean'],
        mode='markers',
        name='Anomaly',
        opacity=0.7,
        marker=dict(
            color='Red',
            size=8,
            line=dict(
                color='Black',
                width=2
            )
        )
    )

    data1_Andri = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11,
                   trace12, trace13, trace14, trace15, trace16, trace17, trace18]

    trace19 = go.Scatter(
        x=df_general.index,
        y=df_general['probDensities'],
        mode='lines',
        name='Estimated Density',
        line=dict(
            color='Black',
            width=2
        )
    )

    trace20 = go.Scatter(
        x=df_general.index,
        y=df_general["Threshold"],
        mode='lines',
        name='Threshold',
        line=dict(
            color='Red',
            width=4
        )
    )

    data2_Andri = [trace19, trace20]

    fig = subplots.make_subplots(
        rows=2, cols=1,
        subplot_titles=("Temperature of Sensor Groups", "Estimated Probability Densities"),
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.05)

    # fig = go.Figure(data = data1_Andri)

    for i in range(19):
        fig.add_trace(data1_Andri[i], row=1, col=1)

    fig.add_trace(data2_Andri[0], row=2, col=1)
    fig.add_trace(data2_Andri[1], row=2, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text="Date & Time", row=2, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Temperature", row=1, col=1)
    fig.update_yaxes(title_text="Density", type="log", row=2, col=1)

    fig.update_layout(height=800, title_text="Temperature Anomalies in Server Environment")

    return fig


# Returns list of ts for anomalies from the gaussian model
def gaussian_get_anomaly_ts(anomalies):
    anomalies_ts = pd.DataFrame(index=anomalies.index.copy())
    anomalies_ts = anomalies_ts.reset_index()

    return anomalies_ts
