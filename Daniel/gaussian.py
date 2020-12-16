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

from query import *




# take the difference in temperature between each time stamp for each sensor group
def gaussian_data_manipulation(dataframe):
	npData = np.array(dataframe)
	# get the changes in temperature
	new_npData = npData[1:] - npData[0:-1]

	return new_npData


# Hard coded mean parameters for Andri's model - NEVER LOOK AT THIS, IT'S SECRET
def gaussian_return_mean():
	mean = [3.40659341e-05, 1.31868132e-04, 1.59340659e-04, 5.34798535e-05
		, 2.01465201e-05, 2.71062271e-05, 3.80952381e-05, 1.44322344e-04
		, 1.69597070e-04, 5.56776557e-05, 2.56410256e-05, 1.53846154e-05
		, 4.83516484e-05, 1.28937729e-04, 1.23809524e-04, 4.54212454e-05
		, 3.07692308e-05, 9.89010989e-06]

	return mean


# Hard coded cov parameters for Andri's model - NEVER LOOK AT THIS, IT'S SECRET
def gaussian_return_cov():
	cov = [[0.02719169, 0.0247103, 0.00538927, 0.00166403, 0.00130125, 0.00084622
			   , 0.01855269, 0.02123193, 0.00614357, 0.00206514, 0.00040708, 0.00046282
			   , 0.01124767, 0.01481579, 0.00493736, 0.00239486, 0.00057597, 0.00023589]
		, [0.0247103, 0.08144764, 0.03388753, 0.00514032, -0.00203963, -0.00425561
			   , 0.01817861, 0.05514702, 0.02870606, 0.0020737, -0.00386256, -0.00428179
			   , 0.01269326, 0.0337239, 0.01827583, 0.00071801, -0.00311381, -0.00543763]
		, [0.00538927, 0.03388753, 0.06403214, 0.03381796, 0.01276803, 0.00390462
			   , 0.00585659, 0.02423579, 0.04469526, 0.02164947, 0.00714213, 0.00406105
			   , 0.00595996, 0.01430928, 0.0233916, 0.01183219, 0.00651803, 0.00308332]
		, [0.00166403, 0.00514032, 0.03381796, 0.05352663, 0.0263228, 0.01134512
			   , 0.00209921, 0.00210563, 0.02607938, 0.0313033, 0.01637863, 0.01093151
			   , 0.00304203, 0.00048591, 0.0089024, 0.01484916, 0.01256515, 0.00974714]
		, [0.00130125, -0.00203963, 0.01276803, 0.0263228, 0.0282219, 0.0166048
			   , 0.00165529, -0.00202212, 0.01234681, 0.02174676, 0.01901551, 0.01539407
			   , 0.00215644, -0.00066049, 0.00857175, 0.01602876, 0.01635975, 0.01442982]
		, [0.00084622, -0.00425561, 0.00390462, 0.01134512, 0.0166048, 0.03611752
			   , 0.00120931, -0.00303793, 0.00506121, 0.01284522, 0.01576842, 0.0147402
			   , 0.00142081, -0.00095742, 0.00667942, 0.01271139, 0.01472371, 0.01401537]
		, [0.01855269, 0.01817861, 0.00585659, 0.00209921, 0.00165529, 0.00120931
			   , 0.02619054, 0.02513861, 0.00704341, 0.00235419, 0.00071286, 0.00069218
			   , 0.01652936, 0.01922356, 0.00496206, 0.00241969, 0.00094595, 0.00053199]
		, [0.02123193, 0.05514702, 0.02423579, 0.00210563, -0.00202212, -0.00303793
			   , 0.02513861, 0.07851554, 0.03213649, 0.00220184, -0.00338038, -0.00350383
			   , 0.01749358, 0.05229728, 0.02401082, 0.00122151, -0.00233368, -0.00454172]
		, [0.00614357, 0.02870606, 0.04469526, 0.02607938, 0.01234681, 0.00506121
			   , 0.00704341, 0.03213649, 0.06664739, 0.0287093, 0.00929999, 0.00545462
			   , 0.00653072, 0.0229011, 0.03816196, 0.01914716, 0.00859508, 0.00428091]
		, [0.00206514, 0.0020737, 0.02164947, 0.0313033, 0.02174676, 0.01284522
			   , 0.00235419, 0.00220184, 0.0287093, 0.03874142, 0.02068935, 0.01340993
			   , 0.002712, 0.00303598, 0.02056199, 0.02483131, 0.01725428, 0.01229816]
		, [0.00040708, -0.00386256, 0.00714213, 0.01637863, 0.01901551, 0.01576842
			   , 0.00071286, -0.00338038, 0.00929999, 0.02068935, 0.02423331, 0.01710358
			   , 0.00110551, -0.00116496, 0.01135323, 0.01872331, 0.0198548, 0.01565222]
		, [0.00046282, -0.00428179, 0.00406105, 0.01093151, 0.01539407, 0.0147402
			   , 0.00069218, -0.00350383, 0.00545462, 0.01340993, 0.01710358, 0.01611497
			   , 0.00107116, -0.00128726, 0.00777826, 0.01368636, 0.01603593, 0.01467482]
		, [0.01124767, 0.01269326, 0.00595996, 0.00304203, 0.00215644, 0.00142081
			   , 0.01652936, 0.01749358, 0.00653072, 0.002712, 0.00110551, 0.00107116
			   , 0.01870008, 0.01747865, 0.00440035, 0.00252027, 0.00135431, 0.00087502]
		, [0.01481579, 0.0337239, 0.01430928, 0.00048591, -0.00066049, -0.00095742
			   , 0.01922356, 0.05229728, 0.0229011, 0.00303598, -0.00116496, -0.00128726
			   , 0.01747865, 0.06139911, 0.02832799, 0.00385439, 0.00015319, -0.00234917]
		, [0.00493736, 0.01827583, 0.0233916, 0.0089024, 0.00857175, 0.00667942
			   , 0.00496206, 0.02401082, 0.03816196, 0.02056199, 0.01135323, 0.00777826
			   , 0.00440035, 0.02832799, 0.06185261, 0.02283718, 0.0122981, 0.00652857]
		, [0.00239486, 0.00071801, 0.01183219, 0.01484916, 0.01602876, 0.01271139
			   , 0.00241969, 0.00122151, 0.01914716, 0.02483131, 0.01872331, 0.01368636
			   , 0.00252027, 0.00385439, 0.02283718, 0.02943048, 0.01872852, 0.01294223]
		, [0.00057597, -0.00311381, 0.00651803, 0.01256515, 0.01635975, 0.01472371
			   , 0.00094595, -0.00233368, 0.00859508, 0.01725428, 0.0198548, 0.01603593
			   , 0.00135431, 0.00015319, 0.0122981, 0.01872852, 0.02094934, 0.01515228]
		, [0.00023589, -0.00543763, 0.00308332, 0.00974714, 0.01442982, 0.01401537
			   , 0.00053199, -0.00454172, 0.00428091, 0.01229816, 0.01565222, 0.01467482
			   , 0.00087502, -0.00234917, 0.00652857, 0.01294223, 0.01515228, 0.01516658]]

	return cov


# Hard coded threshold parameters for Andri's model - NEVER LOOK AT THIS, IT'S SECRET
def gaussian_return_threshold():
	threshold = 3.7003561273008696e-34

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
	anomalies_ts=anomalies_ts.reset_index()

	return anomalies_ts
