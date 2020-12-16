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

### NEW ONES ###
import numpy as np
from scipy.stats import multivariate_normal
# from plotly.subplots import make_subplots
import statsmodels.api as sm
from plotly import subplots

from query import *
from heatmap import *
from prophet import *
from gaussian import *
from hpfilter import *




# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = [dbc.themes.MINTY]
external_stylesheets = [dbc.themes.BOOTSTRAP]
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ['https://unpkg.com/carbon-components/css/carbon-components.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
# app = dash.Dash(__name__, suppress_callback_exceptions=True)


##### NEW FRONTEND START #####
##### NEW FRONTEND START #####
##### NEW FRONTEND START #####


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
	"position": "fixed",
	"top": 0,
	"left": 0,
	"bottom": 0,
	"width": "16rem",
	"padding": "2rem 1rem",
	"background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
	"margin-left": "18rem",
	"margin-right": "2rem",
	"padding": "2rem 1rem",
}

sidebar = html.Div(
	[
		html.H2("IBM RACK TEMPERATURE", className="display-5"),
		html.Hr(),
		html.P(
			"A simple sidebar layout with navigation links", className="lead"
		),
		dbc.Nav(
			[
				dbc.NavLink("Dashboard", href="/page-1", id="page-1-link"),
				dbc.NavLink("Heatmap", href="/page-2", id="page-2-link"),
				dbc.NavLink("Anomalies", href="/page-3", id="page-3-link"),
				dbc.NavLink("About", href="/page-4", id="page-4-link"),
			],
			vertical=True,
			pills=True,
		),
	],
	style=SIDEBAR_STYLE,
)

tab_dashboard = html.Div([
	dbc.Col([
		html.Div([
			dcc.Graph(
				id='live-update-heatmap-graph',
				style={'height': '90vh'}
			),
			dcc.Interval(
				id='interval-component',
				interval=10 * 1000,  # in milliseconds
				n_intervals=0
			),
		])
	]),
	dbc.Col([
		dcc.Graph(
			id='',
			style={'height': '90vh'}
		),
	])
], style={'columnCount': 2})

tab_heatmap = html.Div([
	dbc.Row([
		html.Div([
			dcc.Input(
				id='input_ts',
				type='text',
				placeholder='yy-mm-dd HH:MM:SS'
			),
			html.Button(
				id='submit-ts-button-state',
				n_clicks=0,
				children='Submit'),
		]),
	]),
	dbc.Row([
		dbc.Col([
			dcc.Graph(
				id='user-ts-heatmap-abs-graph',
				style={'height': '80vh'}
			),
		]),
		dbc.Col([
			dcc.Graph(
				id='user-ts-heatmap-diff-graph',
				style={'height': '80vh'}
			),
		]),
	], style={'columnCount': 2}),
])

tab_anomalies = html.Div([
	html.Div(
		id='anomaly-dropdown-holder',
		# style={'display': 'none'}
	),
	html.Div([
		dbc.Row([
		dbc.Col([
			html.H5(
				'Pick a model:'
			),
			dcc.Dropdown(
				id='anomaly-model-dropdown',
				options=[
					{'label': 'Multivariate Gaussian Model', 'value': 'gaussian'},
					{'label': 'HP Filter Model', 'value': 'hp_filter'},
					{'label': 'Prophet Model', 'value': 'prophet'}
				]
			)
		]),
		dbc.Col([
			html.H5(
				'Pick a time frame:'
			),
			dcc.Dropdown(
				id='anomaly-days-dropdown',
				options=[
					{'label': 'Last day', 'value': '1'},
					{'label': 'Last 7 days', 'value': '7'},
					{'label': 'Last 30 days', 'value': '30'}
				],
			),
		]),
		dbc.Col([
			html.H5(
				'Select a Sensor or Region:'
			),
			dcc.Input(
				id='anomaly-sensor-input',
				type='text',
				placeholder='Sensor (e.g. "A1") or Region (e.g. "rA1")',
				style={'width': '100%'}
			),
		], align='center'),
		dbc.Col([
			html.H5('Get results:'),
			html.Button(
				id='anomaly-submit-button',
				n_clicks=0,
				children='Submit'
			),
		], align='center')
	], style={'columnCount': 3}),
		dbc.Row([
		dcc.Graph(
			id='anomaly-output-graph',
			style={'height': 700, 'width': '100%'}
		)
	]),
	],style={'height': '100vh'}),
	html.Div([
		dbc.Row([
			html.Div([
				html.H5(
					'Select a Timestamp:'
				),
				dcc.Dropdown(
					id='anomaly-heatmap-dropdown',
					placeholder='Select a timestamp',
					style={'width': '120%'}
				)
			]),
			dbc.Row([
				dbc.Col([
					dcc.Graph(
						id='anomaly-heatmap-abs-graph',
						style={'height': '80vh'}
					),
				]),
				dbc.Col([
					dcc.Graph(
						id='anomaly-heatmap-diff-graph',
						style={'height': '80vh'}
					),
				]),
			], style={'columnCount': 2})
		])
	]),
])

tab_about = html.Div([
	dbc.Row([
		html.H4("This tool was created by:"),
	]),
	dbc.Row([
		dbc.Col([
			html.P(
				'Marc-Robin Grüner (participant)'
			),
		]),
		dbc.Col([
			html.A(
				"LinkedIn", href='https://www.linkedin.com/in/marc-robin-gruener/', target="_blank"
			),
		]),

	]),
	dbc.Row([
		dbc.Col([
			html.P(
				'Daniel Leal (participant)'
			),
		]),
		dbc.Col([
			html.A(
				"LinkedIn", href='https://www.linkedin.com/in/daniel-a-leal/', target="_blank"
			),
		])
	]),
	dbc.Row([
		dbc.Col([
			html.P(
				'Adam Novak (participant)'
			),
		]),
		dbc.Col([
			html.A(
				"LinkedIn", href='http://linkedin.com/in/adam-novak-2a03b216b', target="_blank"
			),
		])
	]),
	dbc.Row([
		dbc.Col([
			html.P(
				'Lorenz Schmidlin (project group lead)'
			),
		]),
		dbc.Col([
			html.A(
				"LinkedIn", href='https://www.linkedin.com/in/lorenz-schmidlin-0b2516114/', target="_blank"
			),
		])
	]),
	dbc.Row([
		dbc.Col([
			html.P(
				'Andri Turra (participant)'
			),
		]),
		dbc.Col([
			html.A(
				"LinkedIn", href='https://www.linkedin.com/in/andri-turra-85366b127/', target="_blank"
			),
		])
	]),
	dbc.Row([
		dbc.Col([
			html.P(
				'Thomas Weiss (participant)'
			),
		]),
		dbc.Col([
			html.A(
				"LinkedIn", href='https://www.linkedin.com/in/thomas-w-574b1015a/', target="_blank"
			),
		])
	]),
	dbc.Row([
		html.H4("Learn more about the HSG DataScience and Technology Club:")
	]),
	dbc.Row([
		html.A('Website', href='https://dataclub.ch', target='_blank')
	])
])

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
	[Output(f"page-{i}-link", "active") for i in range(1, 5)],
	[Input("url", "pathname")], )
def toggle_active_links(pathname):
	if pathname == "/":
		# Treat page 1 as the homepage / index
		return True, False, False
	return [pathname == f"/page-{i}" for i in range(1, 5)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
	if pathname in ["/", "/page-1"]:
		return tab_dashboard
	elif pathname == "/page-2":
		return tab_heatmap
	elif pathname == "/page-3":
		return tab_anomalies
	elif pathname == "/page-4":
		return tab_about
	# If the user tries to reach a different page, return a 404 message
	return dbc.Jumbotron(
		[
			html.H1("404: Not found", className="text-danger"),
			html.Hr(),
			html.P(f"The pathname {pathname} was not recognised..."),
		]
	)


# Update graph every minute for most current data
@app.callback(Output('live-update-heatmap-graph', 'figure'),
			  Input('interval-component', 'n_intervals'))
def update_heatmap_automatically(n):
	current_data = es_get_most_current_temps()
	heatmap = create_plotly_ff_heatmap_abs(current_data)

	return heatmap


# Create heatmap graph based on user input ts
@app.callback(Output('user-ts-heatmap-abs-graph', 'figure'),
			  Output('user-ts-heatmap-diff-graph', 'figure'),
			  Input('submit-ts-button-state', 'n_clicks'),
			  State('input_ts', 'value'),
			  prevent_initial_call=True)
def update_heatmap_user_ts(n_clicks, ts_value):
	if n_clicks is None:
		return PreventUpdate
	else:
		data_abs_heatmap = es_get_specific_temps(ts_value)
		heatmap_abs = create_plotly_ff_heatmap_abs(data_abs_heatmap)
		data_diff = es_get_diff_ts(ts_value)
		heatmap_diff = create_plotly_ff_heatmap_diff(data_diff)

	return heatmap_abs, heatmap_diff


# Create prophet anomaly graph based on user input
@app.callback(Output('anomaly-output-graph', 'figure'),
			  Output('anomaly-heatmap-dropdown', 'options'),
			  Input('anomaly-submit-button', 'n_clicks'),
			  State('anomaly-model-dropdown', 'value'),
			  State('anomaly-sensor-input', 'value'),
			  State('anomaly-days-dropdown', 'value'),
			  prevent_initial_call=True)
def update_anomaly_model_input(n_clicks, model, sensor, days):
	if n_clicks is None:
		return PreventUpdate
	else:
		if model == 'gaussian':
			aggregated = True
		elif 'r' in sensor:
			aggregated = True
		else:
			aggregated = False

		if model == 'gaussian':
			data = es_get_last_days(days, aggregated=True)
			data_manipulated = gaussian_data_manipulation(data)
			mean, cov, thr = gaussian_return_params()
			multivar = gaussian_fit_model(mean=mean, cov=cov)
			densities = gaussian_calculate_prob_densities(data_manipulated, multivar)
			new_dataframe = gaussian_add_columns(data, thr, densities)
			anomalies_df = gaussian_get_anomalies(new_dataframe)
			fig = gaussian_get_graph(new_dataframe, anomalies_df)

			anomalies_ts = gaussian_get_anomaly_ts(anomalies_df)
			anomalies_dropdown =  [{'label': str(i), 'value': str(i)} for i in anomalies_ts['TS']]

			return fig, anomalies_dropdown

		elif model == 'hp_filter':
			data = es_get_last_days(days, aggregated=aggregated)
			ind_region_data = get_sub_df(data, sensor)
			decomposed_data = hp_filter_decomposition(ind_region_data)
			error_data = hp_filter_calc_error(decomposed_data)
			anomalies = hp_filter_get_anomalies(error_data)
			anomalies_list = hp_filter_get_anom_list(error_data)
			fig = hp_filter_get_graph(anomalies, anomalies_list)

			anomalies_ts = hp_filter_get_anomaly_ts(anomalies_list)
			anomalies_dropdown = [{'label': str(i), 'value': str(i)} for i in anomalies_ts['ds']]

			return fig, anomalies_dropdown

		elif model == 'prophet':
			data_prophet = es_get_last_days(days, aggregated=aggregated)
			sub_df = get_sub_df(data_prophet, sensor)
			forecast = prophet_fit_model(sub_df)
			anomalies = prophet_detect_anomalies(forecast)
			fig = prophet_get_graph(anomalies)

			anomalies_ts = prophet_get_anomaly_ts(anomalies)
			print(anomalies_ts)
			anomalies_dropdown =  [{'label': str(i), 'value': str(i)} for i in anomalies_ts['ds']]

			return fig, anomalies_dropdown


# Update Heatmaps on the anomalies tab in response to a ts being picked from the dropdown menu
@app.callback(Output('anomaly-heatmap-abs-graph', 'figure'),
			  Output('anomaly-heatmap-diff-graph', 'figure'),
			  Input('anomaly-heatmap-dropdown', 'value'),
			  prevent_initial_call=True)
def update_anomaly_heatmap_dropdown(ts_value):
	if ts_value is None:
		return PreventUpdate
	else:
		date = ts_value[2:]
		data_abs_heatmap = es_get_specific_temps(date)
		heatmap_abs = create_plotly_ff_heatmap_abs(data_abs_heatmap)
		data_diff = es_get_diff_ts(date)
		heatmap_diff = create_plotly_ff_heatmap_diff(data_diff)

		return heatmap_abs, heatmap_diff


##### NEW FRONTEND END #####
##### NEW FRONTEND END #####
##### NEW FRONTEND END #####


if __name__ == '__main__':
	app.run_server(debug=True)
