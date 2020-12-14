# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import datetime
import requests
import base64
import time

### BACKEND ###
### BACKEND ###
### BACKEND ###


### Elastic Search Functions ###


### Header ###
elastic_url = 'http://159.122.185.120:9200'
index = "adltm.ch.2"
elastic_url = elastic_url + "/" + index + "/_search"

username = "read_only"
password = "read_only"

data = username + ':' + password

encodedBytes = base64.b64encode(data.encode("utf-8"))
new = encodedBytes.decode("utf-8", "ignore")

token = str(new)

header = {
	"Authorization": "Basic " + token,
	"Content-Type": "application/json"
}


# Turns a regular date into a unix time stamp
def time_in_unix_time(date):
	date = datetime.datetime.strptime(date, "%y-%m-%d %H:%M:%S")
	unix_time = time.mktime(date.timetuple())

	return unix_time


# Turns a unix time stamp into a readable time
def unix_time_in_time(unixtime):
	date = datetime.datetime.fromtimestamp(unixtime)
	normal_time = date.strftime('%y-%m-%d %H:%M:%S')

	return normal_time


# Get the temperatures for one timestamp
def es_get_most_current_temps():
	query = {
		"size": 1,
		"query": {
			'match_all': {}
		},
		"sort": [
			{
				"TS": {
					"order": "desc"
				}
			}
		]
	}

	r = requests.get(url=elastic_url, headers=header, json=query)
	response = r.json()

	cleaned_data = clean_data(response)
	cleaned_data_w_coor = add_coordinates(cleaned_data)

	return cleaned_data_w_coor


# Input a ts0 and returns the temperatures for the first available ts after (including) ts0
def es_get_specific_temps(ts):
	query = {
		"size": 1,
		"query": {
			"bool": {
				"filter": [
					{
						"range": {
							"TS": {
								"gte": time_in_unix_time(ts)
							}
						}
					}
				]
			}
		},
		"sort": [
			{
				"TS": {
					"order": "asc"
				}
			}
		]
	}

	r = requests.get(url=elastic_url, headers=header, json=query)
	response = r.json()

	cleaned_data = clean_data(response)
	cleaned_data_w_coor = add_coordinates(cleaned_data)

	return cleaned_data_w_coor


#Input a ts0 and return the temperatures for the first available ts after (including) ts0 and the previious one
def es_get_specifc_two_temps(ts):
	ts1 = time_in_unix_time(ts)
	ts0 = ts1 - 5 * 60
	query = {
		"size": 2,
		"query": {
			"bool": {
				"filter": [
					{
						"range": {
							"TS": {
								"gte": ts0
							}
						}
					}
				]
			}
		},
		"sort": [
			{
				"TS": {
					"order": "asc"
				}
			}
		]
	}

	r = requests.get(url=elastic_url, headers=header, json=query)
	response = r.json()

	cleaned_data = clean_data(response)
	transposed_data = cleaned_data.pivot(index='sensorId', columns='TS', values='temp').reset_index()
	transposed_data_w_coor = add_coordinates(transposed_data)
	transposed_data_w_coor['diff'] = transposed_data_w_coor[transposed_data_w_coor.columns[2]] - transposed_data_w_coor[transposed_data_w_coor.columns[1]]
	transposed_data_w_coor = transposed_data_w_coor.round({'diff': 1})

	return transposed_data_w_coor


# Cleans the data and puts it into a easy to work with dataframe
def clean_data(json_data):
	datahits = json_data['hits']['hits']

	data_with_ts = pd.DataFrame()

	for i in range(0, len(datahits)):
		temporary = pd.DataFrame.from_dict(datahits[i]["_source"]["Values"])
		temporary["TS"] = unix_time_in_time(datahits[i]["_source"]["TS"])
		data_with_ts = data_with_ts.append(temporary)

	return data_with_ts


# Transposes df and returns a df with sensorIds as columns and ts as rows
def transpose_df(dataframe):
	transposed_df = dataframe.pivot(index='TS', columns='sensorId', values='temp').reset_index()

	return transposed_df


name_to_sensorId_dict = {
	"A1": "28.2650D30B0000",
	"A2": "28.F910D40B0000",
	"A3": "28.F610D40B0000",
	"A4": "28.294DD40B0000",
	"A5": "28.7A5FD40B0000",
	"A6": "28.02CAD30B0000",
	"A7": "28.6242D40B0000",
	"A8": "28.188FD30B0000",
	"A9": "28.688ED30B0000",
	"A10": "28.EA10D40B0000",
	"A11": "28.9074D30B0000",
	"A12": "28.1556D40B0000",
	"A13": "28.AE49D40B0000",
	"A14": "28.9E6BD40B0000",
	"A15": "28.6DD9D30B0000",
	"A16": "28.4ECAD30B0000",
	"A17": "28.635FD40B0000",
	"A18": "28.1B50D30B0000",
	"A19": "28.02B6D40B0000",
	"A20": "28.655FD40B0000",
	"A21": "28.12D9D30B0000",
	"A22": "28.31CAD30B0000",
	"A23": "28.D249D40B0000",
	"A24": "28.49CAD30B0000",
	"A25": "28.F6B5D40B0000",
	"A26": "28.8DD5D30B0000",
	"A27": "28.796FD40B0000",
	"A28": "28.1A50D30B0000",
	"A29": "28.2056D40B0000",
	"A30": "28.A62BD40B0000",
	"A31": "28.868ED30B0000",
	"A32": "28.70D9D30B0000",
	"A33": "28.B1BAD40B0000",
	"A34": "28.C649D40B0000",
	"A35": "28.C4CAD30B0000",
	"A36": "28.79D9D30B0000",
	"A37": "28.F6DCD30B0000",
	"A38": "28.C850D30B0000",
	"A39": "28.60E6D30B0000",
	"A40": "28.3C5ED40B0000",
	"A41": "28.BB2BD40B0000",
	"A42": "28.6B42D40B0000",
	"B1": "28.024CD30B0000",
	"B2": "28.8F8ED30B0000",
	"B3": "28.51E6D30B0000",
	"B4": "28.305ED40B0000",
	"B5": "28.BFDCD30B0000",
	"B6": "28.0943D40B0000",
	"B7": "28.27B6D40B0000",
	"B8": "28.21F7D30B0000",
	"B9": "28.25CAD30B0000",
	"B10": "28.2ACAD30B0000",
	"B11": "28.F7B5D40B0000",
	"B12": "28.BE2BD40B0000",
	"B13": "28.3F73D40B0000",
	"B14": "28.54E6D30B0000",
	"B15": "28.5C8ED30B0000",
	"B16": "28.C72BD40B0000",
	"B17": "28.AF2BD40B0000",
	"B18": "28.1ECAD30B0000",
	"B19": "28.EAB5D40B0000",
	"B20": "28.1456D40B0000",
	"B21": "28.926BD40B0000",
	"B22": "28.0A74D30B0000",
	"B23": "28.950AD40B0000",
	"B24": "28.E110D40B0000",
	"B25": "28.A7DCD30B0000",
	"B26": "28.EC84D30B0000",
	"B27": "28.585FD40B0000",
	"B28": "28.096CD40B0000",
	"B29": "28.4E4BD30B0000",
	"B30": "28.074CD30B0000",
	"B31": "28.FAF5D30B0000",
	"B32": "28.2750D30B0000",
	"B33": "28.B7CAD30B0000",
	"B34": "28.6E6FD40B0000",
	"B35": "28.6D6FD40B0000",
	"B36": "28.F6C9D30B0000",
	"B37": "28.3F5ED40B0000",
	"B38": "28.BB74D30B0000",
	"B39": "28.EBF5D30B0000",
	"B40": "28.9774D30B0000",
	"B41": "28.185ED40B0000",
	"B42": "28.4A73D40B0000",
	"C1": "28.09F7D30B0000",
	"C2": "28.4B73D40B0000",
	"C3": "28.C949D40B0000",
	"C4": "28.0C6CD40B0000",
	"C5": "28.0D0AD40B0000",
	"C6": "28.DE10D40B0000",
	"C7": "28.9B6BD40B0000",
	"C8": "28.0556D40B0000",
	"C9": "28.19CAD30B0000",
	"C10": "28.ED10D40B0000",
	"C11": "28.0211D40B0000",
	"C12": "28.834BD30B0000",
	"C13": "28.A76BD40B0000",
	"C14": "28.CA2BD40B0000",
	"C15": "28.36CAD30B0000",
	"C16": "28.792CD30B0000",
	"C17": "28.D210D40B0000",
	"C18": "28.838ED30B0000",
	"C19": "28.BD49D40B0000",
	"C20": "28.9549D40B0000",
	"C21": "28.1B5ED40B0000",
	"C22": "28.6CE6D30B0000",
	"C23": "28.FAF6D30B0000",
	"C24": "28.FA54D40B0000",
	"C25": "28.C074D30B0000",
	"C26": "28.3E0AD40B0000",
	"C27": "28.FD6BD40B0000",
	"C28": "28.275ED40B0000",
	"C29": "28.842CD30B0000",
	"C30": "28.13DCD30B0000",
	"C31": "28.E5F6D30B0000",
	"C32": "28.D14DD40B0000",
	"C33": "28.5ACAD30B0000",
	"C34": "28.D549D40B0000",
	"C35": "28.5773D40B0000",
	"C36": "28.64D9D30B0000",
	"C37": "28.B474D30B0000",
	"C38": "28.61D9D30B0000",
	"C39": "28.5DE6D30B0000",
	"C40": "28.A449D40B0000",
	"C41": "28.CE23D40B0000",
	"C42": "28.A374D30B0000"
}

sensorId_to_name_dict = {}
for key, value in name_to_sensorId_dict.items():
	sensorId_to_name_dict[value] = key


# Takes an dataframe as an input and returns the ts
def get_timestamp(dataframe):
	ts = dataframe['TS'][0]

	return ts


# Returns the column name of the sensorId provided, named from left to right as A, B, C
def get_sensor_column(sensorId):
	loc = sensorId_to_name_dict[sensorId]
	column = loc[0]
	return column


# Returns the row name of the sensorId provided, counted from the bottom up, starting at 1
def get_sensor_row(sensorId):
	return int(sensorId_to_name_dict[sensorId][1:])


# Takes a sensorId as an input and returns the x and y coordinates
def get_coordinates(sensorId):
	loc = sensorId_to_name_dict[sensorId]
	col, row = loc[0], int(loc[1:])
	return col, row


# Adds two additional columns to the df with the x and y coordinates of the sensors
def add_coordinates(dataframe):
	dataframe['x'] = dataframe.apply(lambda row: get_sensor_column(row['sensorId']), axis=1)
	dataframe['y'] = dataframe.apply(lambda row: get_sensor_row(row['sensorId']), axis=1)
	dataframe = dataframe.sort_values(by=['y', 'x'], ascending=[True, True])

	return dataframe


# Takes dataframe and returns a numpy array that can be used for the create_plotly_ff_heatmap
def create_heatmap_array(dataframe):
	array = dataframe.pivot('y', 'x', 'temp').values

	return array


# Takes dataframe of difference in temp and returns a numpy array that can be used for create_plotly_ff_heatmap_diff
def create_heatmap_diff_array(dataframe):
	array = dataframe.pivot('y', 'x', 'diff').values

	return array


# Takes dataframe as an input and returns x and y lists for
# the axis labels that can be used for the plotly.figure_factory.create_annotated_heatmap
def create_axis_lists(dataframe):
	x_list = dataframe['x'].to_list()
	x_list = list(dict.fromkeys(x_list))
	y_list = dataframe['y'].to_list()
	y_list = list(dict.fromkeys(y_list))
	# y_list.reverse()

	return x_list, y_list


# Takes a dataframe as an input and returns a plotly.graphical_objects heatmap
def create_plotly_go_heatmap(dataframe):
	heatmap = go.Figure(go.Heatmap(
		x=dataframe['x'],
		y=dataframe['y'],
		z=dataframe['temp']))

	return heatmap


# Takes a dataframe as an input and returns a plotly.figure_factory annotated heatmap
def create_plotly_ff_heatmap_abs(dataframe):
	# colorscale_heatmap = [[0, 'rgb(0,67,206)'], [0.5, 'rgb(105,41,196)'],[1, 'rgb(162,25,31)']]
	colorscale_heatmap = [[0, 'rgb(69,137,255)'], [1, 'rgb(250,77,86)']]

	x_list, y_list = create_axis_lists(dataframe)
	heatmap_array = create_heatmap_array(dataframe)
	ts = get_timestamp(dataframe)

	heatmap = ff.create_annotated_heatmap(
		x=x_list,
		y=y_list,
		z=heatmap_array,
		colorscale=colorscale_heatmap,
		xgap = 10,
		ygap = 1,
		showscale=True
	)

	heatmap.update_layout(
		title=('Absolute temperature at ' + ts),
		xaxis=dict(title='Column', color='black', side='bottom'),
		yaxis=dict(title='Row', autorange="reversed"),
		plot_bgcolor='rgba(0,0,0,0)',
	)

	heatmap.data[0].colorbar = dict(title='Temperature', titleside='right')

	return heatmap


# Takes a dataframe with difference in temps as an input and returns a plotly.figure_factory annotated heatmap
def create_plotly_ff_heatmap_diff(dataframe):
	# colorscale_heatmap = [[0, 'rgb(0,67,206)'], [0.5, 'rgb(105,41,196)'],[1, 'rgb(162,25,31)']]
	max, min = dataframe['diff'].max(), dataframe['diff'].min()
	midpoint = abs(min)/(abs(max) + abs(min))
	colorscale_heatmap = [[0, 'rgb(69,137,255)'], [midpoint, 'rgb(255,255,255)'], [1, 'rgb(250,77,86)']]

	x_list, y_list = create_axis_lists(dataframe)
	heatmap_array = create_heatmap_diff_array(dataframe)
	ts0, ts1 = dataframe.columns[1], dataframe.columns[2]

	heatmap = ff.create_annotated_heatmap(
		x=x_list,
		y=y_list,
		z=heatmap_array,
		autocolorscale=False,
		colorscale=colorscale_heatmap,
		font_colors=['black'],
		xgap = 10,
		ygap = 1,
		showscale=True
	)

	heatmap.update_layout(
		title=('Changes in temperature between ' + ts0 + ' and ' + ts1),
		xaxis=dict(title='Column', side='bottom'),
		yaxis=dict(title='Row', autorange="reversed"),
		plot_bgcolor='rgba(0,0,0,0)',
	)

	heatmap.data[0].colorbar = dict(title='Temperature Change', titleside='right')

	return heatmap


### FRONTEND ###
### FRONTEND ###
### FRONTEND ###


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = [dbc.themes.MINTY]
# external_stylesheets = [dbc.themes.BOOTSTRAP]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
	html.H1(children='IBM - Rack Temperature'),
	dcc.Graph(
		id='live-update-heatmap-graph',
		style={'height': 700, 'width': '50%'}
	),
	dcc.Interval(
		id='interval-component',
		interval=10 * 1000,  # in milliseconds
		n_intervals=0
	),
	html.Div([
		dcc.Input(
			id = 'input_ts',
			type = 'text',
			placeholder= 'yy-mm-dd HH:MM:SS'
		),
		html.Button(
			id='submit-ts-button-state',
			n_clicks=0,
			children='Submit'),
	]),
	dcc.Graph(
		id='user-ts-heatmap-abs-graph',
		style={'height': 700, 'width': '50%'}
	),
	dcc.Graph(
		id='user-ts-heatmap-diff-graph',
		style={'height': 700, 'width': '50%'}
	)
])


# Update graph every minute for most current data
@app.callback(Output('live-update-heatmap-graph', 'figure'),
			  Input('interval-component', 'n_intervals'))
def update_heatmap_automatically(n):
	current_data = es_get_most_current_temps()
	heatmap = create_plotly_ff_heatmap_abs(current_data)

	return heatmap


# Create graph based on user input ts
@app.callback(Output('user-ts-heatmap-abs-graph', 'figure'),
			  Output('user-ts-heatmap-diff-graph', 'figure'),
			  Input('submit-ts-button-state', 'n_clicks'),
			  State('input_ts', 'value'),
			  prevent_initial_call=True)
def update_heatmap_user_ts(n_clicks, input_ts):
	if n_clicks is None:
		return PreventUpdate
	else:
		data_abs_heatmap = es_get_specific_temps(input_ts)
		heatmap_abs = create_plotly_ff_heatmap_abs(data_abs_heatmap)
		data_diff = es_get_specifc_two_temps(input_ts)
		heatmap_diff = create_plotly_ff_heatmap_diff(data_diff)


	return heatmap_abs, heatmap_diff

if __name__ == '__main__':
	app.run_server(debug=True)
