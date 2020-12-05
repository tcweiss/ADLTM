# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import datetime
import  requests
import base64
import time
import plotly.graph_objects as go
import plotly.figure_factory as ff


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


### Turns a regular date into a unix time stamp
def timeInUnixTime(date):
	unix_time = time.mktime(date.timetuple())

	return unix_time


###Turns a unix time stamp into a radable time
def unixTimeInTime(unixtime):
	date = datetime.datetime.fromtimestamp(unixtime)
	normal_time = date.strftime('%y-%m-%d %H:%M:%S')

	return normal_time


### Get the temperatures for one timestamp
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

	return response


###Cleans the data and puts it into a easy to work with dataframe
def cleanData(json_data):
	datahits = json_data['hits']['hits']

	dataWithTS = pd.DataFrame()

	for i in range(0, len(datahits)):
		temporary = pd.DataFrame.from_dict(datahits[i]["_source"]["Values"])
		temporary["TS"] = unixTimeInTime(datahits[i]["_source"]["TS"])
		dataWithTS = dataWithTS.append(temporary)

	return dataWithTS


### Reorganizing Data ###
### Reorganizing Data ###
### Reorganizing Data ###

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


def get_sensor_column(sensorId):
	if 'A' in sensorId_to_name_dict[sensorId]:
		return 'A'
	elif 'B' in sensorId_to_name_dict[sensorId]:
		return 'B'
	return 'C'


def get_sensor_row(sensorId):
	return 43 - int(sensorId_to_name_dict[sensorId][1:])


current_data = cleanData(es_get_most_current_temps())


current_data['x'] = current_data.apply (lambda row: get_sensor_column(row['sensorId']), axis=1)
current_data['y'] = current_data.apply (lambda row: get_sensor_row(row['sensorId']), axis=1)
current_data = current_data.sort_values(by=['y','x'], ascending=[True, True])


heatmap_array = current_data.pivot('y', 'x', 'temp').values


### Heatmap with GO
# fig = go.Figure(go.Heatmap(
#     x=current_data['x'],
#     y=current_data['y'],
#     z=current_data['temp']))
# fig.show()

x_list = current_data['x'].to_list()
x_list = list(dict.fromkeys(x_list))
y_list = current_data['y'].to_list()
y_list = list(dict.fromkeys(y_list))


### Heatmap with FF
# colorscale_ = [[0, 'rgb(0,67,206)'], [0.5, 'rgb(105,41,196)'],[1, 'rgb(162,25,31)']]
colorscale_ = [[0, 'rgb(0,67,206)'], [1, 'rgb(162,25,31)']]
heatmap = ff.create_annotated_heatmap(
	x=x_list,
	y=y_list,
	z=heatmap_array,
	colorscale= colorscale_,
	showscale=True)


### FRONTEND ###
### FRONTEND ###
### FRONTEND ###

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
app.layout = html.Div(children=[
    html.H1(children='IBM Server Temperature Dashboard'),

    dcc.Graph(
        id='example-graph',
        figure=heatmap,
		style={'height': 850})
])

if __name__ == '__main__':
    app.run_server(debug=True)