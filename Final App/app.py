###Main Dash App

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
from HPfilter import *
from Prophet import *

# FRONTEND:------------------------------------------------------------------------------------------------

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
                dbc.NavLink("Dashboard", href="/page-1", id="page-1-link", external_link=True, active="exact"),
                dbc.NavLink("Heatmap", href="/page-2", id="page-2-link", external_link=True, active="exact"),
                dbc.NavLink("Anomalies", href="/page-3", id="page-3-link", external_link=True, active="exact"),
                dbc.NavLink("About", href="/page-4", id="page-4-link", external_link=True, active="exact"),
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
    ], style={'height': '100vh'}),
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
    ]),
])

tab_about = html.Div([
    dbc.Jumbotron(
        [
            dbc.Container(
                [
                    html.H1("About this Tool", className="display-3"),
                    html.P(
                        "This tool was created by students and members of the Data Science & Technology Club at the University of St. Gallen. The purpose of it is to help IBM detecting anomalies in temperature data which is measured in front of their server racks. Those anomalies could be caused by inefficient conditions in the server lab which is why they should be further investigated. The tool provides various features and uses three different models. Each model has its own benefits and drawbacks. However, it is important to have this three different perspectives on anomalies in the first prototype of this tool because there is no clear definition for an anomaly in this problem context. Time will tell if one model is better suited than the others. Furthermore, it needs to be considered that some of the model parameters are trained on data generated in 2020 and optimized particularly for this IBM lab. Changing conditions in the system's environment as well as a rollout to other physical systems would require to make adjustments such as retraining the model. In such a case it is recommended to contact the right person of the development team in order to find a solution. Contact details can be found below.",
                        className="lead",
                    ),
                    html.Br(className="my-2"),
                    html.Br(className="my-2"),
                    html.Br(className="my-2"),
                    html.Br(className="my-2"),
                    html.Br(className="my-2"),
                    html.Br(className="my-2"),
                    html.Br(className="my-2"),
                    html.Br(className="my-2"),
                    html.Br(className="my-2"),
                    html.Br(className="my-2"),
                    html.Hr(className="my-2"),
                    html.P(
                        [
                        "Learn more about the HSG Data Science & Technology Club: ",
                        html.A('Website', href='https://dataclub.ch', target='_blank'),]
                    ),
                ],
                fluid=True,
            )
        ],
        fluid=True,
        style={'background': 'linear-gradient(rgba(255,255,255,0.7), rgba(255,255,255,0.7)), url("/assets/analytics.jpg") no-repeat center center fixed',
               "-webkit-background-size": "cover",
               "-moz - background - size": "cover",
               "-o-background-size": "cover",
               "background-size": "cover",
               "height": "700px",
               "color": "rgb(0,0,0)",
               "text-align": "center"},
    ),
    html.Br(className="my-2"),
    dbc.Row([
        html.H3("About the Development Team"),
    ]),
    html.Br(className="my-2"),
    dbc.Row([
        dbc.CardDeck(
            [
                dbc.Card(
                    [
                        dbc.CardImg(src="/assets/marc.jpeg", top=True),
                        dbc.CardBody(
                            [
                                html.H5("Marc-Robin Gruener", className="card-title"),
                                html.P([
                                    "Marc was heavily involved on many different tasks. He mainly focused on integrating the different modules everyone was working on into a comprehensive software solution. Furthermore, he created the different heatmaps, wrote functions for the database queries and therefore ensured that the tool runs efficiently and with real time data, and he examined and tested various modeling approaches himself. Please contact Marc for very general issues: ",
                                    html.A('email', href='mailto:marc-robin.gruener@student.unisg.ch?subject=IBM Temperature Management Dashboard', target='_blank'), ],
                                    className="card-text",
                                    style={"text-align": "justify"},
                                ),
                                dbc.Button(
                                    "LinkedIn", href='https://www.linkedin.com/in/marc-robin-gruener/', target="_blank", color="primary", className="mt-auto"
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Card(
                    [
                        dbc.CardImg(src="/assets/AndriData2.jpg", top=True),
                        dbc.CardBody(
                            [
                                html.H5("Andri Turra", className="card-title"),
                                html.P([
                                    'Andri was responsible for developing the Multivariate Gaussian Model. This model considers correlations of temperature changes across regions and therefore allows the user to identify holistic temperature behaviour which is unlikely to occur under normal circumstances. He also created the corresponding visualization and designed the "About" page. Please do not hesitate to contact Andri for questions regarding the Multivariate Gaussian Model: ',
                                    html.A('email', href='mailto:andri.turra@gmail.com?subject=IBM Temperature Management Dashboard', target='_blank'),],
                                    className="card-text",
                                    style={"text-align": "justify"},
                                ),
                                dbc.Button(
                                    "LinkedIn", href='https://www.linkedin.com/in/andri-turra-85366b127/', target="_blank", color="primary", className="mt-auto"
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Card(
                    [
                        dbc.CardImg(src="/assets/Daniel.png", top=True),
                        dbc.CardBody(
                            [
                                html.H5("Daniel Leal", className="card-title"),
                                html.P([
                                    "Daniel designed and implemented the Prophet Model. This approach models the time series data of individual sensors/regions and flags an anomaly if the observed data breaks out of predefined bounds... Furthermore, he created the corresponding visualization and participated actively in the design of the overall solution. Questions regarding the Prophet Model can be directly addressed to Daniel: ",
                                    html.A('email', href='mailto:daniel.leal@student.unisg.ch?subject=IBM Temperature Management Dashboard', target='_blank'),],
                                    className="card-text",
                                    style={"text-align": "justify"},
                                ),
                                dbc.Button(
                                    "LinkedIn", href='https://www.linkedin.com/in/daniel-a-leal/', target="_blank", color="primary", className="mt-auto"
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Card(
                    [
                        dbc.CardImg(src="/assets/thomas.PNG", top=True),
                        dbc.CardBody(
                            [
                                html.H5("Thomas Weiss", className="card-title"),
                                html.P([
                                    "Thomas guaranteed a smooth cooperation by setting up a Github-Repo. Additionally, he contributed a lot to the development of the application with his expertise in programming and the creation of dashboards... Please contact Thomas if you would like to build this application with R Shiny: ",
                                    html.A('email',
                                           href='mailto:tcweiss@protonmail.com?subject=IBM Temperature Management Dashboard',
                                           target='_blank'), ],
                                    className="card-text",
                                    style={"text-align": "justify", "font-size":"60%"},
                                ),
                                dbc.Button(
                                    "LinkedIn", href='https://www.linkedin.com/in/thomas-w-574b1015a/', target="_blank",
                                    color="primary", className="mt-auto"
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Card(
                    [
                        dbc.CardImg(src="/assets/Adam.PNG", top=True),
                        dbc.CardBody(
                            [
                                html.H5("Adam Novak", className="card-title"),
                                html.P([
                                    "Adam came up with the Hodrick Prescott Filter. His approach first smooths out the time series data and then calculates the error difference between the filtered data and the actual temperatures. This allows us the user to find abrupt changes in temperature readings, which is usually a sign of anomalous sensor behavior. In case you have questions regarding the HP Filter, please do not hesitate to reach out to Adam: ",
                                    html.A('email',
                                           href='mailto:adam.novak@student.unisg.ch?subject=IBM Temperature Management Dashboard',
                                           target='_blank'), ],
                                    className="card-text",
                                    style={"text-align": "justify"},
                                ),
                                dbc.Button(
                                    "LinkedIn", href='https://www.linkedin.com/in/adam-novak-2a03b216b/',
                                    target="_blank", color="primary", className="mt-auto"
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Card(
                    [
                        dbc.CardImg(src="/assets/lorenz.png", top=True),
                        dbc.CardBody(
                            [
                                html.H5("Lorenz Schmidlin", className="card-title"),
                                html.P([
                                    "Lorenz organized the whole project. He was essential for the communication between IBM and the development team. Additionally, he organized important info and learning sessions, and he also helped out if some issues arised during the development process. Please contact Lorenz if you are interested in further projects with the Data Science & Technology Club: ",
                                    html.A('email',
                                           href='mailto:andri.turra@gmail.com?subject=IBM Temperature Management Dashboard',
                                           target='_blank'), ],
                                    className="card-text",
                                    style={"text-align": "justify"},
                                ),
                                dbc.Button(
                                    "LinkedIn", href='https://www.linkedin.com/in/lorenz-schmidlin-0b2516114/',
                                    target="_blank", color="primary", className="mt-auto"
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

    ]),
    html.Br(className="my-2"),
    dbc.Row([
        dbc.CardDeck(
            [
                dbc.Card(
                    [
                        dbc.CardImg(src="/assets/thomas.PNG", top=True),
                        dbc.CardBody(
                            [
                                html.H5("Thomas Weiss", className="card-title"),
                                html.P([
                                    "Thomas guaranteed a smooth cooperation by setting up a Github-Repo. Additionally, he contributed a lot to the development of the application with his expertise in programming and the creation of dashboards... Please contact Thomas if you would like to build this application with R Shiny: ",
                                    html.A('email', href='mailto:tcweiss@protonmail.com?subject=IBM Temperature Management Dashboard', target='_blank'), ],
                                    className="card-text",
                                    style={"text-align": "justify"},
                                ),
                                dbc.Button(
                                    "LinkedIn", href='https://www.linkedin.com/in/thomas-w-574b1015a/', target="_blank", color="primary", className="mt-auto"
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Card(
                    [
                        dbc.CardImg(src="/assets/Adam.PNG", top=True),
                        dbc.CardBody(
                            [
                                html.H5("Adam Novak", className="card-title"),
                                html.P([
                                    "Adam came up with the Hodrick Prescott Filter. His approach first smooths out the time series data and then calculates the error difference between the filtered data and the actual temperatures. This allows us the user to find abrupt changes in temperature readings, which is usually a sign of anomalous sensor behavior. In case you have questions regarding the HP Filter, please do not hesitate to reach out to Adam: ",
                                    html.A('email', href='mailto:adam.novak@student.unisg.ch?subject=IBM Temperature Management Dashboard', target='_blank'), ],
                                    className="card-text",
                                    style={"text-align": "justify"},
                                ),
                                dbc.Button(
                                    "LinkedIn", href='https://www.linkedin.com/in/adam-novak-2a03b216b/', target="_blank", color="primary", className="mt-auto"
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Card(
                    [
                        dbc.CardImg(src="/assets/lorenz.png", top=True),
                        dbc.CardBody(
                            [
                                html.H5("Lorenz Schmidlin", className="card-title"),
                                html.P([
                                    "Lorenz organized the whole project. He was essential for the communication between IBM and the development team. Additionally, he organized important info and learning sessions, and he also helped out if some issues arised during the development process. Please contact Lorenz if you are interested in further projects with the Data Science & Technology Club: ",
                                    html.A('email', href='mailto:andri.turra@gmail.com?subject=IBM Temperature Management Dashboard', target='_blank'), ],
                                    className="card-text",
                                    style={"text-align": "justify"},
                                ),
                                dbc.Button(
                                    "LinkedIn", href='https://www.linkedin.com/in/lorenz-schmidlin-0b2516114/', target="_blank", color="primary", className="mt-auto"
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

    ]),
])

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url", refresh=False), sidebar, content])


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on

@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 5)],
    [Input("url", "pathname")], )
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False, False
    return [pathname == f"/page-{i}" for i in range(1, 5)]

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
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
        raise PreventUpdate
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
            anomalies_dropdown = [{'label': str(i), 'value': str(i)} for i in anomalies_ts['TS']]

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
            anomalies_dropdown = [{'label': str(i), 'value': str(i)} for i in anomalies_ts['ds']]

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
    app.run_server(debug=True, host='127.0.0.1')