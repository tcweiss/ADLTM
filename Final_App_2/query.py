###Query and Data Manipulation Functions

"""
Summary:

This code accesses the elastic search database where the temperature data is stored.
It creates a central DataFrame to store the data. This data frame is updated _______.
Then functions are defined to properly format the temperature data.
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

#Elastic Search Functions:----------------------------------------------------------------------------------

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
                                "lte": time_in_unix_time(ts)
                            }
                        }
                    }
                ]
            }
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


def es_get_diff_ts(ts):
    ts1 = time_in_unix_time(ts)
    query = {
        "size": 2,
        "query": {
            "bool": {
                "filter": [
                    {
                        "range": {
                            "TS": {
                                "lte": ts1,
                            }
                        }
                    }
                ]
            }
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
    print(cleaned_data)
    transposed_data = cleaned_data.pivot(index='sensorId', columns='TS', values='temp').reset_index()
    transposed_data_w_coor = add_coordinates(transposed_data)
    transposed_data_w_coor['diff'] = transposed_data_w_coor[transposed_data_w_coor.columns[2]] - transposed_data_w_coor[
        transposed_data_w_coor.columns[1]]
    transposed_data_w_coor = transposed_data_w_coor.round({'diff': 1})

    return transposed_data_w_coor


# Input a ts0 and return the temperatures for the first available ts before (including) ts0 and the previous one
def es_get_current_diff():
    query = {
        "size": 2,
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
    transposed_data = cleaned_data.pivot(index='sensorId', columns='TS', values='temp').reset_index()
    transposed_data_w_coor = add_coordinates(transposed_data)
    transposed_data_w_coor['diff'] = transposed_data_w_coor[transposed_data_w_coor.columns[2]] - transposed_data_w_coor[
        transposed_data_w_coor.columns[1]]
    transposed_data_w_coor = transposed_data_w_coor.round({'diff': 1})

    return transposed_data_w_coor


# Elastic search to query for data for Daniel's prophet model
def es_get_data(ts, size, aggregated=False):
    query = {
        "size": size,
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
    no_dups = cleaned_data.drop_duplicates(subset=['sensorId', 'TS'], keep='last')
    transposed_data = no_dups.pivot(index='TS', columns='sensorId', values='temp').reset_index()
    renamed_data = transposed_data.rename(columns=sensorId_to_name_dict)

    if aggregated == False:
        renamed_data['TS'] = pd.to_datetime(renamed_data['TS'], yearfirst=True)
        renamed_data = renamed_data.set_index('TS', drop=True, verify_integrity=True)
        return renamed_data
    else:
        region_df = convert_to_regions(renamed_data)
        return region_df


# Query for one day
def es_get_specific_day(day):
    end = day + datetime.timedelta(hours=23, minutes=59, seconds=59)
    query = {
        "size": 288,
        "query": {
            "bool": {
                "filter": [
                    {
                        "range": {
                            "TS": {
                                "gte": time_in_unix_time(str(day)[2:]),
                                "lte": time_in_unix_time(str(end)[2:])
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

    return response


# Takes a number of days and whether or not the data should be aggregated into regions and return the df
def es_get_last_days(days, aggregated=False):
    days = int(days)

    cleaned_data = pd.DataFrame()

    end_date = datetime.datetime.today().replace(microsecond=0)
    start_date = end_date - datetime.timedelta(days=days)

    while True:
        if start_date < end_date:
            data = es_get_specific_day(start_date)
            new_cleaned_data = clean_data(data)
            cleaned_data = cleaned_data.append(new_cleaned_data, ignore_index=True)

            start_date = start_date + datetime.timedelta(days=1)
        else:
            break

    no_dups = cleaned_data.drop_duplicates(subset=['sensorId', 'TS'], keep='last')
    transposed_data = no_dups.pivot(index='TS', columns='sensorId', values='temp').reset_index()
    renamed_data = transposed_data.rename(columns=sensorId_to_name_dict)

    if aggregated == False:
        renamed_data['TS'] = pd.to_datetime(renamed_data['TS'], yearfirst=True)
        renamed_data = renamed_data.set_index('TS', drop=True, verify_integrity=True)
        return renamed_data
    else:
        region_df = convert_to_regions(renamed_data)
        return region_df


# Turn number of days into the corresponding number of ts
def days_into_size(days):
    return days * 24 * 60 / 5


# Takes a df and return a subset of the df that can be used for Daniel's  model
def get_sub_df(df, sub):
    new_df = df[sub].copy()
    new_df = new_df.reset_index(level=0, inplace=False)
    new_df = new_df.rename(columns={"TS": "ds", sub: "y"})

    return new_df


# Cleans the data and puts it into a easy to work with dataframe
def clean_data(json_data):
    datahits = json_data['hits']['hits']

    data_with_ts = pd.DataFrame()

    for i in range(0, len(datahits)):
        temporary = pd.DataFrame.from_dict(datahits[i]["_source"]["Values"])
        temporary["TS"] = unix_time_in_time(datahits[i]["_source"]["TS"])
        data_with_ts = data_with_ts.append(temporary)

    return data_with_ts


# Aggregates df into 3x6 regions
def convert_to_regions(dataframe):
    dict_col_name = {}

    for col in dataframe.columns:
        try:
            sensor_col, sensor_row = col[:1], int(col[1:])
        except:
            continue
        temp_name = 'r' + sensor_col + str(math.ceil(sensor_row / 7))
        if temp_name not in dict_col_name:
            dict_col_name[temp_name] = []
        dict_col_name[temp_name].append(col)

    region_df = pd.DataFrame()

    for region in dict_col_name:
        region_df[region] = dataframe[dict_col_name[region]].mean(axis=1)

    region_df['TS'] = dataframe['TS']
    region_df['TS'] = pd.to_datetime(region_df['TS'], yearfirst=True)
    region_df = region_df.set_index('TS', drop=True, verify_integrity=True)

    region_df = region_df.sort_index(axis=1)

    return region_df


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


