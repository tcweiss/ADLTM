import requests
import base64
import datetime
import time
import pandas as pd
import json
from base64 import b64encode


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


###Returns n=size many documents, starting at start time
def queryTimePeriod(start, end, size):
	# Optionally you can add the "size": n parameter like in the function "queryRandom(n)" to limit the number of returned documents.
	query = {
		"size": size,
		"query": {
			"bool": {
				"filter": [
					{
						"range": {
							"TS": {
								"gte": timeInUnixTime(start),
								"lte": timeInUnixTime(end)
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

	# print(response)

	return response


###Turns a regular date into a unix time stamp
def timeInUnixTime(date):
	unix_time = time.mktime(date.timetuple())

	return unix_time


###Turns a unix time stamp into a radable time
def unixTimeintime(unixtime):
	date = datetime.datetime.fromtimestamp(unixtime)
	normal_time = date.strftime('%y-%m-%d %H:%M:%S')

	return normal_time


###Cleans the data and puts it into a easy to work with dataframe
def cleanData(data):
	datahits = data['hits']['hits']

	datawithTS = pd.DataFrame()

	for i in range(0, len(datahits)):
		temporary = pd.DataFrame.from_dict(datahits[i]["_source"]["Values"])
		temporary["TS"] = unixTimeintime(datahits[i]["_source"]["TS"])
		datawithTS = datawithTS.append(temporary)

	return datawithTS


###Turns a dataframe in with sensorId, temp, and TS into a dataframe where columns=sensorIds, rows=TS, and cells=temp
def transpose_df_by_time(dataFrame):
	transposed_df = dataFrame.pivot(index='TS', columns='sensorId', values='temp').reset_index()

	return transposed_df


###Save data in CSV with corresponding name
def saveDataAsCSV(data, start, end):
	path = ''  ###Was muss hier hin?
	data.to_csv(path + 'IBM_' + str(start) + '_to_' + str(end) + '.csv')


#################################################################################


startDate = "20-10-15-00-00-00"
startDate = datetime.datetime.strptime(startDate, "%y-%m-%d-%H-%M-%S")


###Just for Debugging
# endDate = startDate + datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)
# print(endDate)
# size = 7*24*60/5
# original_data = queryTimePeriod(startDate, endDate, size)
# print('✅ Got Data')
# print(original_data)
# df_original_data = cleanData(original_data)
# print('✅ Cleaned Data')
# print(df_original_data.index[df_original_data.index.duplicated()].unique())
# print(df_original_data)
# transposed_data = transpose_df_by_time(df_original_data)
# print('✅ Transposed Data')
# print(transposed_data)
# saveDataAsCSV(transposed_data, startDate, endDate)
# print('✅ Saved Data')



counter = 1
while True:
	endDate = startDate + datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)
	size = 7*24*60/5

	if datetime.datetime.today() <= endDate:
		endDate = datetime.datetime.today()
		original_data = queryTimePeriod(startDate, endDate, size)
		print('✅ Got Data:', counter)
		df_original_data = cleanData(original_data)
		print('✅ Cleaned Data', counter)
		transposed_data = transpose_df_by_time(df_original_data)
		print('✅ Transposed Data', counter)
		saveDataAsCSV(transposed_data, startDate, endDate)
		print('✅ Saved Data', counter)
		break
	else:
		original_data = queryTimePeriod(startDate, endDate, size)
		print('✅ Got Data:', counter)
		df_original_data = cleanData(original_data)
		print('✅ Cleaned Data', counter)
		transposed_data = transpose_df_by_time(df_original_data)
		print('✅ Transposed Data', counter)
		saveDataAsCSV(transposed_data, startDate, endDate)
		print('✅ Saved Data', counter)

		startDate = endDate + datetime.timedelta(seconds=1)
		counter += 1