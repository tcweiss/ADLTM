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
def unixTimeInTime(unixtime):
	date = datetime.datetime.fromtimestamp(unixtime)
	normal_time = date.strftime('%y-%m-%d %H:%M:%S')

	return normal_time


###Cleans the data and puts it into a easy to work with dataframe
def cleanData(json_data):
	datahits = json_data['hits']['hits']

	dataWithTS = pd.DataFrame()

	for i in range(0, len(datahits)):
		temporary = pd.DataFrame.from_dict(datahits[i]["_source"]["Values"])
		temporary["TS"] = unixTimeInTime(datahits[i]["_source"]["TS"])
		dataWithTS = dataWithTS.append(temporary)

	return dataWithTS


###Removes all duplicates within the dataframe in terms of sensorId and TS, keeping the last
def removeDuplicates(dataframe):
	df_withoutDuplicates = dataframe.drop_duplicates(subset=['sensorId', 'TS'], keep='last')
	return df_withoutDuplicates


###Turns a dataframe in with sensorId, temp, and TS into a dataframe where columns=sensorIds, rows=TS, and cells=temp
def transposeDfByTime(dataframe):
	transposed_df = dataframe.pivot(index='TS', columns='sensorId', values='temp').reset_index()

	return transposed_df


###Sort df by date
def sortDfByDate(dataframe):
	df_sorted = dataframe.sort_values(by=['TS'],ascending=True)
	return df_sorted


###Save data in CSV with corresponding name
def saveDataAsCSV(dataframe, start, end):
	path = '/IBM_Data/'
	dataframe.to_csv(path + 'IBM_' + str(start) + '_to_' + str(end) + '.csv', index=False)


###Get a dataframe with all duplicate rows
def getDuplicates(dataframe):
	df_duplicatedRows = dataframe[dataframe.duplicated()]
	return df_duplicatedRows


###Prints if it has duplicates or not
def hasDuplicates(dataframe):
	df_duplicates =  getDuplicates(dataframe)
	if len(df_duplicates) > 0:
		print('Has duplicates!')
	else:
		print('Has no duplicates!')


###All the way from query to saved CSV including printing status along the way
def doItAllWithPrinting(startDate, endDate, size):
	json_original_data = queryTimePeriod(startDate, endDate, size)
	print('✅ Got Data')
	print(json_original_data)

	df_original_data = cleanData(json_original_data)
	print('✅ Cleaned Data')
	print(df_original_data)

	df_no_duplicates = removeDuplicates(df_original_data)
	print('✅ Removed Duplicate Data')
	print(df_no_duplicates)

	df_transposed_data = transposeDfByTime(df_no_duplicates)
	print('✅ Transposed Data')
	print(df_transposed_data)

	df_sorted = sortDfByDate(df_transposed_data)
	print('✅ Sorted Data by date')
	print(df_sorted)

	saveDataAsCSV(df_sorted, startDate, endDate)
	print('✅ Saved Data')

	duplicatedRows = getDuplicates(df_sorted)
	hasDuplicates(duplicatedRows)


###All the way from query to saved CSV without printing status along the way
def doItAllWithoutPrinting(startDate, endDate, size):
	json_original_data = queryTimePeriod(startDate, endDate, size)

	df_original_data = cleanData(json_original_data)

	df_no_duplicates = removeDuplicates(df_original_data)

	df_transposed_data = transposeDfByTime(df_no_duplicates)

	df_sorted = sortDfByDate(df_transposed_data)

	saveDataAsCSV(df_sorted, startDate, endDate)


#################################################################################


startDate = "20-06-05-00-00-00"
startDate = datetime.datetime.strptime(startDate, "%y-%m-%d-%H-%M-%S")


###JUST FOR TESTING AND DEBUGGING PURPOSES
# endDate = startDate + datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)
# size = 7*24*60/5
# print('End Date:', endDate, 'Size:', size)
#
#
# json_original_data = queryTimePeriod(startDate, endDate, size)
# print('✅ Got Data')
# print(json_original_data)
#
# df_original_data = cleanData(json_original_data)
# print('✅ Cleaned Data')
# print(df_original_data.index[df_original_data.index.duplicated()].unique())
# print(df_original_data)
#
# df_no_duplicates = removeDuplicates(df_original_data)
# print('✅ Removed Duplicate Data')
# print(df_no_duplicates)
#
# df_transposed_data = transposeDfByTime(df_no_duplicates)
# print('✅ Transposed Data')
# print(df_transposed_data)
#
# df_sorted = sortDfByDate(df_transposed_data)
# print('✅ Sorted Data by date')
# print(df_sorted)
#
# saveDataAsCSV(df_sorted, startDate, endDate)
# print('✅ Saved Data')
#
# duplicatedRows = df_no_duplicates[df_no_duplicates.duplicated()]
#
# print('Duplicated Rows:')
# print(duplicatedRows)



counter = 1
while True:
	endDate = startDate + datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)
	size = 7*24*60/5

	if datetime.datetime.today() <= endDate:
		endDate = datetime.datetime.today().replace(microsecond=0)
		doItAllWithoutPrinting(startDate, endDate, size)
		print('✅ Done with Round #'+ str(counter))
		break
	else:
		doItAllWithoutPrinting(startDate, endDate, size)
		print('✅ Done with Round #' + str(counter))

		startDate = endDate + datetime.timedelta(seconds=1)
		counter += 1