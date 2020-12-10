import pandas as pd
import glob
import os
from Data.SensorId_Name_Directory import *
import math
import plotly.express as px


def sortDfByDate(dataframe):
	df_sorted = dataframe.sort_values(by=['TS'],ascending=True)
	return df_sorted


###Loading all files and concatenating them into a df
from_path = '/Users/marc/Desktop/OneDrive/Studium/Master/MBI/Data Science & Technology Club/PG_IBM/IBM_Data/'
all_files = glob.glob(os.path.join(from_path, "*.csv"))

df = pd.concat((pd.read_csv(f) for f in all_files))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

all_data = pd.concat(li, axis=0, ignore_index=True)
all_data = sortDfByDate(all_data)


sensorId_dict = get_name_dict()

dict_col_name = {}

for col in all_data.columns:
	try:
		sensor_col, sensor_row = sensorId_dict[col][:1], int(sensorId_dict[col][1:])
	except:
		continue
	temp_name = sensor_col + str(math.ceil(sensor_row/7))
	if temp_name not in dict_col_name:
		dict_col_name[temp_name] = []
	dict_col_name[temp_name].append(col)


df_avg_region = pd.DataFrame()


for region in dict_col_name:
	df_avg_region[region] = all_data[dict_col_name[region]].mean(axis=1)

df_avg_region['TS'] = all_data['TS']
df_avg_region['TS'] = pd.to_datetime(df_avg_region['TS'], yearfirst=True)
df_avg_region = df_avg_region.set_index('TS', drop=True, verify_integrity=True)

df_avg_region = df_avg_region.sort_index(axis=1)

print(all_data)
print(df_avg_region)

fig = px.line(df_avg_region)
fig.show()

to_path = '/Users/marc/Desktop/OneDrive/Studium/Master/MBI/Data Science & Technology Club/PG_IBM/'
df_avg_region.to_csv(to_path + 'REGION_AVG_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv')