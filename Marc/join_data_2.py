import pandas as pd
import glob
import os


def sortDfByDate(dataframe):
	df_sorted = dataframe.sort_values(by=['TS'],ascending=True)
	return df_sorted


from_path = '/Users/marc/Desktop/OneDrive/Studium/Master/MBI/Data Science & Technology Club/PG_IBM/IBM_Data/'
all_files = glob.glob(os.path.join(from_path, "*.csv"))

df = pd.concat((pd.read_csv(f) for f in all_files))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

all_data = pd.concat(li, axis=0, ignore_index=True)
all_data = sortDfByDate(all_data)

print(all_data)

to_path = '/Users/marc/Desktop/OneDrive/Studium/Master/MBI/Data Science & Technology Club/PG_IBM/'
all_data.to_csv(to_path + 'ALL_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv')