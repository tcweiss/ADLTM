import pandas as pd
import glob
import os

from_path = '' #Was muss hier hin?
all_files = glob.glob(os.path.join(from_path, "*.csv"))

df = pd.concat((pd.read_csv(f) for f in all_files))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

all_data = pd.concat(li, axis=0, ignore_index=True)

print(all_data)

to_path = '' #Was muss hier hin? Darf nicht from_path=to_path
all_data.to_csv(to_path + '.csv') #Filename