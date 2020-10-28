import pandas as pd
import plotly.express as px
from SensorId_Name_Directory import *

df_all = pd.read_csv('ALL_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True)
df_all = df_all.drop(labels='Unnamed: 0', axis=1)
df_all['TS'] = pd.to_datetime(df_all['TS'], yearfirst=True)
df_all = df_all.set_index('TS', drop=True, verify_integrity=True)

window_inter = 6

df_summary = pd.DataFrame(index=df_all.index)
df_summary['max'] = df_all.max(axis=1)
df_summary['min'] = df_all.min(axis=1)
df_summary['median'] = df_all.median(axis=1)
df_summary['mean'] = df_all.mean(axis=1)
df_summary['q1'] = df_all.quantile(q=0.25, axis=1)
df_summary['q3'] = df_all.quantile(q=0.75, axis=1)
df_summary['max_sma6'] = df_summary.iloc[:,0].rolling(window=window_inter).mean()
df_summary['min_sma6'] = df_summary.iloc[:,1].rolling(window=window_inter).mean()
df_summary['median_sma6'] = df_summary.iloc[:,2].rolling(window=window_inter).mean()
df_summary['mean_sma6'] = df_summary.iloc[:,3].rolling(window=window_inter).mean()
df_summary['q1_sma6'] = df_summary.iloc[:,4].rolling(window=window_inter).mean()
df_summary['q3_sma6'] = df_summary.iloc[:,5].rolling(window=window_inter).mean()

print(df_all)
print(df_all.shape)
print(df_all.dtypes)

print(df_summary)
print(df_summary.shape)
print(df_summary.dtypes)

# fig = px.scatter(data_frame=df_all, x=df_all.index, y=df_all.columns)
# fig.show()

fig = px.line(data_frame=df_summary, x=df_summary.index, y=df_summary.columns[6:] )
fig.update_yaxes(range=[-15,40])
fig.show()