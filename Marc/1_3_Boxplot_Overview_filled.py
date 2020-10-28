import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from SensorId_Name_Directory import *

df_all = pd.read_csv('ALL_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True)
df_all = df_all.drop(labels='Unnamed: 0', axis=1)
df_all['TS'] = pd.to_datetime(df_all['TS'], yearfirst=True)
df_all = df_all.set_index('TS', drop=True, verify_integrity=True)

window_inter = 6

df_summary = pd.DataFrame(index=df_all.index)
# df_summary['max'] = df_all.max(axis=1)
# df_summary['min'] = df_all.min(axis=1)
df_summary['median'] = df_all.median(axis=1)
# df_summary['mean'] = df_all.mean(axis=1)
df_summary['q1'] = df_all.quantile(q=0.25, axis=1)
df_summary['q3'] = df_all.quantile(q=0.75, axis=1)


print(df_all)
print(df_all.shape)
print(df_all.dtypes)

print(df_summary)
print(df_summary.shape)
print(df_summary.dtypes)

# fig = px.scatter(data_frame=df_all, x=df_all.index, y=df_all.columns)
# fig.show()

# fig = px.line(data_frame=df_summary, x=df_summary.index, y=df_summary['median'] )
# fig.update_yaxes(range=[-15,40])


x = list(df_summary.index)
x_rev = list(df_summary.index)[::-1]
y_q3 = list(df_summary['q3'])
y_median = list(df_summary['median'])
y_q1 = list(df_summary['q1'])
y_q1_rev = y_q1[::-1]

fig = go.Figure()
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_q3+y_q1_rev,
    fill='toself',
    fillcolor='rgba(0,29,108,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=False))
fig.add_trace(go.Scatter(
    x=x, y=y_median,
    line_color='rgb(0,17,65)'))


fig.show()