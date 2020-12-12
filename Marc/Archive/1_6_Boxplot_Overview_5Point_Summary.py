import pandas as pd
import plotly.graph_objects as go

df_all = pd.read_csv('ALL_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True)
df_all = df_all.drop(labels='Unnamed: 0', axis=1)
df_all['TS'] = pd.to_datetime(df_all['TS'], yearfirst=True)
df_all = df_all.set_index('TS', drop=True, verify_integrity=True)

window_inter = 6

df_summary = pd.DataFrame(index=df_all.index)
df_summary['max'] = df_all.max(axis=1)#
df_summary['q3'] = df_all.quantile(q=0.75, axis=1)
df_summary['median'] = df_all.median(axis=1)
df_summary['mean'] = df_all.mean(axis=1)
df_summary['q1'] = df_all.quantile(q=0.25, axis=1)
df_summary['min'] = df_all.min(axis=1)


print(df_all)
print(df_all.shape)
print(df_all.dtypes)

print(df_summary)
print(df_summary.shape)
print(df_summary.dtypes)
print(df_summary.describe())


# fig = px.box(data_frame=df_summary)
# fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
# fig.show()


fig = go.Figure()
for column in df_summary.columns:
	fig.add_trace(go.Box(y=df_summary[column], name=column, marker_color = '#0f62fe'))
fig.update_layout(plot_bgcolor='rgb(255,255,255)', showlegend=False)

fig.show()