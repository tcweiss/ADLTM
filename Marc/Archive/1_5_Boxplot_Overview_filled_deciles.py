import pandas as pd
import plotly.graph_objects as go


###get pandas column as list
def get_top_list(df, c_name):
    temp_list = list(df[c_name])
    return temp_list

def get_bottom_list(df,c_name):
    temp_list = list(df[c_name])[::-1]
    return temp_list

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

for i in range(0,19):
    p = round(0.05 + i * 0.05,2)
    name = 'p'+str(p)
    df_summary[name] = df_all.quantile(q=p, axis=1)


print(df_all)
print(df_all.shape)
print(df_all.dtypes)

print(df_summary)
print(df_summary.shape)
print(df_summary.dtypes)


x = list(df_summary.index)
x_rev = list(df_summary.index)[::-1]

y_min_rev = get_bottom_list(df_summary, 'min')
y_p05_rev = get_bottom_list(df_summary, 'p0.05')
y_p10_rev = get_bottom_list(df_summary, 'p0.1')
y_p15_rev = get_bottom_list(df_summary, 'p0.15')
y_p20_rev = get_bottom_list(df_summary, 'p0.2')
y_p25_rev = get_bottom_list(df_summary, 'p0.25')
y_p30_rev = get_bottom_list(df_summary, 'p0.3')
y_p35_rev = get_bottom_list(df_summary, 'p0.35')
y_p40_rev = get_bottom_list(df_summary, 'p0.4')
y_p45_rev = get_bottom_list(df_summary, 'p0.45')
y_median = get_top_list(df_summary, 'median')
y_p55 = get_top_list(df_summary, 'p0.55')
y_p60 = get_top_list(df_summary, 'p0.6')
y_p65 = get_top_list(df_summary, 'p0.65')
y_p70 = get_top_list(df_summary, 'p0.7')
y_p75 = get_top_list(df_summary, 'p0.75')
y_p80 = get_top_list(df_summary, 'p0.8')
y_p85 = get_top_list(df_summary, 'p0.85')
y_p90 = get_top_list(df_summary, 'p0.9')
y_p95 = get_top_list(df_summary, 'p0.95')
y_max = get_top_list(df_summary, 'max')


fig = go.Figure()
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_max+y_min_rev,
    fill='toself',
    fillcolor='rgba(237,245,255,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=True,
    name='P100'))
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_p95+y_p05_rev,
    fill='toself',
    fillcolor='rgba(208,226,255,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=True,
    name='P90'))
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_p90+y_p10_rev,
    fill='toself',
    fillcolor='rgba(166,200,255,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=True,
    name='P80'))
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_p85+y_p15_rev,
    fill='toself',
    fillcolor='rgba(120,169,255,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=True,
    name='P70'))
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_p80+y_p20_rev,
    fill='toself',
    fillcolor='rgba(69,137,255,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=True,
    name='P60'))
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_p75+y_p25_rev,
    fill='toself',
    fillcolor='rgba(15,98,254,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=True,
    name='P50'))
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_p70+y_p30_rev,
    fill='toself',
    fillcolor='rgba(0,67,206,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=True,
    name='P40'))
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_p65+y_p35_rev,
    fill='toself',
    fillcolor='rgba(0,45,156,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=True,
    name='P30'))
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_p60+y_p40_rev,
    fill='toself',
    fillcolor='rgba(0,29,108,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=True,
    name='P20'))
fig.add_trace(go.Scatter(
	x=x+x_rev,
    y=y_p55+y_p45_rev,
    fill='toself',
    fillcolor='rgba(0,17,65,1)',
    line_color='rgba(255,255,255,0)',
    showlegend=True,
    name='P10'))
fig.add_trace(go.Scatter(
    x=x, y=y_median,
    line_color='rgb(255,255,255)',
    line=dict(width=1),
    showlegend=True,
    name='Median'))

# fig.update_layout(plot_bgcolor='rgb(0,0,0)')
fig.update_layout(plot_bgcolor='rgb(255,255,255)')

fig.show()