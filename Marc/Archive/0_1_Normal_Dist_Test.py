from scipy import stats
import pandas as pd
import plotly.express as px
import numpy as np

df_all = pd.read_csv('ALL_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True)
df_all = df_all.drop('Unnamed: 0', axis=1)
df_all['TS'] = pd.to_datetime(df_all['TS'], yearfirst=True)
df_all = df_all.set_index('TS', drop=True, verify_integrity=True)

k, p = stats.normaltest(df_all)

print(df_all)
print('Max P-Value:', max(p))

df_indiv_values = pd.DataFrame()

df_indiv_values['values'] = df_all.stack()
df_indiv_values = df_indiv_values.reset_index(drop=True)

df_indiv_values['ln'] = np.log(df_indiv_values['values'])
df_indiv_values['diff'] = df_indiv_values.diff()

print(df_indiv_values)

k_all, p_all = stats.normaltest(df_indiv_values['values'])
print('P-Value:', p_all)

fig = px.histogram(df_indiv_values['diff'])
fig.show()





