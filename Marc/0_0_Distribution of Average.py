import pandas as pd
import numpy as np
import plotly.express as px

df_average = pd.read_csv('Average_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True, index_col=0)
df_average.index = pd.to_datetime(df_average.index, yearfirst=True)
print(df_average)


df_average['diff'] = df_average['average'].diff()

### Original Distribution not Power Law Distribution --> doesn't make sense
# df_average['nat_log'] = np.log(df_average['average'])
# df_average['diff_nat_log'] = np.log(df_average['diff'])
print(df_average)

### Histograms of distribution
fig = px.histogram(data_frame=df_average, x=df_average['average'], title='Average')
fig.show()
fig = px.histogram(data_frame=df_average, x=df_average['diff'], title='Differentiated')
fig.show()

### Histograms of distribution: Original Distribution not Power Law Distribution --> doesn't make sense
# fig = px.histogram(data_frame=df_average, x=df_average['nat_log'], title='Natural Log')
# fig.show()
# fig = px.histogram(data_frame=df_average, x=df_average['diff_nat_log'], title='Natural Log of Differentiated')
# fig.show()

### Line Graph of Differentiation
fig = px.line(data_frame=df_average, x=df_average.index, y=df_average['diff'], title='Absolute Changes in Temp over time aka Differentiation')
fig.show()

df_average.to_csv('Differentiation_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv')