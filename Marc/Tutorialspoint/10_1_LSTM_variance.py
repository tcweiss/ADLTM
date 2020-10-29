import pandas as pd
import plotly.express as px
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed


'''THE DATA'''
df_all = pd.read_csv('ALL_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True)
df_all = df_all.drop(labels='Unnamed: 0', axis=1)
df_all['TS'] = pd.to_datetime(df_all['TS'], yearfirst=True)
df_all = df_all.set_index('TS', drop=True, verify_integrity=True)
df_all['var'] = df_all.var(axis=1)

print(df_all)


'''Visualization'''
fig = px.line(df_all['var'])
fig.show()


'''Preprocessing'''
###Cutting data to last 4 weeks
df_cut = df_all[-8064:]
print(df_cut)


###Selecting one sensor as the output variable
output_sensorId = '28.0211D40B0000'


split = int(0.8 * len(df_cut))
train, test = df_cut[:split] , df_cut[split:]
print(train.shape, test.shape)


scaler = StandardScaler()
scaler = scaler.fit(train[['Close']])

train['Close'] = scaler.transform(train[['Close']])
test['Close'] = scaler.transform(test[['Close']])

TIME_STEPS = 30


def create_sequences(X, y, time_steps=TIME_STEPS):
	Xs, ys = [], []
	for i in range(len(X) - time_steps):
		Xs.append(X.iloc[i:(i + time_steps)].values)
		ys.append(y.iloc[i + time_steps])

	return np.array(Xs), np.array(ys)


X_train, y_train = create_sequences(train[['Close']], train['Close'])
X_test, y_test = create_sequences(test[['Close']], test['Close'])

print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')