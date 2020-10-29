import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats


'''THE DATA'''
df_all = pd.read_csv('ALL_DATA_IBM_2020-06-05-00-00-00_to_2020-10-18-16-25-33.csv', parse_dates=True)
df_all = df_all.drop(labels='Unnamed: 0', axis=1)
df_all['TS'] = pd.to_datetime(df_all['TS'], yearfirst=True)
df_all = df_all.set_index('TS', drop=True, verify_integrity=True)


### Only testing for one individual sensor atm
sensorId = '28.0211D40B0000'
df_sensorId = pd.DataFrame()
df_sensorId[sensorId] = df_all[sensorId]


### Splitting up TS into trend, seasonal, and residual and putting it in a DF with the actual temperature
result = seasonal_decompose(df_sensorId, model='multiplicative', period=2016)
df_result = pd.DataFrame()

df_result['seasonal'] = result.seasonal
df_result['residual'] = result.resid
df_result['trend'] = result.trend
df_result[sensorId] = df_sensorId[sensorId]

print(df_result)


###Plotting all 4 lines
fig = px.line(df_result)
fig.show()


###Checking if roughly normal
fig = px.histogram(df_result['residual'])
fig.show()


'''Preprocessing'''
###Cutting data to last 4 weeks
df_result_cut = df_result[-16128:]
print(df_result_cut)


###Splitting in test & training data with 80-20
split = int(0.8 * len(df_result_cut))
train, test = df_result_cut[:split] , df_result_cut[split:]
print(train.shape, test.shape)


scaler = StandardScaler()
scaler = scaler.fit(train[['residual']])

train['residual'] = scaler.transform(train[['residual']])
test['residual'] = scaler.transform(test[['residual']])


TIME_STEPS = 24

def create_sequences(X, y, time_steps=TIME_STEPS):
	Xs, ys = [], []
	for i in range(len(X) - time_steps):
		Xs.append(X.iloc[i:(i + time_steps)].values)
		ys.append(y.iloc[i + time_steps])

	return np.array(Xs), np.array(ys)


X_train, y_train = create_sequences(train[['residual']], train['residual'])
X_test, y_test = create_sequences(test[['residual']], test['residual'])

print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')


'''Build the Model'''
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()


'''Train the Model'''
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

model.evaluate(X_test, y_test)


'''Determine Anomalies'''
X_train_pred = model.predict(X_train, verbose=0)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of Samples')
plt.show()

threshold = np.max(train_mae_loss)
print(f'Reconstruction error threshold: {threshold}')



X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

plt.hist(test_mae_loss, bins=50)
plt.xlabel('Test MAE loss')
plt.ylabel('Number of samples')
plt.show()



test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['residual'] = test[TIME_STEPS:]['residual']

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df.index, y=test_score_df['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df.index, y=test_score_df['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()

anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
anomalies.shape


'''Visualize Anomalies'''
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df.index, y=scaler.inverse_transform(test_score_df['residual']), name='Temperature'))
fig.add_trace(go.Scatter(x=anomalies.index, y=scaler.inverse_transform(anomalies['residual']), mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()