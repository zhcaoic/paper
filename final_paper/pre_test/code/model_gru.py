# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:30:27 2019

@author: zhcao
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from matplotlib import pyplot

# parameter
timesteps = 10


# load data 
dataset = pd.read_csv('D:/python_workspace/final_paper/pre_test/input/data_560_22_noise_all_2.csv', header = 0, index_col = 0)

# gen out col-- predict 400Hz, 1200Hz, 2000Hz, 2800Hz, 4400Hz
df = dataset.shift(1)
df_out = pd.DataFrame(dataset['400'])
df_out = pd.concat([df_out, pd.DataFrame(dataset['1200'])], axis = 1)
df_out = pd.concat([df_out, pd.DataFrame(dataset['2000'])], axis = 1)
df_out = pd.concat([df_out, pd.DataFrame(dataset['2800'])], axis = 1)
df_out = pd.concat([df_out, pd.DataFrame(dataset['4400'])], axis = 1)
df = pd.concat([df, df_out], axis = 1)
df = df.dropna()
df.columns = ['0', '359', '360', '361', '399', '400', '401', '439', '440', '441', '1199', '1200', 
              '1201', '1999', '2000', '2001', '2799', '2800', '2801', '4399', '4400', '4401',
              '400_Out', '1200_Out', '2000_Out', '2800_Out', '4400_Out']
#df.to_csv("D:/python_workspace/source/multi_output/output/data_560_22_all_in_out.csv")


# preprocess data
values = df.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range = (0, 1))
values_scaled = scaler.fit_transform(values)


# split data to train & valid
train = values_scaled[:300, :]
valid = values_scaled[300:390, :]

# split into input and output
train_x, train_y = train[:, :-5], train[:, -5:]
valid_x, valid_y = valid[:, :-5], valid[:, -5:]


# reshape input to 3D [samples, timesteps, features]
train_in = train_x[0:timesteps, :]
valid_in = valid_x[0:timesteps, :]
for i in range(1, (train_x.shape[0] - (timesteps - 1))):
    train_in = concatenate((train_in, train_x[i:i + timesteps, :]), axis = 0)
for j in range(1, (valid_x.shape[0] - (timesteps - 1))):
    valid_in = concatenate((valid_in, valid_x[j:j + timesteps, :]), axis = 0)
print(train_in.shape, valid_in.shape)
train_out = train_y[timesteps - 1:]
valid_out = valid_y[timesteps - 1:]
print(train_out.shape, valid_out.shape)
train_in = train_in.reshape(((int(int(train_in.shape[0]) / timesteps)), timesteps, train_in.shape[1]))
valid_in = valid_in.reshape(((int(int(valid_in.shape[0]) / timesteps)), timesteps, valid_in.shape[1]))
print(train_in.shape, train_out.shape, valid_in.shape, valid_out.shape)


# LSTM network
model = Sequential()
model.add(LSTM(64, activation = 'tanh', input_shape = (train_in.shape[1], train_in.shape[2])))
model.add(Dense(48))
model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dropout(0.1))
model.add(Dense(8))
model.add(Dense(5))
model.compile(loss = 'mse', optimizer = 'adam')

# fit network
history = model.fit(train_in, train_out, epochs = 300, batch_size = 20, validation_data = (valid_in, valid_out), verbose = 1, shuffle = False)

# plot history
pyplot.plot(history.history['loss'], label = 'train')
pyplot.plot(history.history['val_loss'], label = 'valid')
pyplot.legend()
pyplot.show()

# save model
model.save("D:/python_workspace/final_paper/pre_test/model/test_model_multi_step_" 
           + str(timesteps) + "_1_2" + ".h5")
