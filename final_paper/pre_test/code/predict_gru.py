# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:40:29 2019

@author: zhcao
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from math import sqrt
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

# preprocess data
values = df.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range = (0, 1))
values_scaled = scaler.fit_transform(values)

# split predict data
pre_values = values_scaled[390:, :]
pre_x, pre_y = pre_values[:, :-5], pre_values[:, -5:]

# reshape input to 3D [samples, timesteps, features]
pre_in = pre_x[0:timesteps, :]
for i in range(1, (pre_x.shape[0] - (timesteps - 1))):
    pre_in = concatenate((pre_in, pre_x[i:i + timesteps, :]), axis = 0)
print(pre_in.shape)
pre_out = pre_y[timesteps - 1:]
print(pre_out.shape)
pre_in = pre_in.reshape(((int(int(pre_in.shape[0]) / timesteps)), timesteps, pre_in.shape[1]))
print(pre_in.shape, pre_out.shape)

# predict
model = load_model("D:/python_workspace/final_paper/pre_test/model/test_model_multi_step_" 
                   + str(timesteps) + "_1_2.h5")
pre_result = model.predict(pre_in)
print(pre_result.shape)

# deal with result
pre_x = pre_x[timesteps - 1:, :]
# invert scaler forecast out
inv_pre = concatenate((pre_x[:, :], pre_result), axis = 1)
inv_pre = scaler.inverse_transform(inv_pre)
value_pre = inv_pre[:, -5:]

# invert scaler actual out
pre_out = pre_out.reshape((len(pre_out), 5))
inv_real = concatenate((pre_x[:, :], pre_out), axis = 1)
inv_real = scaler.inverse_transform(inv_real)
value_real = inv_real[:, -5:]

# 取基波数据
base_value = inv_real[:, 5]
value_399 = inv_real[:, 4]
value_401 = inv_real[:, 6]

print(type(value_real))
print(type(value_pre))
print(type(base_value))
print(value_real.shape, value_pre.shape, base_value.shape, value_399.shape, value_401.shape)

# 计算基波正确情况与谐波含量
result_400_real = []
result_400_pre = []
contain_1200_real = []
contain_1200_pre = []
contain_2000_real = []
contain_2000_pre = []
contain_2800_real = []
contain_2800_pre = []
contain_4400_real = []
contain_4400_pre = []
for p in range(0, len(base_value)):
    if (value_real[p, 0] >= value_401[p]) and (value_real[p, 0] >= value_399[p]):
        result_400_real.append("normal")
    else:
        result_400_real.append("error")
    if (value_pre[p, 0] >= value_401[p]) and (value_pre[p, 0] >= value_399[p]):
        result_400_pre.append("normal")
    else:
        result_400_pre.append("error")
    contain_1200_real.append((value_real[p, 1] / base_value[p]))
    contain_1200_pre.append((value_pre[p, 1] / base_value[p]))
    contain_2000_real.append((value_real[p, 2] / base_value[p]))
    contain_2000_pre.append((value_pre[p, 2] / base_value[p]))
    contain_2800_real.append((value_real[p, 3] / base_value[p]))
    contain_2800_pre.append((value_pre[p, 3] / base_value[p]))
    contain_4400_real.append((value_real[p, 4] / base_value[p]))
    contain_4400_pre.append((value_pre[p, 4] / base_value[p]))
    

# calculate RMSE
rmse_400 = sqrt(mean_squared_error(value_real[:, 0], value_pre[:, 0]))
rmse_1200 = sqrt(mean_squared_error(value_real[:, 1], value_pre[:, 1]))
rmse_2000 = sqrt(mean_squared_error(value_real[:, 2], value_pre[:, 2]))
rmse_2800 = sqrt(mean_squared_error(value_real[:, 3], value_pre[:, 3]))
rmse_4400 = sqrt(mean_squared_error(value_real[:, 4], value_pre[:, 4]))

print('400Hz RMSE: %.6f' % rmse_400)
print('1200Hz RMSE: %.6f' % rmse_1200)
print('2000Hz RMSE: %.6f' % rmse_2000)
print('2800Hz RMSE: %.6f' % rmse_2800)
print('4400Hz RMSE: %.6f' % rmse_4400)

# save result
fo = "D:/python_workspace/final_paper/pre_test/output/predict_result_test_1_2.txt"
time = dataset.index.tolist()
time_test = time[390 + timesteps:]

error_400_cnt = 0
error_1200_cnt = 0
error_2000_cnt = 0
error_2800_cnt = 0
error_4400_cnt = 0

fout = open(fo, "w")
fout.write("400Hz RMSE:" + "\t" + str(rmse_400) + "\n")
fout.write("1200Hz RMSE:" + "\t" + str(rmse_1200) + "\n")
fout.write("2000Hz RMSE:" + "\t" + str(rmse_2000) + "\n")
fout.write("2800Hz RMSE:" + "\t" + str(rmse_2800) + "\n")
fout.write("4400Hz RMSE:" + "\t" + str(rmse_4400) + "\n")
fout.write("Time" + "\t" + "400Hz_REAL" + "\t" + "400Hz_PRE" + "\t" + "400_real_result" + "\t" + "400_pre_result" + "\t" 
           + "1200Hz_REAL" + "\t" + "1200Hz_PRE" + "\t" + "1200%_real" + "\t" + "1200%_pre"
           + "2000Hz_REAL" + "\t" + "2000Hz_PRE" + "\t" + "2000%_real" + "\t" + "2000%_pre"
           + "2800Hz_REAL" + "\t" + "2800Hz_PRE" + "\t" + "2800%_real" + "\t" + "2800%_pre"
           + "4400Hz_REAL" + "\t" + "4400Hz_PRE" + "\t" + "4400%_real" + "\t" + "4400%_pre\n")
for x in range(0, len(time_test)):
    fout.write(str(time_test[x]) + "\t" 
               + str(int(value_real[x, 0])) + "\t" + str(int(value_pre[x, 0])) + "\t" + str(result_400_real[x]) + "\t" + str(result_400_pre[x]) + "\t" 
               + str(int(value_real[x, 1])) + "\t" + str(int(value_pre[x, 1])) + "\t" + str(contain_1200_real[x]) + "\t" + str(contain_1200_pre[x]) + "\t" 
               + str(int(value_real[x, 2])) + "\t" + str(int(value_pre[x, 2])) + "\t" + str(contain_2000_real[x]) + "\t" + str(contain_2000_pre[x]) + "\t"
               + str(int(value_real[x, 3])) + "\t" + str(int(value_pre[x, 3])) + "\t" + str(contain_2800_real[x]) + "\t" + str(contain_2800_pre[x]) + "\t"
               + str(int(value_real[x, 4])) + "\t" + str(int(value_pre[x, 4])) + "\t" + str(contain_4400_real[x]) + "\t" + str(contain_4400_pre[x]) + "\n")
    # 计算准确率
    if (result_400_real[x] != result_400_pre[x]) or ((result_400_real[x] == "error") and (result_400_pre[x] == "error")):
        error_400_cnt = error_400_cnt + 1
    if ((contain_1200_real[x] > 0.04) and (contain_1200_pre[x] <= 0.04)) or ((contain_1200_real[x] <= 0.04) and (contain_1200_pre[x] > 0.04)):
        error_1200_cnt = error_1200_cnt + 1
    if ((contain_2000_real[x] > 0.04) and (contain_2000_pre[x] <= 0.04)) or ((contain_2000_real[x] <= 0.04) and (contain_2000_pre[x] > 0.04)):
        error_2000_cnt = error_2000_cnt + 1
    if ((contain_2800_real[x] > 0.04) and (contain_2800_pre[x] <= 0.04)) or ((contain_2800_real[x] <= 0.04) and (contain_2800_pre[x] > 0.04)):
        error_2800_cnt = error_2800_cnt + 1
    if ((contain_4400_real[x] > 0.04) and (contain_4400_pre[x] <= 0.04)) or ((contain_4400_real[x] <= 0.04) and (contain_4400_pre[x] > 0.04)):
        error_4400_cnt = error_4400_cnt + 1

# 错误率
per_400_error = error_400_cnt / len(time_test)    
per_1200_error = error_1200_cnt / len(time_test) 
per_2000_error = error_2000_cnt / len(time_test) 
per_2800_error = error_2800_cnt / len(time_test) 
per_4400_error = error_4400_cnt / len(time_test)  
# 正确率
per_400_right = 1 - per_400_error
per_1200_right = 1 - per_1200_error
per_2000_right = 1 - per_2000_error
per_2800_right = 1 - per_2800_error
per_4400_right = 1 - per_4400_error
print("400Hz 正确率：" + str(per_400_right) + " 错误数: " + str(error_400_cnt))
print("1200Hz 正确率：" + str(per_1200_right) + " 错误数: " + str(error_1200_cnt))
print("2000Hz 正确率：" + str(per_2000_right) + " 错误数: " + str(error_2000_cnt))
print("2800Hz 正确率：" + str(per_2800_right) + " 错误数: " + str(error_2800_cnt))
print("4400Hz 正确率：" + str(per_4400_right) + " 错误数: " + str(error_4400_cnt))
fout.write("400Hz 正确率：" + str(per_400_right) + "\n")
fout.write("1200Hz 正确率：" + str(per_1200_right) + "\n")
fout.write("2000Hz 正确率：" + str(per_2000_right) + "\n")
fout.write("2800Hz 正确率：" + str(per_2800_right) + "\n")
fout.write("4400Hz 正确率：" + str(per_4400_right) + "\n")
    
fout.close()


# plot
x = time_test
y = 2
pyplot.plot(x, value_pre[:, y], label = 'pre')
pyplot.plot(x, value_real[:, y], label = 'real')
pyplot.legend()
pyplot.show()



