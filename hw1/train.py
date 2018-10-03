import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from mpl_toolkits.mplot3d import Axes3D
###consider SO2, NO2, CO, O3

def tloss(diff):
	return np.sqrt((diff**2).mean())

def train_data(data_array, y_array, para, data_num, ita, previous_loss, lamda, para_num):
	#function: y = b + w1 * x1 + w2 * x2 + ... + w9 * x9
	#print(para)
	sum_grad = 0
	diff = np.zeros(data_num)
	rmse = 0
	iterate = [x for x in range(100000)]
	rmse_list = []
	for m in range(100000):
		partial = np.zeros(para_num)
		diff = y_array - np.dot(data_array , para)
		#print(diff, diff.shape)
		partial = np.dot(data_array.transpose(), diff) * (-2)
		regular = 2 * lamda * para
		regular[para_num - 1] = 0
		partial += regular
		#print(partial, partial.size)
		sum_grad += partial ** 2
		para = para - (ita / np.sqrt(sum_grad)) * partial
		#print(ita / np.sqrt(sum_grad))
		#para = para - (ita * partial)
		rmse = tloss(diff)
		rmse_list.append(rmse)
	eva_array = np.dot(data_array , para)
	#x = np.arange(0, data_array.shape[0])
	#plt.plot(data_array[:,0], eva_array, 'r--', data_array[:,0], y_array, 'bs')
	#plt.savefig('evaluate.png')
	#print(iterate, rmse_list)
	
	plt.figure(1)
	plt.plot(iterate, rmse_list)
	plt.ylabel('RMSE')
	plt.xlabel('Iteration')
	plt.savefig('rmse_iterate.png')
	
	return para, rmse

def scaling(data_array):
	for i in range(data_array.shape[1] - 1):
		mean = np.mean(data_array[:,i])
		std = np.std(data_array[:,i])
		data_array[:,i] = (data_array[:,i] - mean) / std
		#print(data_array[:,i].mean(), data_array[:,i].std())
	return data_array


df = pd.read_csv('train1.csv')
df.columns = np.arange(0,len(df.columns))
pm = df[df[2] == 'PM2.5']
temp = df[df[2] == 'AMB_TEMP']
wind = df[df[2] == 'WIND_SPEED']
pm10 = df[df[2] == 'PM10']
so2 = df[df[2] == 'SO2']
no2 = df[df[2] == 'NO2']
co = df[df[2] == 'CO']
o3 = df[df[2] == 'O3']
pm.index = range(pm.shape[0])
temp.index = range(pm.shape[0])
wind.index = range(pm.shape[0])
pm10.index = range(pm.shape[0])
so2.index = range(pm.shape[0])
no2.index = range(pm.shape[0])
co.index = range(pm.shape[0])
o3.index = range(pm.shape[0])
#print(temp, wind)
#print(pm10, so2, no2, co, o3)
row = list(range(0, 81)) + list(range(161, 240))
valid_row = list(range(81, 161))# + list(range(161, 240))

ita = 0.005
lamda = 1000
traning_set = []
data_set = []
y_set = []
pm_days = 1
temp_days = 1
wind_days = 1
pm10_days = 1
so2_days = 1
no2_days = 1
co_days = 1
o3_days = 2
#print(pm.shape)

for r in row:
	for c in range(3,26 - pm_days + 1):
		traning_set = pm.loc[r,c:c+pm_days-1].values.tolist()
		traning_set_int = [float(x) for x in traning_set]
		if(any(n < 0 or n > 150 for n in traning_set_int)):
			continue
		
		temp_set = temp.loc[r,c+pm_days-temp_days:c+pm_days-1].values.tolist()
		temp_float = [float(x) for x in temp_set]
		if(any(n < 0 or n > 40 for n in temp_float)):
			continue
		traning_set_int.extend(list(temp_float))
		
		'''
		temp_set = wind.loc[r,c+pm_days-wind_days:c+pm_days-1].values.tolist()
		temp_float = [float(x) for x in temp_set]
		if(any(n < 0 or n > 5 for n in temp_float)):
			continue
		traning_set_int.extend(list(temp_float))
		'''
		
		temp_set = pm10.loc[r,c+pm_days-pm10_days:c+pm_days-1].values.tolist()
		temp_float = [float(x) for x in temp_set]
		if(any(n < 0 or n > 200 for n in temp_float)):
			continue
		traning_set_int.extend(list(temp_float))
		
		temp_set = so2.loc[r,c+pm_days-so2_days:c+pm_days-1].values.tolist()
		temp_float = [float(x) for x in temp_set]
		if(any(n < 0 or n > 17 for n in temp_float)):
			continue
		traning_set_int.extend(list(temp_float))

		temp_set = no2.loc[r,c+pm_days-no2_days:c+pm_days-1].values.tolist()
		temp_float = [float(x) for x in temp_set]
		if(any(n < 0 or n > 70 for n in temp_float)):
			continue
		traning_set_int.extend(list(temp_float))
		#print(temp_float)
		
		'''
		temp_set = co.loc[r,c+pm_days-co_days:c+pm_days-1].values.tolist()
		temp_float = [float(x) for x in temp_set]
		traning_set_int.extend(list(temp_float))
		
		temp_set = o3.loc[r,c+pm_days-o3_days:c+pm_days-1].values.tolist()
		temp_float = [float(x) for x in temp_set]
		traning_set_int.extend(list(temp_float))
		'''
		traning_set_int.append(1)
		#if(any(n < 0 or n > 150 for n in traning_set_int)):
			#print("find true!")
			#continue
		if (int(pm.loc[r, c+pm_days])) < 0 or (int(pm.loc[r, c+pm_days])) > 150:
			continue
		y_set.append(int(pm.loc[r, c+pm_days]))
		data_set.append(list(traning_set_int))

data_array = np.array(data_set)
y_array = np.array(y_set)
#data_array = scaling(data_array)
print(data_array, data_array.shape)
print(y_array, y_array.shape)
para = np.zeros(data_array.shape[1])
print(para.shape)
para, final_loss = train_data(data_array, y_array, para, data_array.shape[0], ita, -1, lamda, data_array.shape[1])

print(para, final_loss)

#reference:https://blog.csdn.net/xavierri/article/details/78927116

#validation
#valid_start = 0
#valid_end = 120
valid_set = []
valid_y_set = []
for r in valid_row:
	for c in range(3,26 - pm_days + 1):
		valid_traning_set = pm.loc[r,c:c+pm_days-1].values.tolist()
		valid_traning_set.extend(temp.loc[r,c+pm_days-temp_days:c+pm_days-1].values.tolist())
		#valid_traning_set.extend(wind.loc[r,c+pm_days-wind_days:c+pm_days-1].values.tolist())
		valid_traning_set.extend(pm10.loc[r,c+pm_days-pm10_days:c+pm_days-1].values.tolist())
		
		valid_traning_set.extend(so2.loc[r,c+pm_days-so2_days:c+pm_days-1].values.tolist())
		valid_traning_set.extend(no2.loc[r,c+pm_days-no2_days:c+pm_days-1].values.tolist())
		#valid_traning_set.extend(co.loc[r,c+pm_days-co_days:c+pm_days-1].values.tolist())
		#valid_traning_set.extend(o3.loc[r,c+pm_days-o3_days:c+pm_days-1].values.tolist())
		
		valid_set_int = [float(x) for x in valid_traning_set]
		valid_set_int.append(1)
		#if(any(n < 0 or n > 200 for n in valid_set_int)):
		#	continue
		valid_y_set.append(int(pm.loc[r, c+pm_days]))
		valid_set.append(list(valid_set_int))
valid_array = np.array(valid_set)
#valid_array = scaling(valid_array)
valid_y_array = np.array(valid_y_set)
print(valid_y_array.shape, valid_array.shape, para.shape)
rmse = np.sqrt(((valid_y_array - np.dot(valid_array , para))**2).mean())
print("validation error...")
print(rmse)
'''
valid_eva = np.dot(valid_array , para)
plt.figure(2)
plt.plot(valid_array[:,0], valid_eva, 'r--', valid_array[:,0], valid_y_array, 'bs')
plt.savefig('evaluate2.png')
'''
