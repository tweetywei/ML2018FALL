import pandas as pd
import numpy as np
import sys
#para = np.array([0.03591933,-0.02346021,-0.00550479,0.00645854,0.06112335,0.05393404,-0.16488058,0.11061338,0.77577751,4.39894842])
#para = np.array([0.10676936,-0.02925451,0.0413517,-0.11709358,0.0591571 ,0.11237673,-0.18561242,0.14502906,0.81472381,2.22931181])
#para = [0.13772614,-0.22811708,0.08354281,0.6578869,0.1718664,0.0766841,0.01126696,2.88928371,0.02302005,-2.07430809]
#para = [-0.03638008,0.06807449,0.09320396,-0.22805346,0.10522287,0.63986689,0.18442023,0.04935317,-0.76294569]
#para =  [-0.01834959,0.58892098,0.23076597,0.08510117,-1.7548938]
#para = [5.61126286e-02,-2.37969369e-02,2.23930739e-03,-4.73188080e-02,6.14113801e-02,6.11709123e-02,-2.05730202e-01,1.14359950e-01,5.88613666e-01,1.85395434e-01,8.92841427e-02,5.98024619e-02,1.46299716e+00,3.91948955e-02,-2.54023331e+00]
#para = [0.94333128,1.93393372]
#para = [-0.08816272,-0.07196398,0.97978691,-0.04691205,-0.14649299,0.08147248,2.29272291]
#para = [0.07770115,-0.01236961,0.04088286,-0.07390638,0.07640325,0.1128939,-0.22115933,0.0792663,0.72795411,-0.04523214,-0.02562305,0.14339966,-0.07638494,-0.02452982,-0.06489283,0.01389666,0.02197126,0.07069443
#,0.03703448,0.14070517,-0.06365844,0.08196655,-0.20024847,0.29883371,-0.27701849,-0.04169219,-0.03758049,-0.0020867,0.00239682,-0.05157749,0.03783738,-0.00743324,-0.02925925,-0.00728936,0.06949691,0.06296044,1.16651639]
#para = [0.07668868,-0.02051402,0.00880988,-0.05794572,0.07347964,0.10253927,-0.22195395,0.11728887,0.83655002,2.78591687]
#para = [-0.09977733,-0.00920201,0.9072051,-0.02170585,0.00557578,0.03814531,-0.25651211,-0.12512497,-0.17693883,-0.00814586,0.03544217,0.05616723,-0.10262589,-0.03800806,0.24680786,-0.03070341,0.07788117,0.03682958,0.09980031]
#para = [-0.11191907,-0.00221135,0.91394881,-0.15928898,0.02936393,0.11626876,0.00093533,-0.11252387,0.19708447,-0.32524387,-0.08405366,0.57256954,-0.08142695,0.01986156,0.10293399,0.02321713]
para = [7.80392382e-01,-2.85785828e-04,7.48020836e-02,4.65170900e-01,8.93764390e-02,-8.76204603e-01]
#for report 2
#para = [5.81830848e-02,7.72714543e-04,-5.40332289e-02,1.72107069e-01,1.66662368e-01,-2.04400106e-02,-1.78812974e-01,1.57624519e-01,4.49818009e-01,8.18480903e-01]
#for report 3
#para = [5.82278418e-02,7.39909700e-04,-5.41439619e-02,1.72167947e-01,1.66736136e-01,-2.04689989e-02,-1.79029119e-01,1.57731334e-01,4.49981941e-01,8.18433304e-01]
#para = [0.05412542,0.00379767,-0.04367289,0.16642439,0.15975336,-0.01774089,-0.15867317,0.14797506,0.43399108,0.82303914]
#para = [0.02397871,0.02388269,0.02417745,0.02476068,0.02451022,0.02369381,0.02396944,0.02566558,0.02839589,0.97985325]

pm_days = 1
temp_days = 1
wind_days = 1
pm10_days = 1
so2_days = 1
no2_days = 1
co_days = 2
o3_days = 2

#columns_index = np.arange(0,12)
print(sys.argv[1])
df = pd.read_csv(sys.argv[1], header=None)
#df.reset_index()
print(df)
df.columns = np.arange(0,len(df.columns))
pm = df[df[1] == 'PM2.5']
temp = df[df[1] == 'AMB_TEMP']
wind = df[df[1] == 'WIND_SPEED']
pm10 = df[df[1] == 'PM10']
so2 = df[df[1] == 'SO2']
no2 = df[df[1] == 'NO2']
co = df[df[1] == 'CO']
o3 = df[df[1] == 'O3']
print(pm.shape)
#print(temp)
pm.index = range(pm.shape[0])
temp.index = range(pm.shape[0])
wind.index = range(pm.shape[0])
pm10.index = range(pm.shape[0])
so2.index = range(pm.shape[0])
no2.index = range(pm.shape[0])
co.index = range(pm.shape[0])
o3.index = range(pm.shape[0])
#print(pm.shape)
#print(pm)
data_set = []
y_set = []
for r in range(260):
	testing_set = pm.loc[r,10-pm_days+1:].values.tolist()
	testing_set.extend(temp.loc[r,10-temp_days+1:].values.tolist())
	#testing_set.extend(wind.loc[r,10-temp_days+1:].values.tolist())
	testing_set.extend(pm10.loc[r,10-temp_days+1:].values.tolist())
	testing_set.extend(so2.loc[r,10-temp_days+1:].values.tolist())
	testing_set.extend(no2.loc[r,10-temp_days+1:].values.tolist())
	#testing_set.extend(co.loc[r,10-temp_days+1:].values.tolist())
	#testing_set.extend(o3.loc[r,10-temp_days+1:].values.tolist())
	#print(testing_set)
	testing_set_int = [float(x) for x in testing_set]
	testing_set_int.append(1)
	data_set.append(list(testing_set_int))
data_array = np.array(data_set)
#print(data_array)
ans_array = np.dot(data_array ,para)
#print(ans_array)
#create row list
row = ["id_"+str(x) for x in range(260)]
#print(row)
result_array = np.column_stack((row, ans_array))
#print(result_array)
df = pd.DataFrame(result_array,columns=['id','value'])
print(df)
df.to_csv(sys.argv[2], index=False)
'''
for r in row:
	for c in range(3,26 - pm_days):
		traning_set = pm.loc[r,c:c+pm_days].values.tolist()
		traning_set.extend([pm10.loc[r,c+pm_days+1],so2.loc[r,c+pm_days+1],no2.loc[r,c+pm_days+1],co.loc[r,c+pm_days+1],o3.loc[r,c+pm_days+1]])
		traning_set_int = []
		try:
			traning_set_int = [float(x) for x in traning_set]
		except ValueError:
			print("error",pm10.loc[r,c+pm_days+1],so2.loc[r,c+pm_days+1],no2.loc[r,c+pm_days+1],co.loc[r,c+pm_days+1],o3.loc[r,c+pm_days+1])
		#traning_set_int.append()
		traning_set_int.append(1)
		if(any(n < 0 or n > 150 for n in traning_set_int)):
			continue
		y_set.append(int(pm.loc[r, c+pm_days+1]))
		data_set.append(list(traning_set_int))
'''