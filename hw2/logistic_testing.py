import numpy as np 
import pandas as pd
import sys
def FillData(X_test, w, outputfile):
	predict = 1 / (1 + np.exp(-np.dot(X_test, w)))
	#print(predict)
	predict[predict > 0.5] = 1
	predict[predict != 1] = 0
	predict = predict.astype(np.int32)
	row = ["id_"+str(x) for x in range(10000)]
	result_array = np.column_stack((row, predict))
	df = pd.DataFrame(result_array,columns=['id','Value'])
	df.to_csv(outputfile, index=False)
def FeatureScaling(minData, maxData, dataArray, train=False):    
	if train:
		minData = np.mean(dataArray, axis=0)
		maxData = np.std(dataArray, axis=0)
	return minData, maxData, ((dataArray - minData) / maxData)
def ReadFile(test_file):
	test_x_df = pd.read_csv(test_file)
	#test_x = test_x_df.loc[0:, "LIMIT_BAL":].values
	test_x = test_x_df.loc[0:, "LIMIT_BAL":].values
	return test_x
def OneHot(X_train, OneHotList, newFeatureList, train = True):
	newFeatureNum = 0
	newFeature = []
	for l in OneHotList:
		upBound = np.max(X_train[:,l])
		lowBound = np.min(X_train[:,l])
		if train == True:
			for u in range(int(lowBound), int(upBound) + 1):
				newCol = (X_train[:,l] == u).astype(int)
				newCol = newCol.reshape(len(newCol), 1)
				newFeatureNum += 1
				X_train = np.concatenate((X_train, newCol), axis=1)
				newFeature.append((l, u))
		else:
			for k in newFeatureList:
				if l != k[0]:
					continue
				newCol = (X_train[:,l] == k[1]).astype(int)
				newCol = newCol.reshape(len(newCol), 1)
				newFeatureNum += 1
				X_train = np.concatenate((X_train, newCol), axis=1)
	deleted = 0
	for l in OneHotList:
		X_train = np.delete(X_train, l - deleted, 1)
		deleted += 1
	return X_train, newFeatureNum, list(newFeature)

def main():
	###parameter
	w = np.array([-0.51969379,  0.03383471, -0.08793056,  0.11023524,  0.20428589,
       -0.04351334,  0.10985161, -0.10496505, -0.1811756 , -0.30431874,
       -0.00646977, -0.02164425, -0.04381427, -0.05072307,  0.27809236,
       -0.08382137, -0.2089714 , -2.64222451, -0.10325367, -0.06239712,
       -0.1821729 , -1.14345739, -1.73561877, -0.50411686, -1.06353479,
       -0.06710606, -0.23214723,  0.03043777, -0.51532935, -0.06960923,
       -0.76061576,  0.21158552,  1.48397789,  1.5247353 ,  0.85790011,
        0.84140564, -0.12141154,  2.05503922, -1.93930612, -0.05209737,
       -0.26960842, -0.11031445, -0.4696505 , -0.00664217, -0.02468903,
       -0.66342219,  0.98334696,  0.63810931,  2.02911217, -0.32478431,
       -0.14069928, -0.13599668, -1.37742845,  0.18901863,  0.01854826,
       -0.30261164, -0.93529038,  1.3693772 ,  0.57895506, -1.56984608,
       -0.02990811, -0.24876285, -0.15701311,  4.00808884,  0.10812223,
       -0.22374888, -0.13496085, -1.05540461, -4.34425038, -0.20613875,
        1.28895178, -0.10143018, -0.18508578, -0.17733076,  0.        ,
        0.13867046,  0.02920774, -0.50599968,  1.19608362,  2.30227323,
        0.8434268 ,  1.28895178,  0.0752243 , -0.0656372 , -0.27895238,
        0.        ,  0.12769643,  0.72156355,  0.25327889, -1.47913757,
        0.60817202, -0.23183077,  4.35052742, -0.16509478])
	meanData = np.array([1.67878800e+05, 3.55021500e+01, 5.11101965e+04, 4.93136727e+04,
       4.70591082e+04, 4.34738049e+04, 4.05449141e+04, 3.89666433e+04,
       5.69082700e+03, 5.83644950e+03, 5.26164530e+03, 4.86517495e+03,
       4.82106800e+03, 5.18667345e+03, 4.50306728e+10])
	stdData = np.array([1.29797463e+05, 9.26035612e+00, 7.34892830e+04, 7.14592033e+04,
       6.98597037e+04, 6.51139026e+04, 6.15167283e+04, 6.00416744e+04,
       1.69712667e+04, 2.25120391e+04, 1.87533188e+04, 1.58150559e+04,
       1.50971057e+04, 1.76213223e+04, 6.39934913e+10])
	newFeatureList = [(1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (3, 3), (5, -2), (5, -1), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, -2), (6, -1), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, -2), (7, -1), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (8, -2), (8, -1), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (9, -2), (9, -1), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (10, -2), (10, -1), (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8)]
	newFeatureNum = 78

	X_test = ReadFile(sys.argv[1])
	OneHotList = [1, 2, 3, 5, 6, 7, 8, 9, 10]
	square = [0]
	X_test = np.concatenate((X_test, X_test[:, square] ** 2), axis=1)
	X_test, newFeatureNum, newFeatureList = OneHot(X_test, OneHotList, newFeatureList, False)
	X_test = np.c_[X_test, np.ones(X_test.shape[0])]
	meanData, stdData, X_test[0:, 0:(-1 - newFeatureNum)] = FeatureScaling(meanData, stdData, X_test[0:, 0:(-1 - newFeatureNum)], False)
	FillData(X_test, w, sys.argv[2])


if __name__ == "__main__":
	main()