import numpy as np
import pandas as pd 
import sys
import warnings
#0.82020
def Classify(train_x, label):
	list1 = []
	list2 = []
	for l in range(label.shape[0]):
		if label[l] == 0:
			list1.append(train_x[l, 0:])
		else:
			list2.append(train_x[l, 0:])
	return np.vstack(list1), np.vstack(list2)

def ReadFile(train_file, label_file, test_file):
	train_x_df = pd.read_csv(train_file)
	train_x = train_x_df.loc[0:, "LIMIT_BAL":].values
	label_df = pd.read_csv(label_file)
	label = label_df.loc[0:, "Y"].values
	test_x_df = pd.read_csv(test_file)
	test_x = test_x_df.loc[0:, "LIMIT_BAL":].values
	return train_x, label, test_x
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

def FeatureScaling(minData, maxData, dataArray, train=False):    
	#print(dataArray)
	if train:
		minData = np.mean(dataArray, axis=0)
		maxData = np.std(dataArray, axis=0)
	#print((dataArray - minData) / maxData)
	return minData, maxData, ((dataArray - minData) / maxData)

def main():
	#np.set_printoptions(threshold=np.inf)
	warnings.filterwarnings("ignore")
	if (len(sys.argv) < 5):
		print("Please specify 1: the traning data file 2: the label file 3: the testing data file 4: the output file")
		sys.exit(1)
	X_train, label, X_test = ReadFile(sys.argv[1], sys.argv[2], sys.argv[3])
	OneHotList = [1, 2, 3, 5, 6, 7, 8, 9, 10]
	X_train, newFeatureNum, newFeatureList = OneHot(X_train, OneHotList, [], True)
	X_test, newFeatureNum, newFeatureList = OneHot(X_test, OneHotList, newFeatureList, False)
	X_train = X_train.astype(float)
	X_test = X_test.astype(float)
	minData, maxData, X_train[0:, 0:(-1 * newFeatureNum)] = FeatureScaling(0, 0, X_train[0:, 0:(-1 * newFeatureNum)], True)
	minData, maxData, X_test[0:, 0:(-1 * newFeatureNum)] = FeatureScaling(minData, maxData, X_test[0:, 0:(-1 * newFeatureNum)], False)
	class1, class2 = Classify(X_train, label)
	#print(class1.shape, class2.shape)
	D = X_train.shape[1] / 2
	mean1 = np.mean(class1, axis = 0)
	mean2 = np.mean(class2, axis = 0)
	minus_mean1 = X_test - mean1
	minus_mean2 = X_test - mean2
	cov = (class1.shape[0] / X_train.shape[0]) * np.cov(np.transpose(class1)) + (class2.shape[0] / X_train.shape[0]) * np.cov(np.transpose(class2))
	cov_det = np.linalg.det(cov)
	#print(cov_det)
	#print(cov, file = open("cov.txt", "a"))
	cov_inv = np.linalg.pinv(cov)
	matrix_mult1 = np.diag((-1/2) * np.dot(minus_mean1, cov_inv).dot(np.transpose(minus_mean1)))
	matrix_mult2 = np.diag((-1/2) * np.dot(minus_mean2, cov_inv).dot(np.transpose(minus_mean2)))
	constant_num = np.power(2 * np.pi, -D) * (1 / (np.sqrt(cov_det) + 1e-20))
	prob_x_c1 = constant_num * np.exp(matrix_mult1)
	prob_x_c2 = constant_num * np.exp(matrix_mult2)
	prob_c1 = class1.shape[0] / X_train.shape[0]
	prob_c2 = class2.shape[0] / X_train.shape[0]
	prob1 = (prob_x_c1 * prob_c1) / (prob_x_c1 * prob_c1 + prob_x_c2 * prob_c2)
	prob1[np.isnan(prob1)] = 0
	prob1[prob1 > 0.5] = 0
	prob1[prob1 != 0] = 1
	#print(prob1.dtype)
	prob1 = prob1.astype(np.int32)
	row = ["id_"+str(x) for x in range(10000)]
	result_array = np.column_stack((row, prob1))
	df = pd.DataFrame(result_array,columns=['id','Value'])
	df.to_csv(sys.argv[4], index=False)


if __name__ == "__main__":
	main()