import numpy as np 
import pandas as pd
import sys

def FillData(X_test, w, outputfile):
	predict = 1 / (1 + np.exp(-np.dot(X_test, w)))
	print(predict)
	predict[predict > 0.5] = 1
	predict[predict != 1] = 0
	predict = predict.astype(np.int32)
	row = ["id_"+str(x) for x in range(10000)]
	result_array = np.column_stack((row, predict))
	df = pd.DataFrame(result_array,columns=['id','Value'])
	df.to_csv(outputfile, index=False)

def ValidData(X_train, label, w):
	predict = 1 / (1 + np.exp(-np.dot(X_train, w)))
	predict[predict > 0.5] = 1
	predict[predict != 1] = 0
	predict = predict.astype(np.int32)
	correct = (predict == label)
	accuracy = correct.sum() / correct.size
	return accuracy

def fix_error(error, exp_fx, label):
	count = 0
	for l in range(error.shape[0]):
		if exp_fx[l] is np.inf:
			error[l] = -1 * label[l]
			count += 1
	return error

def FeatureScaling(minData, maxData, dataArray, train=False):    
	if train:
		minData = np.mean(dataArray, axis=0)
		maxData = np.std(dataArray, axis=0)
	return minData, maxData, ((dataArray - minData) / maxData)

def TrainData(X_train, label, w, eta):
	learningW = np.zeros(w.shape)
	for l in range(3000):
		fx = np.dot(X_train, w)
		exp_fx = (np.exp(-fx))
		error = (1 / (1 + np.exp((-1) * fx))) - label
		error = fix_error(error, exp_fx, label)
		if l % 100 == 0:
			sigmoid = (1 / (1 + np.exp((-1) * fx)))
			loss = -np.mean(label * np.log(sigmoid) + (1 - label) * np.log(1 - sigmoid))
		gradW = np.dot(X_train.T, error)
		learningW += gradW ** 2
		w = w - (eta / (np.sqrt(learningW) + 1e-20)) * gradW
	train_accuracy = ValidData(X_train, label, w)
	return w

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
def main():
	if (len(sys.argv) < 5):
		print("Please specify 1: the traning data file 2: the label file 3: the testing data file 4: the output file")
		sys.exit(1)
	X_train, label, X_test = ReadFile(sys.argv[1], sys.argv[2], sys.argv[3])
	eta = 0.05
	newFeatureNum = 0
	OneHotList = [1, 2, 3, 5, 6, 7, 8, 9, 10]
	square = [0]
	X_train = np.concatenate((X_train, X_train[:, square] ** 2), axis=1)
	X_test = np.concatenate((X_test, X_test[:, square] ** 2), axis=1)
	X_train, newFeatureNum, newFeatureList = OneHot(X_train, OneHotList, [], True)
	X_train = np.c_[X_train, np.ones(X_train.shape[0])]       

	minData, maxData, X_train[0:, 0:(-1 - newFeatureNum)] = FeatureScaling(0, 0, X_train[0:, 0:(-1 - newFeatureNum)], True)
	w = np.zeros(X_train.shape[1])
	w = TrainData(X_train[0:, 0:], label[0:], w, eta)
	'''
	minData, maxData, X_train[0:1000, 0:(-1 - newFeatureNum)] = FeatureScaling(minData, maxData, X_train[0:1000, 0:(-1 - newFeatureNum)], False)
	accuracy = ValidData(X_train[0:1000, 0:], label[0:1000], w)
	print(accuracy)
	'''
	
	print("w", file = open("parameter.txt", "a"))
	print(repr(w), file = open("parameter.txt", "a"))
	print("mean", file = open("parameter.txt", "a"))
	print(repr(minData), file = open("parameter.txt", "a"))
	print("std", file = open("parameter.txt", "a"))
	print(repr(maxData), file = open("parameter.txt", "a"))
	print("newFeatureList", file = open("parameter.txt", "a"))
	print(repr(newFeatureList), file = open("parameter.txt", "a"))
	print("newFeatureNum", file = open("parameter.txt", "a"))
	print(newFeatureNum, file = open("parameter.txt", "a"))
	

	OneHotList = [1, 2, 3, 5, 6, 7, 8, 9, 10]
	X_test, newFeatureNum, newFeatureList = OneHot(X_test, OneHotList, newFeatureList, False)
	X_test = np.c_[X_test, np.ones(X_test.shape[0])]
	minData, maxData, X_test[0:, 0:(-1 - newFeatureNum)] = FeatureScaling(minData, maxData, X_test[0:, 0:(-1 - newFeatureNum)], False)
	FillData(X_test, w, sys.argv[4])

if __name__ == "__main__":
	main()