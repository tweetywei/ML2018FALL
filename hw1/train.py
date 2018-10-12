import numpy as np
import pandas as pd 
import sys
def ReadArgumentFile(filename):
	itemList = []
	dayList = []
	index = 0
	maxDay = 0
	file = open(filename, "r")
	arguments = file.read().splitlines()
	for l in range(len(arguments)):
		itemList.append(arguments[l].split()[0])
		dayList.append(int(arguments[l].split()[1]))
		if arguments[l].split()[0] == "PM2.5":
			index = l
		if int(arguments[l].split()[1]) > maxDay:
			maxDay = int(arguments[l].split()[1])
	#print(itemList, dayList)
	return itemList, dayList, index, maxDay

def FixData(monthData):
	for r in range(monthData.shape[0]):
		for c in range(0, monthData.shape[1]):
			if monthData[r][c] == 'NR':
				monthData[r][c] = 0
			else:
				monthData[r][c] = float(monthData[r][c])
			if c > 0 and (monthData[r][c] <= 2 or monthData[r][c] > 120):
				monthData[r][c] = monthData[r][c - 1]
		for c in range(monthData.shape[1] - 2, -1, -1):
			if monthData[r][c] <= 2 or monthData[r][c] > 120:
				monthData[r][c] = monthData[r][c + 1]
	return monthData

def ChooseTraningData(monthData, daysList, pm25Index, validCut, maxDay):
	yt = []
	dt = []
	yv = []
	dv = []
	for i in range(monthData.shape[1] - 1, maxDay - 1, -1):
		oneData = []
		yData = 0
		isValid = 1
		for r in range(len(daysList)):
			#check pm2.5 data reasonable
			oneData.extend(monthData[r][i - daysList[r]:i].tolist())
			if r == pm25Index:
				pmData = monthData[r][i - daysList[r]:i + 1].tolist()
				pmList = [float(p) for p in pmData]
				#print("in pm25Index")
				if any(p <= 0 or p >= 150 for p in pmList):
					#print("invalid!!!")
					isValid = 0
		if isValid == 0:
			continue
		yData = monthData[pm25Index][i]
		yData = float(yData)
		oneData = ['0' if x =='NR' else x for x in oneData]
		oneData = [float(x) for x in oneData]
		if any (s < 0 for s in oneData):
			continue 
		oneData.append(1)
		if i < 600:
			dt.append(oneData)
			yt.append(yData)
		else:
			dv.append(oneData)
			yv.append(yData)
	return yt, dt, yv, dv

def ReadTrainData(trainDataFile, itemsList, daysList, pm25Index, maxDay):
	df = pd.read_csv('train.csv', encoding = "gb18030")
	df.columns = np.arange(0,len(df.columns))
	argDf = df[df[2].isin(itemsList)]
	argDf.index = range(argDf.shape[0])
	perMonthArrayList = []
	argLength = len(itemsList)
	yTraningList = []
	dataTraningList = []
	yValidList = []
	dataValidList = []
	validCut = 4
	for i in range(12):
		perMonthArrayList.append(argDf.loc[i * argLength * 20:(i + 1) * argLength * 20 - 1, 3:].values)
	for a in perMonthArrayList:
		splitData = np.split(a, 20, axis = 0)
		monthData = np.hstack(splitData)
		#print(monthData)
		#monthData = FixData(monthData)
		yt, dt, yv, dv = ChooseTraningData(monthData, daysList, pm25Index, validCut, maxDay)
		yTraningList.extend(yt)
		dataTraningList.extend(dt)
		yValidList.extend(yv)
		dataValidList.extend(dv)
	return np.array(yTraningList), np.array(dataTraningList), np.array(yValidList), np.array(dataValidList)

def TrainData(w, yArray, dataArray, eta, iteration, lamda):
	batchSize = dataArray.shape[0]
	learningW = np.zeros(w.shape)
	rmseList = []
	for i in range(iteration):
		diff = yArray - np.dot(dataArray , w)
		gradW = -np.dot(dataArray.transpose(), diff) / batchSize	
		regularVector = lamda * w / batchSize
		regularVector[w.shape[0] - 1] = 0
		gradW += regularVector
		learningW += gradW ** 2
		if np.linalg.norm(learningW) > 0:
			w = w - eta / np.sqrt(learningW) * gradW
		else:
			w = w - eta * gradW
		rmseList.append(GetRmse(yArray, dataArray, w))
	return w

def GetRmse(yArray, dataArray, w):
	diff = yArray - np.dot(dataArray , w)
	return np.sqrt((diff**2).mean())

def CastType(Array):
	Array[Array == 'NR'] = '0'
	Array = Array.astype(float)
	return Array

def FeatureScaling(minData, maxData, dataArray, train=False):    
	if train:
		minData = np.min(dataArray, axis=0)
		maxData = np.max(dataArray, axis=0)
	return minData, maxData, ((dataArray - minData) / (maxData - minData))

def FeatureScaling2(meanData, stdData, dataArray, train=False):
	if train:
		meanData = np.mean(dataArray, axis=0)
		stdData = np.std(dataArray, axis=0)
	return meanData, stdData, ((dataArray - meanData) / stdData)

def main():
	#check the argument file
	if (len(sys.argv) < 6):
		print("Please specify 1: the traning data file 2: which file save the arguments 3: eta 4: iteration 5: regularization parameter")
		sys.exit(1)

	#read argument file
	items, days, pm25Index, maxDay = ReadArgumentFile(sys.argv[2])

	#dataArray: one row specify one data, ex: 3 hr PM2.5 + 2 hr PM10 + 1
	#dataArray: shape[0]: number of data, shape[1]: number of parameters
	yArray, dataArray, validyArray, validDataArray = ReadTrainData(sys.argv[1], items, days, pm25Index, maxDay)
	#print(yArray, yArray.shape, dataArray, dataArray.shape)
	meanData, stdData, dataArray[:,0:-1] = FeatureScaling2(0, 0, dataArray[:,0:-1], True)
	#meanData, stdData, validDataArray[:,0:-1] = FeatureScaling2(meanData, stdData, validDataArray[:,0:-1], False)
	w = np.zeros(dataArray.shape[1])
	#yArray, dataArray, eta, iteration, lamda
	w = TrainData(w, yArray, dataArray, float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
	trainRmse = GetRmse(yArray, dataArray, w)
	#validRmse = GetRmse(validyArray, validDataArray, w)
	print(repr(w), file = open("output.txt", "a"))
	print("\n", file = open("output.txt", "a"))
	print("training rmse = " + str(trainRmse), file = open("output.txt", "a"))
	print("\n", file = open("output.txt", "a"))
	#print("validating rmse = " + str(validRmse), file = open("output.txt", "a"))
	print("\n", file = open("output.txt", "a"))
	#print("mean = " , repr(meanData) ," ,std = " , repr(stdData), file = open("output.txt", "a"))


if __name__ == "__main__":
	main()

