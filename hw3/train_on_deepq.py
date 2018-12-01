import numpy as np
import pandas as pd
import keras
import sys
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,AveragePooling2D, LeakyReLU, Input
from keras import regularizers
from keras.models import Model
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import os
import io

def ReadFile(filename, isValid, valid_number):
	df = pd.read_csv(filename)
	df["feature"] = df["feature"].str.split(' ')
	#print(df)
	x_train = df.loc[0:, "feature"].values#.astype(int)
	x_train = np.array([np.array(x) for x in x_train])
	x_train = x_train.astype(int).reshape((x_train.shape[0], 48, 48))
	#x_train = x_train.reshape((x_train.shape[0], 48 ,48))
	y_train = df.loc[0:, "label"].values
	#print(x_train, y_train)
	if isValid == 1:
		return x_train[:-1 * valid_number, :, :], y_train[:-1 * valid_number], x_train[-1 * valid_number:,:,:], y_train[-1 * valid_number:]
	else:
		return x_train, y_train, x_train[-1 * valid_number:,:,:], y_train[-1 * valid_number:]

def train_model1(x_train, y_trainOneHot, x_valid, y_validOneHot, dataGenerator, isValid, modelNum):
	######create model######
	model = Sequential()
	model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',input_shape=(48,48,1)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))
	
	model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))
	
	model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())
	model.add(Dense(256,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(7,activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
	print("===================training model1===========")
	print(model.summary())
	if isValid == 0:
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 0, epochs=5)
		model.save('model%d100.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 100, epochs=200)
		model.save('model%d200.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 200, epochs=300)
		model.save('model%d300.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 300, epochs=350)
		model.save('model%d350.h5' % modelNum)
	else:
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 0, epochs=100, validation_data=(x_valid, y_validOneHot))
		model.save('model%d100.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 100, epochs=200, validation_data=(x_valid, y_validOneHot))
		model.save('model%d200.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 200, epochs=300, validation_data=(x_valid, y_validOneHot))
		model.save('model%d300.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 300, epochs=350, validation_data=(x_valid, y_validOneHot))
		model.save('model%d350.h5' % modelNum)
	print("=================finish training model1==============")
	return model

def train_model2(x_train, y_trainOneHot, x_valid, y_validOneHot, dataGenerator, isValid, modelNum):
	######create model######
	model = Sequential()
	model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',input_shape=(48,48,1)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	
	model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))
	
	model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dense(1024,activation='relu'))
	model.add(Dropout(0.5))
	model.add(BatchNormalization())
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(7,activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
	print("===================training model2===========")
	print(model.summary())
	if isValid == 0:
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 0, epochs=5)
		model.save('model%d100.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 100, epochs=200)
		model.save('model%d200.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 200, epochs=300)
		model.save('model%d300.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 300, epochs=350)
		model.save('model%d350.h5' % modelNum)
	else:
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 0, epochs=100, validation_data=(x_valid, y_validOneHot))
		model.save('model%d100.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 100, epochs=200, validation_data=(x_valid, y_validOneHot))
		model.save('model%d200.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 200, epochs=300, validation_data=(x_valid, y_validOneHot))
		model.save('model%d300.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 300, epochs=350, validation_data=(x_valid, y_validOneHot))
		model.save('model%d350.h5' % modelNum)
	print("=================finish training model2==============")
	return model

def train_model3(x_train, y_trainOneHot, x_valid, y_validOneHot, dataGenerator, isValid, modelNum):
	######create model######
	model = Sequential()
	model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',input_shape=(48,48,1)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))
	
	model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))
	
	model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(256,activation='relu'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha = 0.001))
	model.add(Dropout(0.4))
	model.add(Dense(512,activation='relu'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha = 0.001))
	model.add(Dropout(0.4))
	model.add(Dense(7,activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
	print("===================training model3===========")
	print(model.summary())
	if isValid == 0:
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 0, epochs=5)
		model.save('model%d100.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 100, epochs=200)
		model.save('model%d200.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 200, epochs=300)
		model.save('model%d300.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 300, epochs=350)
		model.save('model%d350.h5' % modelNum)
	else:
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 0, epochs=100, validation_data=(x_valid, y_validOneHot))
		model.save('model%d100.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 100, epochs=200, validation_data=(x_valid, y_validOneHot))
		model.save('model%d200.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 200, epochs=300, validation_data=(x_valid, y_validOneHot))
		model.save('model%d300.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 300, epochs=350, validation_data=(x_valid, y_validOneHot))
		model.save('model%d350.h5' % modelNum)
	print("=================finish training model3==============")
	return model

def train_model4(x_train, y_trainOneHot, x_valid, y_validOneHot, dataGenerator, isValid, modelNum):
	######create model######
	model = Sequential()
	model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',input_shape=(48,48,1)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.1))
	
	model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',))
	model.add(BatchNormalization())
	#model.add(MaxPooling2D(pool_size=(3, 3), strides = 1))
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.1))
	
	model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	#model.add(MaxPooling2D(pool_size=(3, 3), strides = 1))
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=512,kernel_size=(3,3),padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(1024,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(256,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(7,activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
	print("===================training model4===========")
	print(model.summary())
	if isValid == 0:
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 0, epochs=5)
		model.save('model%d100.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 100, epochs=200)
		model.save('model%d200.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 200, epochs=300)
		model.save('model%d300.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 300, epochs=350)
		model.save('model%d350.h5' % modelNum)

	else:
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 0, epochs=100, validation_data=(x_valid, y_validOneHot))
		model.save('model%d100.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 100, epochs=200, validation_data=(x_valid, y_validOneHot))
		model.save('model%d200.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 200, epochs=300, validation_data=(x_valid, y_validOneHot))
		model.save('model%d300.h5' % modelNum)
		model.fit_generator(dataGenerator.flow(x_train, y_trainOneHot, batch_size=150), steps_per_epoch=x_train.shape[0] // 150, initial_epoch = 300, epochs=350, validation_data=(x_valid, y_validOneHot))
		model.save('model%d350.h5' % modelNum)
	print("=================finish training model4==============")
	return model

def ensembleModels(models, modelInput):
	yModels=[model(modelInput) for model in models]
	yAvg=layers.average(yModels)
	modelEns = Model(inputs=modelInput, outputs=yAvg, name='ensemble')  
	return modelEns

def main():
	if (len(sys.argv) < 2):
		print("Please specify: train file")
		sys.exit(1)
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	###Parameter###
	isValid = 0
	valid_number = 3000

	x_train, y_train, x_valid, y_valid = ReadFile(sys.argv[1],isValid, valid_number)
	x_train = (x_train.reshape(x_train.shape[0],48,48,1).astype('float32')) / 255
	x_valid = (x_valid.reshape(x_valid.shape[0],48,48,1).astype('float32')) / 255
	y_trainOneHot = np_utils.to_categorical(y_train)	
	y_validOneHot = np_utils.to_categorical(y_valid)

	###DataGenerate
	dataGenerator = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=[1-0.2, 1+0.2],horizontal_flip=True)
	###Train
	model1 = train_model1(x_train, y_trainOneHot, x_valid, y_validOneHot, dataGenerator, isValid, 1)
	model2 = train_model2(x_train, y_trainOneHot, x_valid, y_validOneHot, dataGenerator, isValid, 2)
	model3 = train_model3(x_train, y_trainOneHot, x_valid, y_validOneHot, dataGenerator, isValid, 3)
	model4 = train_model4(x_train, y_trainOneHot, x_valid, y_validOneHot, dataGenerator, isValid, 4)	
	models = [model1, model2, model3, model4]
	modelInput = Input(shape=models[0].input_shape[1:])
	modelEns = ensembleModels(models, modelInput)
	#print(model.summary)
	modelEns.save('EnsembleModelfinal.h5')

if __name__ == "__main__":
	main()


###Reference: https://medium.com/randomai/ensemble-and-store-models-in-keras-2-x-b881a6d7693f