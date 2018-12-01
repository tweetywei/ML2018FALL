import numpy as np
import pandas as pd
import keras
import sys
from keras.models import load_model
import os

def ReadFile(filename):
	df_test = pd.read_csv(filename)
	df_test["feature"] = df_test["feature"].str.split(' ')

	x_test = df_test.loc[0:, "feature"].values#.astype(int)
	x_test = np.array([np.array(x) for x in x_test])
	x_test = x_test.astype(int).reshape((x_test.shape[0], 48, 48))
	return x_test


def main():
	if (len(sys.argv) < 3):
		print("Please specify: test file, output file")
		sys.exit(1)
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	x_test = ReadFile(sys.argv[1])
	x_test = (x_test.reshape(x_test.shape[0],48,48,1).astype('float32')) / 255

	model = load_model('EnsembleModelfinal.h5')
	prediction=model.predict(x_test, batch_size = 150)
	prediction = np.argmax(prediction, axis=1)
	prediction = np.expand_dims(prediction, axis=1)
	#print(prediction)
	prediction = prediction.reshape(prediction.shape[0]).astype(np.int32)
	row = [str(x) for x in range(prediction.shape[0])]
	result_array = np.column_stack((row, prediction))
	df = pd.DataFrame(result_array,columns=['id','label'])
	df.to_csv(sys.argv[2], index=False)


if __name__ == "__main__":
	main()


