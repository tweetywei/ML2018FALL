 #-*- coding=utf-8 -*-
import pandas as pd
import sys
import jieba
import numpy as np
import random
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import load_model

 #-*- coding=utf-8 -*-
import pandas as pd
import sys
import jieba
import numpy as np
import random
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import load_model

def ReadFile(testx_filename, dic_file):
	df = pd.read_csv(testx_filename,encoding='utf-8',  sep='\n')
	df = df.values.flatten().tolist()
	df = [li.split(',', 1)[1] for li in df]
	df = np.asarray(df)
	jieba.set_dictionary(dic_file)
	cutString = []
	for i in range(df.shape[0]):
		result=[]
		seg_list = jieba.cut(df[i])
		for w in seg_list:
			result.append(w)
		cutString.append(result)
	return cutString

def wordToVector(stringList):
	model = Word2Vec(stringList, size=32, window=5, min_count=5, workers=16, sg=0, negative=5)
	###check more similar
	#similar_df = mostSimilar(model, stringList)
	#similar_df.to_csv('similar.csv')
	return model

def word2idx(word, model):
	if(word in model.wv.vocab):
		return model.wv.vocab[word].index
	else:
		return 0
def idx2word(idx, model):
	return model.wv.index2word[idx]
def main():
	cutStringList = ReadFile(sys.argv[1], sys.argv[2])
	#word2VecModel = wordToVector(cutStringList)
	#word2VecModel.save("word2vec.model")
	word2VecModel = Word2Vec.load("word2vec2.model")
	MAX_NB_WORDS = len(word2VecModel.wv.vocab)
	MAX_SEQUENCE_LENGTH = 200
	#textVectors = [[word2VecModel[word] for word in sentence if word in word2VecModel] for sentence in cutStringList]
	#print(textVectors, file = open("drive/ML/hw4/outputVectors.txt", "a"))
	sequence = [[(word2idx(word, word2VecModel)+1) for word in sentence] for sentence in cutStringList]
	test_x = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")

	WV_DIM = 100
	nb_words = MAX_NB_WORDS
	# we initialize the matrix with random numbers
	# embedding_matrix = np.zeros((len(word2VecModel.wv.vocab), WV_DIM))
	#for i in range(len(word2VecModel.wv.vocab)):
		#embedding_vector = word2VecModel.wv[word2VecModel.wv.index2word[i]]
		#if embedding_vector is not None:
			#embedding_matrix[i] = embedding_vector

	model = load_model('rnnmodel2_2.h5')
	print(model.summary())
	prediction=model.predict_classes(test_x, batch_size = 32)
	print(prediction)
	prediction = prediction.reshape(prediction.shape[0]).astype(np.int32)
	row = [str(x) for x in range(prediction.shape[0])]
	result_array = np.column_stack((row, prediction))
	df = pd.DataFrame(result_array,columns=['id','label'])
	df.to_csv(sys.argv[3], index=False)

if __name__ == "__main__":
	main()


