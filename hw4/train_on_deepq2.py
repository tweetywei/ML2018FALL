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
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding

def ReadFile(trainx_filename, trainy_filename, test_filename, isValid, validSize, dic_file):
	df = pd.read_csv(trainx_filename,encoding='utf-8',  sep='\n')
	dfy = pd.read_csv(trainy_filename)
	dfy = dfy.loc[0:, "label"].values
	dftest = pd.read_csv(test_filename,encoding='utf-8',  sep='\n')
	#df.to_csv('output2.csv')
	df = df.values.flatten().tolist()
	df = [li.split(',', 1)[1] for li in df]
	df = np.asarray(df)
	dftest = dftest.values.flatten().tolist()
	dftest = [li.split(',', 1)[1] for li in dftest]
	dftest = np.asarray(dftest)
	#print(df, file = open("outputArray.txt", "a"))
	
	jieba.set_dictionary(dic_file)
	cutString = []
	allString = []
	for i in range(df.shape[0]):
		result=[]
		seg_list = jieba.cut(df[i])
		for w in seg_list:
			result.append(w)
		cutString.append(result)
		allString.append(result)

	for l in range(dftest.shape[0]):
		result=[]
		seg_list = jieba.cut(dftest[l])
		for w in seg_list:
			result.append(w)
		allString.append(result)

	if isValid == 0:
		return cutString, dfy, cutString[:1],  dfy[:1], allString#[:1000]
	else:
		return cutString[validSize:], dfy[validSize:], cutString[:validSize], dfy[:validSize], allString


def mostSimilar(w2v_model, stringList, topn=10):
	words = []
	for i in range(10):
		random_suite = random.choice(stringList)
		random_card = random.choice(random_suite)
		words.append(random_card)
	similar_df = pd.DataFrame()
	for word in words:
		try:
			similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
			similar_df = pd.concat([similar_df, similar_words], axis=1)
		except:
			print(word, "not found in Word2Vec model!")
	return similar_df
		
def wordToVector(stringList, dim):
	model = Word2Vec(stringList, size=dim, window=5, min_count=5, workers=16, sg=0, negative=5)
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
	###parameter###
	isValid = 0
	validSize = 24000
	WV_DIM = 100
	###############

	cutStringList, train_y, cutStringListValid, valid_y, allStringList = ReadFile(sys.argv[1], sys.argv[2], sys.argv[3], isValid, validSize, sys.argv[4])
	word2VecModel = wordToVector(allStringList, WV_DIM)
	word2VecModel.save("word2vec2.model")
	#word2VecModel = Word2Vec.load("word2vec.model")
	MAX_NB_WORDS = len(word2VecModel.wv.vocab)
	MAX_SEQUENCE_LENGTH = 200
	#textVectors = [[word2VecModel[word] for word in sentence if word in word2VecModel] for sentence in cutStringList]
	#print(textVectors, file = open("drive/ML/hw4/outputVectors.txt", "a"))
	sequence = [[(word2idx(word, word2VecModel)+1) for word in sentence] for sentence in cutStringList]
	validSequence = [[(word2idx(word, word2VecModel)+1) for word in sentence] for sentence in cutStringListValid]
	train_x = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
	valid_x = pad_sequences(validSequence, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
	pretrained_weights = word2VecModel.wv.syn0

	nb_words = MAX_NB_WORDS + 1
	# we initialize the matrix with random numbers
	embedding_matrix = np.zeros((nb_words, WV_DIM))
	for i in range(nb_words - 1):
		embedding_vector = word2VecModel.wv[word2VecModel.wv.index2word[i]]
		if embedding_vector is not None:
			embedding_matrix[i+1] = embedding_vector
	#print(embedding_matrix.shape)
	#print(embedding_matrix)   
	model = Sequential()
	model.add(Embedding(nb_words, WV_DIM, input_length=MAX_SEQUENCE_LENGTH, weights = [embedding_matrix]))
	#model.add(Dropout(0.2))
	#model.add(Conv1D(64, 5, activation='relu'))
	#model.add(MaxPooling1D(pool_size=4))
	model.add(LSTM(100))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	###use rnnmodel2!!
	if isValid == 0:
		model.fit(train_x, train_y, initial_epoch = 0, epochs=1, batch_size=32)	
		model.save('rnnmodel2_1.h5')
		model.fit(train_x, train_y, initial_epoch = 1, epochs=2, batch_size=32)	
		model.save('rnnmodel2_2.h5')
		model.fit(train_x, train_y, initial_epoch = 2, epochs=3, batch_size=32)	
		model.save('rnnmodel2_3.h5')
		model.fit(train_x, train_y, initial_epoch = 3, epochs=5, batch_size=32)
		model.save('rnnmodel2_5.h5')
		model.fit(train_x, train_y, initial_epoch = 5, epochs=6, batch_size=32)	
		model.save('rnnmodel2_6.h5')
		model.fit(train_x, train_y, initial_epoch = 6, epochs=7, batch_size=32)	
		model.save('rnnmodel2_7.h5')
	if isValid == 1:
		model.fit(train_x, train_y, validation_data = (valid_x, valid_y), epochs=10, batch_size=32)

if __name__ == "__main__":
	main()