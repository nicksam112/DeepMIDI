from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv1D, Flatten, Embedding, Dropout, MaxPooling1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import random
from time import time

from mido import Message, MidiFile, MidiTrack
from operator import itemgetter

#command line args
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--run", help="Run Model", action="store_true", default=False)
parse.add_argument("--train", help="Train model", action="store_true", default=False)
parse.add_argument("--conv", help="Use 1D Convolution Model instead of LSTM", action="store_true", default=False)
parse.add_argument("--load_weights", help="Weights to load", type=str, default="midi_weights.h5")
parse.add_argument("--save_weights", help="Save weights as", type=str, default="midi_weights.h5")
parse.add_argument("--midi_data_path", help="Path to midi data", type=str, default="data/")
parse.add_argument("--save_song_path", help="Path to save produced song", type=str, default="./")
args = parse.parse_args()

def collect(l, index):
   return map(itemgetter(index), l)

batch_size = 32  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.

lengthSample = 16 #length of sample


data = np.load(args.midi_data_path + 'midi_songs.npy')
dataInd = np.load(args.midi_data_path + 'midi_notes.npy')

def noteToInd(note):
	return np.where((dataInd == note).all(axis=1))[0][0]
	return -1
def indToNote(ind):
	return dataInd[ind]

def music_gen():
	while True:
		batch_features = np.zeros((batch_size, lengthSample))
		batch_labels = np.zeros((batch_size, dataInd.shape[0]))
		for i in range(batch_size):
			song = random.randrange(data.shape[0])
			while len(data[song]) == 0:
				song = random.randrange(data.shape[0])

			ind = random.randrange(len(data[song])-lengthSample)
			for x in range(lengthSample):
				batch_features[i][x] = data[song][ind+x]
			batch_labels[i][data[song][ind+lengthSample]] = 1

		yield batch_features, batch_labels


def createConvModel():
	inp = Input(shape=(lengthSample,))
	x = Embedding(dataInd.shape[0], 128, input_length=lengthSample)(inp)
	x = Conv1D(128,kernel_size=3,activation='elu')(x)
	x = Conv1D(128,kernel_size=3,activation='elu')(x)
	x = Conv1D(128,kernel_size=3,activation='elu')(x)
	x = Flatten()(x)
	out = Dense(dataInd.shape[0], activation='softmax')(x)

	model = Model([inp],[out])
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def createLSTMModel():
	inp = Input(shape=(lengthSample,))
	x = Embedding(dataInd.shape[0], 128, input_length=lengthSample)(inp)
	x = LSTM(128, return_sequences=True)(x)
	x = LSTM(128)(x)
	out = Dense(dataInd.shape[0], activation='softmax')(x)

	model = Model([inp],[out])
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def trainModel():
	if args.conv:
		model = createConvModel()
	else:
		model = createLSTMModel()
	print(model.summary())
	checkpoint = ModelCheckpoint(args.save_weights, monitor='loss', verbose=0, save_best_only=False, mode='max')
	tensorboard = TensorBoard("logs/{}".format(time()))
	callbacks_list = [checkpoint, tensorboard]
	model.fit_generator(music_gen(), steps_per_epoch=128, epochs=256, callbacks=callbacks_list)
	model.save(args.save_weights)

def runModel():
	if not args.train:
		if args.conv:
			model = createConvModel()
		else:
			model = createLSTMModel()
		model.load_weights(args.load_weights)
	
	#start with some random notes to generate off of
	ls = []
	for x in range(lengthSample):
		ls.append(random.randint(0,dataInd.shape[0]))

	#set up midi file
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	track.append(Message('program_change', program=12))

	time = 0
	#run for 10 seconds
	while time < 10.:
		print(str(ls[-5:]) + " " + str(time))

		#predict
		res = model.predict(np.expand_dims(np.array(ls[-lengthSample:]), axis=0))[0]
		listAdd = res

		#convert to note
		res = indToNote(np.argmax(np.array(res)))
		time += res[2]
		res[0] = np.rint(res[0])
		res[1] = np.rint(res[1]*127)
		res[2] = np.rint(res[2]*880)
		
		if(res[0] > 0):
			track.append(Message('note_on', note=int(res[1]), time=int(res[2])))
		else:
			track.append(Message('note_off', note=int(res[1]), time=int(res[2])))
		
		#convert back to a format the mdoel can read
		res[1] = res[1]/127.
		res[2] = res[2]/880.

		#append the note to the running list of notes
		ls.append(np.argmax(np.array(listAdd)))
		
		#ocasionally add random notes
		#helps avoid loops
		if time%2. <= 0.05:

			addInd = random.randint(0,len(data[data.shape[0]-1]))
			add = data[data.shape[0]-1][addInd:addInd+5]
			# for x in range(5):
			# 	add[x] = noteToInd(add[x])
			ls += add
			#ls.append(random.randint(0,dataInd.shape[0]))
	mid.save(args.save_song_path + 'new_song.mid')

if args.train:
	trainModel()
if args.run:
	runModel()