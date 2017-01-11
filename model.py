# Necessary imports
import os
import argparse
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

# Split data into training, normalization and test set
def split_into_sets(X, y):
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
	#print(X.shape)
	#print(y.shape)
	print(X_train.shape)
	#print(y_train.shape)
	#print(X_val.shape)
	#print(y_val.shape)
	return X_train, y_train, X_val, y_val

# Here, add random brightness and scale down
# Generate image to feed the model
def get_next_batch(X_train, y_train, batch_size):

	# Will contains images and corresponding angle
	X_batch = np.zeros((batch_size, 40, 160, 3))
	y_batch = np.zeros(batch_size)

	while True:
		for i in range(0, batch_size):
			# Take a random image and angle from X_train and y_train
			random_index = np.random.randint(len(X_train))
			X_batch[i], y_batch[i] = utils.process_image(X_train[random_index])
		yield X_batch, y_batch	

def train(model, X_train, y_train, X_val, y_val, batch_size, nb_epoch):
	model.fit_generator(
		generator=get_next_batch(X_train, y_train, batch_size),
		samples_per_epoch=12000,
		nb_epoch=nb_epoch,
		validation_data=get_next_batch(X_val, y_val, batch_size),
		nb_val_samples=len(X_val)
	)
	return model

# Use the model defined in Commai following repo:

def get_model():

	model = Sequential()
	adam = Adam(lr=0.001)

	''' Nvidia
	model.add(Convolution2D(66, 3, 3, input_shape=(160, 320, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(24, 3, 3, subsample=(5, 5), border_mode="same", activation='relu'))
	model.add(Convolution2D(36, 3, 3, subsample=(5, 5), border_mode="same", activation='relu'))
	model.add(Convolution2D(48, 3, 3, subsample=(5, 5), border_mode="same", activation='relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	model.compile(optimizer=adam, loss="mse")
	return model
	'''
	'''		
	model.add(Convolution2D(32, 3, 3, input_shape=(16, 32, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(32, 3, 3, subsample=(1, 1), border_mode="same", activation='relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same", activation='relu'))
	model.add(Convolution2D(128, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(128, 3, 3, subsample=(1, 1), border_mode="same", activation='relu'))
	model.add(Convolution2D(256, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(256, 3, 3, subsample=(1, 1), border_mode="same", activation='relu'))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(1))
	'''

	'''
	model.add(Convolution2D(32, 3, 3, input_shape=(32, 64, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(64, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(128, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(256, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(1))
	'''

	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(40, 160, 3), output_shape=(40, 160, 3)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	# Use the Adam optimizer to optimize the mean squared error
	model.compile(optimizer=adam, loss="mse")	

	return model

# Load Data
data = utils.load_data()
# Split to have left, right and center images with the corresponding steering angle
data = utils.split_input(data)
data = utils.flip_images(data)
# Randomly Shuffle data
np.random.shuffle(data)
# Split data into training, test and validation set
y_data = np.array(data[:,1], dtype=float)
X_train, y_train, X_val, y_val = split_into_sets(data, y_data)
# Get the model
model = get_model()
# Train the model
trained_model = train(model, X_train, y_train, X_val, y_val, 6000, 5) # To do: handle the case where the batch_size (sample_per_epochs) is not a factor of len(data)
# Save it
utils.save_model(trained_model)
