# Necessary imports
import os
import argparse
import utils
import numpy as np
from sklearn import cross_validation
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

# Split data into training, normalization and test set
def split_into_sets(X, y):
	X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size=0.1)
	#print(X.shape)
	#print(y.shape)
	print(X_train.shape)
	#print(y_train.shape)
	#print(X_val.shape)
	#print(y_val.shape)
	return X_train, y_train, X_val, y_val

def get_next_batch(X, y, batch_size):
	while(True):
		batch_mask = np.random.choice(len(X), size=batch_size, replace=False)
		X_train = X[batch_mask,:,:]
		y_train = y[batch_mask]
		yield X_train, y_train

def train(model, X_train, y_train, X_val, y_val, batch_size, nb_epoch):
	model.fit_generator(
		generator=get_next_batch(X_train, y_train, batch_size),
		samples_per_epoch=batch_size,
		nb_epoch=nb_epoch,
		validation_data=get_next_batch(X_val, y_val, batch_size),
		nb_val_samples=len(X_val)
	)
	return model

# Use the model defined in Commai following repo:
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def get_model(nvidia=False):

	model = Sequential()
	adam = Adam(lr=0.001)

	# Nvidia model
	if(nvidia):
		model.add(Convolution2D(66, 3, 3, input_shape=(66, 200, 3), border_mode="same", activation='relu'))
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

	model.add(Convolution2D(32, 3, 3, input_shape=(16, 32, 3), border_mode="same", activation='relu'))
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

	# Use the Adam optimizer to optimize the mean squared error
	model.compile(optimizer=adam, loss="mse")	

	return model

# Load Data
data = utils.load_data()
# Split to have left, right and center images with the corresponding steering angle
data = utils.split_input(data)
# Genererate more data
data = utils.flip_center_images(data)
# Randomly Shuffle data
np.random.shuffle(data)
# Pre-Process data
test_images = utils.process_images(data, nvidia=True)
# Split data into training, test and validation set
y_data = np.array(data[:,1], dtype=float)
X_train, y_train, X_val, y_val = split_into_sets(test_images, y_data)
# Get the model
model = get_model(nvidia=True)
# Train the model
trained_model = train(model, X_train, y_train, X_val, y_val, 1024, 20) # To do: handle the case where the batch_size (sample_per_epochs) is not a factor of len(data)
# Save it
utils.save_model(trained_model)