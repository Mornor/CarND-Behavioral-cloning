# Necessary imports
import os
import argparse
import json
import utils
import numpy as np
from sklearn import cross_validation
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D

# Flip image and inverse the steering angle
def generate_more_data(img, steering_angle):
	print(img.size[0])
	print(img.size[1])

# Split data into training, normalization and test set
def split_input(X, y):
	X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size=0.1)
	#print(X_train.shape)
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

# Save the model under ./model.json, as well as the weights under ./model.h5
def save_model(model):
	with open('./model.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)
	with open('./model.h5', 'w') as outfile:
		model.save_weights('model.h5')

def train(model, X_train, y_train, X_val, y_val, batch_size, nb_epoch):
	model.fit_generator(
		get_next_batch(X_train, y_train, batch_size),
		len(X_train),
		nb_epoch,
		validation_data=get_next_batch(X_val, y_val, batch_size),
		nb_val_samples=len(X_val)
	)
	return model


# Use the model defined in Commai following repo:
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# 10 layers:
# - 1 Normalization Layer
# - 5 Convolutional Layers
# - 3 Fully connected Layers
# - 1 Flatten Layer
def get_model():

	model = Sequential()

	# 1st Layer: Normalized Layer
	model.add(Convolution2D(32, 3, 3, input_shape=(16, 32, 3), border_mode="same", activation='relu'))

	# Convolutional Layers, stride of 3*3 everywhere
	model.add(Convolution2D(64, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(128, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(256, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Dropout(0.5))
	#model.add(Convolution2D(512, 3, 3, subsample=(3, 3), border_mode="same", activation='relu'))

	# 7th Layer: Flatten Layer
	model.add(Flatten())

	# Fully Connected Layers
	model.add(Dense(1164))
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))

	# Final layer
	model.add(Dense(1))

	# Use the Adam optimizer to optimize the mean squared error
	model.compile(optimizer="adam", loss="mse", lr=0.0001)	

	return model

# Load Data
data = utils.load_data()
# Pre-Process data
center_images, left_images, right_images = utils.process_image(data)
# Split data into training, test and validation set
y_data = np.array(data[:,3], dtype=float)
X_train, y_train, X_val, y_val = split_input(center_images, y_data)
# Get the model
model = get_model()
# Train the model
trained_model = train(model, X_train, y_train, X_val, y_val, 452, 2) # To do: handle the case where the batch_size (sample_per_epochs) is not a factor of len(data)
# Save it
save_model(trained_model)
