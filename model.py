# Necessary imports
import os
import argparse
import json
import csv
import numpy as np
import matplotlib.image as mpimg
from sklearn import cross_validation
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

# Usefull constants
DATA_PATH = "driving_log.csv"

# Load the CSV names into memory as tuples
# Center image: np.array(csv[:,0])
# Left image: np.array(csv[:,1])
# Right image: np.array(csv[:,2])
# Sterring angle: np.array(csv[:,3], dtype=float)
# (160, 320, 3)
def load_data():
	with open(DATA_PATH, 'r') as file:
		reader = csv.reader(file)
		file.readline() # Skip headers
		data = np.array([row for row in reader])
	return data

# Process a single image
# Normalize data from 0-255 to -1 - 1
# TODO size down image by 2, reducing the number of pixels by 4
def process_img(img):
	return img/127.5 - 1.

# Process images from input
def process_image(data):
	# Load the images into np array
	center_images = np.zeros((len(data), 160, 320, 3), dtype=float)
	left_images = np.zeros((len(data), 160, 320, 3), dtype=float)
	right_images = np.zeros((len(data), 160, 320, 3), dtype=float)
	for i in range(8000, len(data)):
		center_images[i] = process_img(mpimg.imread(data[:,0][i].strip()))
		left_images[i] = process_img(mpimg.imread(data[:,1][i].strip()))
		right_images[i] = process_img(mpimg.imread(data[:,2][i].strip()))
	
	print(center_images[0].shape)
	print(left_images[0].shape)
	print(right_images[0].shape)
	#test_images = np.array([mpimg.imread(path_folder + "/" + file) for file in filenames])

# Split data into training, normalization and test set
def split_input(X, y):
	X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size=0.1)
	print(X_train.shape)
	print(y_train.shape)
	print(X_val.shape)
	print(y_val.shape)
	return X_train, y_train, X_val, y_val

def get_next_batch(x, y, batch_size):
    batch_mask = np.random.choice(len(x), size=batch_size, replace=False)
    x_train = x[batch_mask,:,:]
    y_train = y[batch_mask,:] 
    return x_train, y_train

# Save the model under ./model.json, as well as the weights under ./model.h5
def save_model(model):
	with open('./model.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)
	with open('./model.h5', 'w') as outfile:
		model.save_weights('model.h5')

def train():
 	model.fit_generator(
 		gen_batches(X_train, y_train, FLAGS.batch_size),
        len(X_train),
        FLAGS.num_epochs,
        validation_data=gen_batches(X_val, y_val, FLAGS.batch_size),
        nb_val_samples=len(X_val)
    )


# Use the model defined in Commai following repo:
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# 10 layers:
# - 1 Normalization Layer
# - 5 Convolutional Layers
# - 3 Fully connected Layers
# - 1 Flatten Layer
def get_model():

	model = Sequential()
	inp = Input(shape=(40,160,3))

	# 1st Layer: normalized Layer
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='relu'))

	# Convolutional Layers
	model.add(Convolution2D(32, 5, 5, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(32, 5, 5, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(32, 5, 5, subsample=(3, 3), border_mode="same", activation='relu'))
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='relu'))
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='relu'))

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
	model.compile(optimizer="adam", loss="mse")	

	return model

# Load Data
data = load_data()
# Pre-Process data
process_image(data)
# Split data into training, test and validation set
# Get the model
# Train the model
# Save it
