# Necessary imports
import os
import argparse
import json
import csv
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

# Usefull constants
DATA_PATH = "driving_log.csv"

# Load the CSV into memory
# Center image: np.array(csv[:,0])
# Left image: np.array(csv[:,1])
# Left image: np.array(csv[:,2])
# Sterring angle: np.array(csv[:,3], dtype=float)
def load_data():
	with open(DATA_PATH, 'r') as file:
		reader = csv.reader(file)
		data = np.array([row for row in reader])
	return data

# Normalize data from 0-255 to -1 - 1
# TODO size down image by 2, reducing the number of pixels by 4
def process_input(input_data):
	return input_data/127.5 - 1.

# Split data into training, normalization and test set
def split_input(input_data): 


# Use the model defined in Commai following repo :
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

	# 2nd Layer: Convolutional Layer
	model.add(Convolution2D(32, 5, 5, subsample=(3, 3), border_mode="same", activation='relu'))
	# 3rd Layer: Convolutional Layer
	model.add(Convolution2D(32, 5, 5, subsample=(3, 3), border_mode="same", activation='relu'))
	# 4th Layer: Convolutional Layer
	model.add(Convolution2D(32, 5, 5, subsample=(3, 3), border_mode="same", activation='relu'))
	# 5th Layer: Convolutional Layer
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='relu'))
	# 6th Layer: Convolutional Layer
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='relu'))

	# 7th Layer: Flatten Layer
	model.add(Flatten())

	# 8th Layer: Fully Connected Layer
	model.add(Dense(100))
	# 9th Layer: Fully Connected Layer
	model.add(Dense(50))
	# 10th Layer; Fully Connected Layer
	model.add(Dense(10))

	# Final layer
	model.add(Dense(1))

	# Use the Adam optimizer to optimize the mean squared error
	model.compile(optimizer="adam", loss="mse")	

	return model

# Save the model under ./model.json
def save_model():
	with open('./model.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)

# Save the weigth under ./model.h5
def save_weigths(model): 
	model.save_weights("./model.h5", True)

data = load_data()
