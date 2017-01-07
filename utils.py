import csv
import json
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

# Usefull constants
DATA_PATH = "driving_log.csv"

# Load the CSV names into memory as tuples
# Center image: np.array(csv[:,0])
# Left image: np.array(csv[:,1])
# Right image: np.array(csv[:,2])
# Sterring angle: np.array(csv[:,3], dtype=float)
def load_data():
	with open(DATA_PATH, 'r') as file:
		reader = csv.reader(file)
		file.readline() # Skip headers
		data = np.array([row for row in reader])
	return data

# Process a single image
# Normalize data from 0-255 to -1 - 1
# Scale down image to 32 * 16 (from 160, 320, 3)
def process_img(img, nvidia=False):
	if(nvidia): # Reisize to fit Nvidia model
		img = np.resize(img, (66, 200, 3))
		return img/127.5 - 1.
	img = img[::2,::2].copy()
	return img/127.5 - 1.

# Will 'split' the data to obtain the following structure
# center_image, steering_angle_1
# left_image, steering_angle_1 + 0.25
# right_image, steering_angle_1 - 0.25
def split_input(data):
	new_data = np.zeros([0, 2]) # Will be of shape(3*len(data), 2) because 3 images for one steering angle

	for i in range(0, len(data)):
		path_center_images = np.array(data[:,0][i].strip())
		path_left_images = np.array(data[:,1][i].strip())
		path_right_images = np.array(data[:,2][i].strip())
		steering_angle = np.array(data[:,3][i], dtype=float)
		new_row_center = [path_center_images, steering_angle]
		new_row_left = [path_left_images, steering_angle+0.08]
		new_row_right = [path_right_images, steering_angle-0.08]
		new_data = np.vstack([new_data, new_row_center])
		new_data = np.vstack([new_data, new_row_left])
		new_data = np.vstack([new_data, new_row_right])

	return new_data

# Save the model under ./model.json, as well as the weights under ./model.h5
def save_model(model):
	with open('./model.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)
	with open('./model.h5', 'w') as outfile:
		model.save_weights('model.h5')

# Flip image horizontally
def flip_img(img_path):
	img_to_flip = Image.open(img_path)
	flipped_img = img_to_flip.transpose(Image.FLIP_LEFT_RIGHT)
	return flipped_img

# Flip center images and inverse the steering angle. 
# Also, avoid bias to the left (since the track mostly turn to left direction)
def flip_center_images(data):
	new_data = np.copy(data)
	for i in range(0, len(data)):
		if("center" in data[:,0][i]):
			flipped_img = flip_img(data[:,0][i]) # Flip the image
			new_image_name = data[:,0][i][:-4] + "_flipped.jpg"	
			flipped_img.save(new_image_name) # Save the image 
			steering_angle = float(data[:,1][i])
			if(steering_angle != 0.0):
				steering_angle = -steering_angle # Inverse the steering angle
			new_data = np.vstack([new_data, [new_image_name, steering_angle]]) # Append the new image to the already-present
	
	# Save as expanded_driving_log.csv	(just in case)
	np.savetxt("expanded_driving_log.csv", new_data, delimiter=", ", fmt="%s")
	return new_data

# Process images from input
def process_images(data, nvidia=False):
	
	if(nvidia): # Fit Nvidia model input image
		test_images = np.zeros((len(data), 66, 200, 3), dtype=float)

	else: # Load the images into np array of shape (len(data), original_height/5, original_width/20, original_channel)
		test_images = np.zeros((len(data), 80, 160, 3), dtype=float)
	
	for i in range(0, len(data)):
		if(i == 12000):
			print("Half images have been processed")
		test_images[i] = process_img(mpimg.imread(data[:,0][i]), nvidia)
	
	#test_images = np.array([mpimg.imread(path_folder + "/" + file) for file in filenames])
	return test_images
# wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
