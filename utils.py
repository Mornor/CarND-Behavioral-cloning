import csv
import json
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

# Usefull constants
UDACITY_DATA_PATH_CSV = "udacity_data/driving_log.csv"
UDACITY_DATA_PATH_DIR = "udacity_data/"
RECOVERY_DATA_PATH_CSV = "recovery_data/driving_log.csv"
RECOVERY_DATA_PATH_DIR = "recovery_data/"
EXPANDED_DRIVING_LOG_PATH = "expanded_driving_log.csv"

def merge_csv():
	new_csv = np.zeros([0, 2])
	udacity_data = load_csv(UDACITY_DATA_PATH_CSV)
	recovery_data = load_csv(RECOVERY_DATA_PATH_CSV)
	temp_csv = lay_new_data(UDACITY_DATA_PATH_DIR, udacity_data, new_csv)
	final_csv = lay_new_data(RECOVERY_DATA_PATH_DIR, recovery_data, temp_csv)
	np.savetxt(EXPANDED_DRIVING_LOG_PATH, final_csv, delimiter=", ", fmt="%s")

def lay_new_data(dir_path, data, csv_file):
	for i in range(0, len(data)):
		path_center_image = data[:,0][i].strip()
		steering_angle = float(data[:,3][i])
		new_row_center = [dir_path + path_center_image, steering_angle]
		csv_file = np.vstack([csv_file, new_row_center])
	return csv_file

# Load the CSV names into memory as tuples
# Center image: np.array(csv[:,0])
# Left image: np.array(csv[:,1])
# Right image: np.array(csv[:,2])
# Sterring angle: np.array(csv[:,3], dtype=float)
	
def load_csv(csv_path):
	with open(csv_path, 'r') as file:
		reader = csv.reader(file)
		file.readline() # Skip headers
		data = np.array([row for row in reader])
	return data

# Process a single image
# Normalize data from 0-255 to -1 - 1
# Scale down image to 32 * 16 (from 160, 320, 3)
def process_img(img, nvidia):
	if(nvidia): # Reisize to fit Nvidia model
		#img = np.resize(img, (66, 200, 3))
		img = img[::5,::5].copy()
		return img/127.5 - 1.
	img = img[::5,::5].copy()
	return img/127.5 - 1.

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
		flipped_img = flip_img(data[:,0][i]) # Flip the image
		new_image_name = data[:,0][i][:-4] + "_flipped.jpg"	
		flipped_img.save(new_image_name) # Save the image 
		steering_angle = float(data[:,1][i])
		if(steering_angle != 0.0):
			steering_angle = -steering_angle # Inverse the steering angle
		new_data = np.vstack([new_data, [new_image_name, steering_angle]]) # Append the new image to the already-present
	
	# Save as expanded_driving_log.csv	(just in case)
	np.savetxt(EXPANDED_DRIVING_LOG_PATH, new_data, delimiter=", ", fmt="%s")
	return new_data

# Process images from input
def process_images(data, nvidia):
	
	if(nvidia): # Fit Nvidia model input image
		test_images = np.zeros((len(data), 32, 64, 3), dtype=float)

	else: # Load the images into np array of shape (len(data), original_height/5, original_width/20, original_channel)
		test_images = np.zeros((len(data), 32, 64, 3), dtype=float)
	
	for i in range(0, len(data)):
		if(i == len(data)/2):
			print("Half images have been processed")
		test_images[i] = process_img(mpimg.imread(data[:,0][i]), nvidia)
	
	#test_images = np.array([mpimg.imread(path_folder + "/" + file) for file in filenames])
	return test_images