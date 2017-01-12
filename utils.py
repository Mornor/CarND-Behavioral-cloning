import csv
import json
import cv2
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

# Usefull constants
DATA_CSV_PATH = "udacity_data/driving_log.csv"
DATA_DIR_PATH = "udacity_data/"

# Apply random brightness on the image
# If the image is from the center camera, flip it and inverse the angle 50% of the time.
# Scale down the image from (160, 320, 3) to (40, 160, 3)
# Normalize the image between -1 and 1
def process_image(X_train):
	# Read the image and set it to use RGB
	result_img = cv2.imread(X_train[0])
	result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
	# Scale down
	result_img = scale_down(result_img)
	# Apply random brightness
	result_img = change_brightness_img(result_img)
	# Flip it if necessary
	angle = float(X_train[1])
	if("center" in X_train[0]): 
		if np.random.randint(2) == 1:
			result_img, angle = flip_img(result_img, angle)

	return result_img, angle


# Load the CSV names into memory as tuples
# Center image: np.array(csv[:,0])
# Left image: np.array(csv[:,1])
# Right image: np.array(csv[:,2])
# Sterring angle: np.array(csv[:,3], dtype=float)
def load_data():
	with open(DATA_CSV_PATH, 'r') as file:
		reader = csv.reader(file)
		file.readline() # Skip headers
		data = np.array([row for row in reader])
	return data

# Resize to (40, 160, 3)
def scale_down(img): 
	 # Proportionally get lower half portion of the image
    nrow, ncol, nchannel = img.shape
    
    start_row = int(nrow * 0.35)
    end_row = int(nrow * 0.875)   
    
    ## This removes most of the sky and small amount below including the hood
    img_no_sky = img[start_row:end_row, :]

    # This resizes to 66 x 220 for NVIDIA's model
    new_img = cv2.resize(img_no_sky, (220,66), interpolation=cv2.INTER_AREA)
    return new_img
	#return cv2.resize(img, (160, 40), interpolation=cv2.INTER_AREA)

# Process a single image
# Normalize data from 0-255 to -1 - 1
# Scale down image to 32 * 16 (from 160, 320, 3)
def scale_and_normalize(img):
	img = img[::4,::2].copy()
	return img/127.5 - 1.

# Will 'split' the data to obtain the following structure
# center_image, steering_angle_1
# left_image, steering_angle_1 + 0.25
# right_image, steering_angle_1 - 0.25
def split_input(data):
	new_data = np.zeros([0, 2]) # Will be of shape(3*len(data), 2) because 3 images for one steering angle

	for i in range(0, len(data)):
		path_center_images = data[:,0][i].strip()
		path_left_images = data[:,1][i].strip()
		path_right_images = data[:,2][i].strip()
		steering_angle = float(data[:,3][i])
		new_row_center = [DATA_DIR_PATH + path_center_images, steering_angle]
		new_row_left = [DATA_DIR_PATH + path_left_images, steering_angle+0.2]
		new_row_right = [DATA_DIR_PATH + path_right_images, steering_angle-0.2]
		new_data = np.vstack([new_data, new_row_center])
		new_data = np.vstack([new_data, new_row_left])
		new_data = np.vstack([new_data, new_row_right])

	np.savetxt("expanded_driving_log.csv", new_data, delimiter=", ", fmt="%s")
	return new_data

# Save the model under ./model.json, as well as the weights under ./model.h5
def save_model(model):
	with open('./model.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)
	with open('./model.h5', 'w') as outfile:
		model.save_weights('model.h5')

# Flip image horizontally and inverse angle
def flip_img(img, angle):
	return cv2.flip(img, 1), -angle

def change_brightness_img(img): 
	# Randomly select a percent change
	change_pct = np.random.uniform(0.4, 1.2)

	# Change to HSV to change the brightness V
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	hsv[:,:,2] = hsv[:,:,2] * change_pct

	# Convert back to RGB
	img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	return img