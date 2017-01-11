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
	if("center" in X_train[0]): 	
		result_img = cv2.imread(X_train[0])
		result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
		# Scale down
		result_img = scale_down(result_img)
		# Apply random brightness
		result_img = change_brightness_img(result_img)
		# Flip it if necessary
		angle = float(X_train[1])

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
	return cv2.resize(img, (160, 40), interpolation=cv2.INTER_AREA)

# Will 'split' the data to obtain the following structure
# center_image, steering_angle_1
def split_input(data):
	new_data = np.zeros([0, 2]) # Will be of shape(3*len(data), 2) because 3 images for one steering angle

	for i in range(0, len(data)):
		path_center_images = data[:,0][i].strip()
		steering_angle = float(data[:,3][i])
		new_row_center = [DATA_DIR_PATH + path_center_images, steering_angle]
		new_data = np.vstack([new_data, new_row_center])
	
	np.savetxt("expanded_driving_log.csv", new_data, delimiter=", ", fmt="%s")
	return new_data

# Save the model under ./model.json, as well as the weights under ./model.h5
def save_model(model):
	with open('./model.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)
	with open('./model.h5', 'w') as outfile:
		model.save_weights('model.h5')

# Flip image horizontally
def flip_img(img):
	return cv2.flip(img, 1)

def flip_images(data): 
	new_data = np.copy(data)
	for i in range(0, len(data)):
		if("center" in data[:,0][i]):
			image = cv2.imread(data[:,0][i])
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			flipped_img = flip_img(image) # Flip the image
			new_image_name = data[:,0][i][:-4] + "_flipped.jpg"	
			Image.fromarray(flipped_img).save(new_image_name) # Save the image 
			steering_angle = float(data[:,1][i])
			if(steering_angle != 0.0):
				steering_angle -= steering_angle # Inverse the steering angle
			new_data = np.vstack([new_data, [new_image_name, steering_angle]]) # Append the new image to the already-present
	
	# Save as expanded_driving_log.csv	(just in case)
	np.savetxt("expanded_driving_log.csv", new_data, delimiter=", ", fmt="%s")
	return new_data

def change_brightness_img(img): 
	# Randomly select a percent change
	change_pct = np.random.uniform(0.4, 1.2)

	# Change to HSV to change the brightness V
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	hsv[:,:,2] = hsv[:,:,2] * change_pct

	# Convert back to RGB
	img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	return img