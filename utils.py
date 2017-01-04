import numpy as np
import csv
import matplotlib.image as mpimg

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
# Scale down image to 32 * 16 (from 160, 320, 3)
def process_img(img):
	img = img[::10,::10].copy()
	return img/127.5 - 1.

# Process images from input
def process_images(data):
	# Load the images into np array of shape (len(data), original_height/5, original_width/20, original_channel)
	center_images = np.zeros((len(data), 16, 32, 3), dtype=float)
	left_images = np.zeros((len(data), 16, 32, 3), dtype=float)
	right_images = np.zeros((len(data), 16, 32, 3), dtype=float)
	for i in range(8000, len(data)):
		center_images[i] = process_img(mpimg.imread(data[:,0][i].strip()))
		left_images[i] = process_img(mpimg.imread(data[:,1][i].strip()))
		right_images[i] = process_img(mpimg.imread(data[:,2][i].strip()))
	
	#print(center_images.shape)
	#test_images = np.array([mpimg.imread(path_folder + "/" + file) for file in filenames])
	return center_images, left_images, right_images