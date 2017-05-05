from iteration_utilities import deepflatten
import tensorflow as tf
import numpy as np
import math
import cv2

session = tf.InteractiveSession()

# miscill... functions
def define_variable(shape, name): 
    return tf.Variable(tf.truncated_normal(shape, name = name))

def resampling(image):
	return cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)

def binary_image(image):
	ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
	return thresh

def find_centroid(image):
	centx, centy = np.where(image!=0)
	return [np.sum(centx)/len(image)**2, np.sum(centy)/len(image)**2]

def find_distance(image, centroid):
	cv2.imshow("tmp",image)
	distances = []
	for row in xrange(len(image)):
		for col in xrange(len(image)):
			if image[row][col] == 1:
				distances.append(math.sqrt((centroid[0] - row)**2 + (centroid[1]-col)**2))
			else:
				distances.append(0)
	print len(distances)
	return distances

def pre_processing(images):
	dtype = "float32"
	processed = []
	for image in images:
		tmp = resampling(image)
		final_img = binary_image(tmp)
		processed.append(find_distance(final_img, find_centroid(final_img)))
	print len(processed)
	return np.array(processed, dtype=dtype)