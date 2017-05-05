from iteration_utilities import deepflatten
import tensorflow as tf
import numpy as np
import math
import cv2

session = tf.InteractiveSession()

# miscill... functions
def define_variable(shape, name): 
    return tf.Variable(tf.truncated_normal(shape, name = name))

def conv2d(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def unroll_image(images):
	processed = []
	for image in images:
		tmp = resampling(image)
		final_img = binary_image(tmp)
		processed.append(np.array(final_img).flatten())
	return processed

def resampling(image):
	return cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)

def binary_image(image):
	image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
	return thresh

# this finds centroid using direct numpy function..
def find_centroid(image):
	centx, centy = np.where(image!=0)
	return [np.sum(centx)/len(image), np.sum(centy)/len(image)]

# finding COM in very trivial method...
def alternative_centroid(image):
	centx, centy = 0, 0
	trans_image = np.transpose(image)
	for row in xrange(len(image)):
		for col in xrange(len(image)):
			if image[row][col] == 255:
				centx += row
			if trans_image[row][col] == 255:
				centy += col
	return [centx/len(image), centy/len(image)]


# this distance is based of relative centroid of the image centroid is like COM of the image...
# all the white pixel are given -10 as their distance so as to diff. them with other pixels
def find_distance(image, centroid):
	cv2.imshow("tmp",image)
	distances = []
	for row in xrange(len(image)):
		for col in xrange(len(image)):
			if image[row][col] == 255:
				distances.append(math.sqrt((centroid[0]-row)**2 + (centroid[1]-col)**2))
			else:
				distances.append(-1)
	return distances

# finding the distance based of relative position w.r.t origin after unrolling the binary image which gives discrete signal... 
def distance_with_referance(image):
	# cv2.imshow("tmp",image)
	fshift = np.fft.fftshift(image)
	rows, cols = image.shape
	crow,ccol = rows/2 , cols/2
	fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
	distances = []
	image = np.array(fshift).flatten()
	for px in image:
		distances.append(np.abs(px))
	return distances

def pre_processing(images):
	dtype = "float32"
	processed = []
	for image in images:
		tmp = resampling(image)
		final_img = binary_image(tmp)
		processed.append(distance_with_referance(np.fft.fft2(final_img)))
	return processed