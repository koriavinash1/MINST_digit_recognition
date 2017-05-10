import numpy as np
from constants import learning_rate
import math
import cv2

# convert real to complex image..
def real2complex(image):
	image = np.fft.fft2(image)
	fshift = np.fft.fftshift(image)
	rows, cols = image.shape
	crow,ccol = rows/2 , cols/2
	fshift[crow-30:crow+30, ccol-30:ccol+30] = 0	
	return fshift

# returns unit vector
def complex2polarcoordinates(arrays):
	polar_array = []
	for array in arrays:
		polar_array.append(np.exp(1j * np.angle(array)))
	return polar_array

def resampling(image):
	return cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)

def grey2binary(image):
	image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
	return thresh

def unroll_image(images):
	processed = []
	for image in images:
		reshapeImage = resampling(image)
		binaryImage = grey2binary(reshapeImage)
		# cv2.imshow("test", binaryImage)
		# cv2.waitKey(800)
		complexImage =  real2complex(binaryImage)
		processed.append(np.exp(1j * np.angle(np.array(complexImage, dtype="complex128").flatten())))
	return processed



##### network functions.........


def initweights(rows, cols):
	weights = np.random.uniform(-1.0, 1.0, (cols, rows))
	weights = np.array(weights, ndmin = 2, dtype = 'complex128')
	weights += 1j * np.random.uniform(-1.0, 1.0, (cols, rows))
	return weights

def initbiases(cols):
	weights = np.random.uniform(-1.0, 1.0, (cols, 1))
	weights = np.array(weights, ndmin = 1, dtype = 'complex128')
	weights += 1j * np.random.uniform(-1.0, 1.0, (cols, 1))
	return weights

def activation(array):
	sigmoid = np.divide(1, np.add(1, np.exp(-np.abs(array)))) 
	out = np.multiply(sigmoid, np.exp(1j * np.angle(array)))
	print "activation:   ", max((out+1)/2)
	return (out + 1)/2

def find_error(weightsList, nodes, label):
	error_array = []
	error_array.append(np.sqrt(np.sum(np.square(np.subtract(nodes[len(nodes)-1], label).T))))
	# print nodes[len(nodes)-1], label
	# print weightsList[len(weightsList)-1].T.shape, error_array[0].shape
	for i in range(len(weightsList)):
		error_array.append(np.dot(weightsList[len(weightsList)-i-1].T, error_array[i]))
	return error_array

def update_weights(weightsList, activationList, errorList):
	for i in xrange(len(weightsList)):
		# print weightsList[i].shape, np.array(errorList[len(weightsList)-i], ndmin=2).shape, np.array(np.conj(activationList[i+1]), ndmin=2).shape
		np.add(weightsList[i], learning_rate * np.dot(np.array(errorList[len(weightsList)-i], ndmin=2), np.array(np.conj(activationList[i+1]), ndmin=2)).T/ len(activationList[i]))
	# print weightsList[1]
	return weightsList