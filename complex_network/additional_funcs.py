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
		complexImage =  real2complex(binaryImage)
		processed.append(np.array(complexImage, dtype="complex128").flatten())
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
	return np.multiply(np.tanh(np.abs(array)), np.exp(1j * np.angle(array)))

def find_error(weightsList, output, label):
	weightsList = np.flip(weightsList, 0)
	error_array = []
	label = np.array(label, ndmin = 2).T
	for i in xrange(len(weightsList) + 1):
		if i == 0:
			error_array.append(output - label)
		else:
			error_array.append(np.dot(np.array(weightsList[i-1]).T, error_array[i-1]))
	return np.flip(error_array, 0)

def update_weights(weightsList, activationList, errorList):
	weightsList = np.flip(weightsList, 0)
	errorList = np.flip(errorList, 0)
	activationList = np.flip(activationList, 0)
	for i in xrange(len(weightsList)):
		# print weightsList[i].shape, errorList[i].shape, np.divide(np.conj(activationList[i+1]), np.abs(activationList[i+1])).T.shape
		weightsList[i] += np.dot(errorList[i], np.divide(np.conj(activationList[i+1]), np.abs(activationList[i+1])+0.5).T) / len(activationList[i])
	return np.flip(weightsList, 0)