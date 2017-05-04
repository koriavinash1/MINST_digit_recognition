import tensorflow as tf
import numpy as np
import cv2
import math

image = cv2.imread("1.jpg")
print image

def find_centroid():
	global image
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (4, 4), interpolation = cv2.INTER_AREA)
	centx, centy = np.where(image!=0)
	return [np.sum(centx)/len(image)**2, np.sum(centy)/len(image)**2]

def find_distance(centroid):
	print centroid
	distance = []
	for row in xrange(len(image)):
		for col in xrange(len(image)):
			if image[row][col]:
				distance.append(math.sqrt((centroid[0] - row)**2 + (centroid[1]-col)**2))
			else:
				distance.append(0)
	return distance

print find_distance(find_centroid())
