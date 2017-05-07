import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from additional_funcs import resampling, binary_image, find_centroid, find_distance, pre_processing, alternative_centroid, distance_with_referance

image = cv2.imread("103.jpg")
image1 = cv2.imread("1.jpg")

data = pre_processing([image])[0]
data2 = pre_processing([image1])[0]
# image = resampling(image)
# image = binary_image(image)

# image1 = resampling(image1)
# image1 = binary_image(image1)

# image2 = cv2.imread("104.jpg")
# image2 = resampling(image2)
# image2 = binary_image(image2)

# image = np.fft.fft(image)
# image1 = np.fft.fft(image1)
# image2 = np.fft.fft(image2)

# # cv2.imshow("test", np.array(image, dtype = "float32"))
# # cv2.imshow("test1", np.array(image1, dtype = "float32"))
# # cv2.waitKey(10000)

# image = np.array(image).flatten()
# image1 = np.array(image1).flatten()
# image2 = np.array(image2).flatten()
img = np.array(data).reshape(28,28)
# cv2.imshow("test", img)
# cv2.waitKey(0)
plt.plot(data, 'r', data2, 'g')
plt.show()
