import tensorflow as tf
import numpy as np
import cv2
import os

n_input = 784
n_classes = 10
batch_size = 1
IS_POSITION_BASED = True

x = tf.placeholder(tf.float32, shape=(1, n_input))

if IS_POSITION_BASED:
    X_pos = tf.placeholder(tf.float32, [1, n_input])
    Y_pos = tf.placeholder(tf.float32, [1, n_input])

# some input arrays
nx, ny = (28, 28)
xt = np.linspace(0, 1, nx)
yt = np.linspace(0, 1, ny)
xpos, ypos = np.meshgrid(xt, yt)
xpos = np.square(np.array(xpos).flatten())
ypos = np.square(np.array(ypos).flatten())
# print len(xpos), len(ypos)

load_bool = input("Press 1 load an image or 2 to take new image:  ")

if load_bool == 1:
	load_str = input("Enter image path...  ")

if not os.path.exists(load_str):
	load_str = input("Enter image path...  ")

test_image = cv2.imread(load_str)
	
def pre_processing(image):
	print "Pre-processing the image...."
	channel_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(channel_1, (28, 28), interpolation = cv2.INTER_CUBIC)
	unrolled = np.array(resized, dtype="float").flatten()
	normalized = np.divide(unrolled, 255)
	return np.array(normalized, ndmin=2)

def define_variable(value, name): 
    return tf.Variable(value, name = name)

def activation(x, w, b):
    return tf.add(tf.matmul(x, w), b)

def nonlinear(x):
    return tf.nn.relu(x)

def conv2d(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


print "Loading weights and biases for network...."
with tf.device("/cpu:0"):
    weights = {
        'wc1_v': define_variable(np.load("./weights/wc1_v.npy"), 'wc1'), 
        'wc2': define_variable(np.load("./weights/wc2.npy"), 'wc2'),
        'wfc1': define_variable(np.load("./weights/wfc1.npy"), 'wfc1'),
        'wfc2': define_variable(np.load("./weights/wfc2.npy"), 'wfc2'), 
        'out': define_variable(np.load("./weights/out.npy"), 'out')
    }

    if IS_POSITION_BASED:
        weights['wc1_xpos'] = define_variable(np.load("./weights/wc1_xpos.npy"), "wxpos")
        weights['wc1_ypos'] = define_variable(np.load("./weights/wc1_ypos.npy"), "wypos")

    biases = {
        'bc1_v': define_variable(np.load("./biases/bc1_v.npy"), 'bc1'),
        'bc2': define_variable(np.load("./biases/bc2.npy"), 'bc2'),
        'bfc1': define_variable(np.load("./biases/bfc1.npy"), 'bfc1'),
        'bfc2': define_variable(np.load("./biases/bfc2.npy"), 'bfc2'),
        'out': define_variable(np.load("./biases/out.npy"), 'out')
    }

    if IS_POSITION_BASED:
        biases['bc1_xpos'] = define_variable(np.load("./biases/bc1_xpos.npy"), "bxpos")
        biases['bc1_ypos'] = define_variable(np.load("./biases/bc1_ypos.npy"), "bypos")

def main_network(x, weights, biases):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1_v = conv2d(x, weights['wc1_v'], biases['bc1_v'])
    conv1 = conv1_v

    if IS_POSITION_BASED:
        xpos = tf.reshape(X_pos, shape=[-1, 28, 28, 1])
        ypos = tf.reshape(Y_pos, shape=[-1, 28, 28, 1])
        conv1_xpos = conv2d(xpos, weights['wc1_xpos'], biases['bc1_xpos'])
        conv1_ypos = conv2d(ypos, weights['wc1_ypos'], biases['bc1_ypos'])
        conv1 = tf.add(tf.add(conv1_xpos, conv1_ypos), conv1_v)
        conv1 = tf.divide(conv1,3)

    pool1 = maxpool2d(conv1)

    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    pool2 = maxpool2d(conv2)
        
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool2, [-1, weights['wfc1'].get_shape().as_list()[0]])
    
    # fully connected layer 1
    fc1 = tf.add(tf.matmul(fc1, weights['wfc1']), biases['bfc1'])
    fc1 = tf.nn.relu(fc1)
    
    # fully connected layer 2
    fc2 = tf.add(tf.matmul(fc1, weights['wfc2']), biases['bfc2'])
    fc2 = tf.nn.relu(fc2)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)
batch_x = pre_processing(test_image)

predict = main_network(x, weights, biases)
print predict

print "input: {}".format(batch_x.shape) + "xpos: {}".format(np.array([xpos,]*batch_size, ndmin=2).shape)

if IS_POSITION_BASED:
    sess.run(predict, feed_dict={x: batch_x, X_pos: np.array([xpos,]*batch_size, ndmin=2), Y_pos: np.array([ypos,]*batch_size, ndmin=2)})
else:
    sess.run(predict, feed_dict={x: batch_x})

print "IMAGE UPLOADED IS:  {}".format(sess.run(predict))