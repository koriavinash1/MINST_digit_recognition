# import tensorflow library
import tensorflow as tf
import time
import math
import cv2
import numpy as np
# Import MNIST data
import input_data
# one_hot key implies lables in onehot encoding
mnist = input_data.read_data_sets(one_hot=True, train_image_number=50, test_image_number=10)

# Parameters
learning_rate = 0.001
training_iters = 20
batch_size = 10
display_step = 2

# Network Parameters
n_inputs = 784 # 28*28
n_classes = 10 # (0-9 digits)
dropout = 0.75 # What use?
sess = tf.InteractiveSession()

# define placeholders
x1 = tf.placeholder(tf.float32, shape=(None, n_inputs))
x2 = tf.placeholder(tf.float32, shape=(None, n_inputs))
y = tf.placeholder(tf.float32, shape=(None, n_classes))

# miscill... functions
def define_variable(shape, name): 
    return tf.Variable(tf.truncated_normal(shape, name = name))

def conv2d(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def resampling(image):
	return cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)

def find_centroid(image):
	centx, centy = 0, 0
	for row in xrange(len(image)):
		for col in xrange(len(image[0])):
			if image[row][col]:
				centx += row*col + col
	for row in xrange(len(image)):
		for col in xrange(len(image[0])):
			if image[col][row]:
				centy += row*col + row
	return [centx/len(image), centy/len(image)]

def find_distance(image, centroid):
	distance = 0.00
	for row in xrange(len(image)):
		for col in xrange(len(image[0])):
			if image[row][col]:
				distance += math.sqrt(row**2 + col**2)
	return distance

weights1 = {
    'wc1': define_variable([6, 6, 1, 32], 'w11'), # 5x5 conv, 1 input, 40 outputs 
    'wc2': define_variable([6, 6, 32, 64], 'w12'), # 5x5 conv, 40 inputs, 80 outputs
    'wc3': define_variable([3, 3, 64, 1024], 'w13'), # 5x5 conv, 80 inputs, 500 outputs
    'out': define_variable([1024, n_classes], 'out1') # 1024 inputs, 10 outputs (class prediction)
}

weights2 = {
	'wc1': define_variable([1, 32], 'w21'), 
    'wc2': define_variable([32, 64], 'w22'),
    'wc3': define_variable([64, 1024], 'w23'),
    'out': define_variable([1024, n_classes], 'out2')
}

biases1 = {
    'bc1': define_variable([32], 'b11'),
    'bc2': define_variable([64], 'b12'),
    'bc3': define_variable([1024], 'b13'),
    'out': define_variable([n_classes], 'out1')
}

biases2 = {
    'bc1': define_variable([32], 'b21'),
    'bc2': define_variable([64], 'b22'),
    'bc3': define_variable([1024], 'b23'),
    'out': define_variable([n_classes], 'out2')
}

#tensorboard......
tf.summary.histogram("w21", weights2['wc1'])
tf.summary.histogram("w22", weights2['wc2'])
tf.summary.histogram("w23", weights2['wc3'])
tf.summary.histogram("out2", weights2['out'])

tf.summary.histogram("w11", weights1['wc1'])
tf.summary.histogram("w12", weights1['wc2'])
tf.summary.histogram("w13", weights1['wc3'])
tf.summary.histogram("out1", weights1['out'])

tf.summary.histogram("bc1", biases1['bc1'])
tf.summary.histogram("bc2", biases1['bc2'])
tf.summary.histogram("bc3", biases1['bc3'])
tf.summary.histogram("out", biases1['out'])

tf.summary.histogram("bc1", biases2['bc1'])
tf.summary.histogram("bc2", biases2['bc2'])
tf.summary.histogram("bc3", biases2['bc3'])
tf.summary.histogram("out", biases2['out'])

def network1(x, weights1, biases1, dropout = dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    conv1 = conv2d(x, weights1['wc1'], biases1['bc1']) # feature map dimension : 23x23
    pool1 = maxpool2d(conv1) # feature map dimension :  11x11

    conv2 = conv2d(pool1, weights1['wc2'], biases1['bc2']) # feature map dimension : 6x6
    pool2 = maxpool2d(conv2) # feature map dimension : 3x3
    
    conv3 = conv2d(pool2, weights1['wc3'], biases1['bc3']) # feature map dimension : 1x1
    unroll = tf.reshape(conv3, [-1, weights1['out'].get_shape().as_list()[0]])
    
    out = tf.add(tf.matmul(unroll, weights1['out']), biases1['out'])
    return out

def network2(x, weights2, biases2, dropout = dropout):
	x = tf.reshape(x, shape=[-1, 28, 28, 1])

	x = find_distance(sess.run(x), find_centroid(sess.run(x))) 

	fc1 = tf.add(tf.matmul(x, weights2['wc1']), biases2['bc1'])
	fc1 = tf.nn.relu(fc1)

	fc2 = tf.add(tf.matmul(fc1, weights2['wc2']), biases2['bc2'])
	fc2 = tf.nn.relu(fc2)

	fc3 = tf.add(tf.matmul(fc2, weights2['wc3']), biases2['bc3'])
	fc3 = tf.nn.relu(fc3)

	out = tf.add(tf.matmul(fc3, weights2['out']), biases2['out'])
	return out

pred1 = network1(x1, weights1, biases1)
pred2 = network2(x2, weights2, biases2)

pred = 0.5*pred1 + 0.5*pred2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('/tmp/data', sess.graph)
    step = 1
    while step * batch_size < training_iters:
        st = time.time()
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("loss= {:.6f}".format(loss) + ", Accuracy= {:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    print st
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256]}))