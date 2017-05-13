import tensorflow as tf
import numpy as np
import cv2
from additional_funcs import pre_processing
import input_data
mnist = input_data.read_data_sets(one_hot=True, train_image_number=60000, test_image_number=1000)
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("./dataset", one_hot=True)

n_input = 784
n_classes = 10
learning_rate = 0.001
batch_size = 128
epochs = 25
display_steps = batch_size/2
test_examples = 256
IS_POSITION_BASED = False

print "IS_POSITION_BASED: ", IS_POSITION_BASED

x = tf.placeholder(tf.float32, shape=(None, n_input))
y = tf.placeholder(tf.float32, shape=(None, n_classes))

if IS_POSITION_BASED:
    X_pos = tf.placeholder(tf.float32, [None, n_input])
    Y_pos = tf.placeholder(tf.float32, [None, n_input])

# some input arrays
nx, ny = (28, 28)
xt = np.linspace(0, 1, nx)
yt = np.linspace(0, 1, ny)
xpos, ypos = np.meshgrid(xt, yt)
xpos = np.array(xpos).flatten()
ypos = np.array(ypos).flatten()
# print len(xpos), len(ypos)

def define_variable(shape, name): 
    return tf.Variable(tf.truncated_normal(shape, name = name))

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

weights = {
    'wc1_v': define_variable([5, 5, 1, 32], 'wc1'), 
    'wc2': define_variable([5, 5, 32, 64], 'wc2'),
    'wfc1': define_variable([7*7*64, 2048], 'wfc1'),
    'wfc2': define_variable([2048, 1024], 'wfc2'), 
    'out': define_variable([1024, n_classes], 'out')
}

if IS_POSITION_BASED:
    weights['wc1_xpos'] = define_variable([5, 5, 1, 32], "wxpos")
    weights['wc1_ypos'] = define_variable([5, 5, 1, 32], "wypos")

biases = {
    'bc1_v': define_variable([32], 'bc1'),
    'bc2': define_variable([64], 'bc2'),
    'bfc1': define_variable([2048], 'bfc1'),
    'bfc2': define_variable([1024], 'bfc2'),
    'out': define_variable([n_classes], 'out')
}

if IS_POSITION_BASED:
    biases['bc1_xpos'] = define_variable([32], "bxpos")
    biases['bc1_ypos'] = define_variable([32], "bypos")
    

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

pred = main_network(x, weights, biases)
    
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= epochs * batch_size:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = pre_processing(batch_x)
        # print len(batch_x)
        
        if IS_POSITION_BASED:
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, X_pos: np.array([xpos,]*batch_size, ndmin=2), Y_pos: np.array([ypos,]*batch_size, ndmin=2)})
        else:
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        
        if step % display_steps == 0:
            
            if IS_POSITION_BASED:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, X_pos: np.array([xpos,]*batch_size, ndmin=2), Y_pos: np.array([ypos,]*batch_size, ndmin=2)})
            else:
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                
            print("EPOCH= {:.1f}".format(step/batch_size)+", loss= {:.6f}".format(loss) + ", Accuracy= {:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    test_images = pre_processing(mnist.test.images[:test_examples])
    
    if IS_POSITION_BASED:
    	print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_images, y: mnist.test.labels[:test_examples], X_pos: np.array([xpos,]*test_examples, ndmin=2), Y_pos: np.array([ypos,]*test_examples, ndmin=2)}))
    else:
    	print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_images, y: mnist.test.labels[:test_examples]}))