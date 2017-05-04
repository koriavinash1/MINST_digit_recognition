# import all functions
from constants import learning_rate, training_iters, batch_size, display_step, n_inputs, n_classes, dropout
from additional_funcs import find_centroid, find_distance, pre_processing
from variables import x, y, weights2, biases2 
# import tensorflow library
import tensorflow as tf
import time
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# one_hot key implies lables in onehot encoding
mnist = input_data.read_data_sets("dataSet", one_hot=True)

session = tf.InteractiveSession()

def network2(x, weights2, biases2, dropout = dropout):
	# x = tf.reshape(x, shape=[-1, 28, 28, 1])
	# x = find_distance(session.run(x), find_centroid(session.run(x))) 

	fc1 = tf.add(tf.matmul(x, weights2['wc1']), biases2['bc1'])
	fc1 = tf.nn.relu(fc1)

	fc2 = tf.add(tf.matmul(fc1, weights2['wc2']), biases2['bc2'])
	fc2 = tf.nn.relu(fc2)

	fc3 = tf.add(tf.matmul(fc2, weights2['wc3']), biases2['bc3'])
	fc3 = tf.nn.relu(fc3)

	out = tf.add(tf.matmul(fc3, weights2['out']), biases2['out'])
	return out

pred2 = network2(x, weights2, biases2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred2, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred2, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    step = 1
    while step * batch_size < training_iters:
        st = time.time()
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = pre_processing(batch_x)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("loss= {:.6f}".format(loss) + ", Accuracy= {:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    print st
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256]}))