# import all functions
from constants import learning_rate, training_iters, batch_size, display_step, n_inputs, n_classes, dropout
from additional_funcs import find_centroid, find_distance, pre_processing, conv2d, maxpool2d, unroll_image
from variables import x, y, weights, biases 
# import tensorflow library
import tensorflow as tf
import time
# Import MNIST data
import input_data
# one_hot key implies lables in onehot encoding
mnist = input_data.read_data_sets(one_hot=True, train_image_number=60000,test_image_number=10000)

def main_net(x, weights, biases, dropout = dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    conv1 = conv2d(x, weights['wc1'], biases['bc1']) # feature map dimension : 23x23
    pool1 = maxpool2d(conv1) # feature map dimension :  11x11

    conv2 = conv2d(pool1, weights['wc2'], biases['bc2']) # feature map dimension : 6x6
    pool2 = maxpool2d(conv2) # feature map dimension : 3x3
    
    conv3 = conv2d(pool2, weights['wc3'], biases['bc3']) # feature map dimension : 1x1
    
    unroll = tf.reshape(conv3, [-1, weights['out'].get_shape().as_list()[0]])
    print unroll
    
    out = tf.add(tf.matmul(unroll, weights['out']), biases['out'])
    return out

pred = main_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        st = time.time()
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = unroll_image(batch_x)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("loss= {:.6f}".format(loss) + ", Accuracy= {:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    print st
    test_x = pre_processing(mnist.test.images[:256])
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: mnist.test.labels[:256]}))