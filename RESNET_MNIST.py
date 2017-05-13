import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def conv_2D(x, w, b=None, stride=1, padding='SAME', activation=None):
    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
    # add biases
    if b is not None:
        x = tf.nn.bias_add(x, b)

    if activation is not None:
        x = activation(x)

    return x


def print_tensor_shape(x, msg=''):
    print(msg, x.get_shape().as_list())


class RepBlock(object):
    def __init__(self, num_repeats, num_filters, bottleneck_size, name_scope):
        self.num_repeats = num_repeats
        self.num_filters = num_filters
        self.bottleneck_size = bottleneck_size
        self.name_scope = name_scope

    def apply_block(self, net):

        print_tensor_shape(net, 'entering apply_block')

        # loop over repeats
        for i_repeat in range(self.num_repeats):

            print_tensor_shape(net, 'layer %i' % i_repeat)

            # subsampling is performed by a convolution with stride=2, only
            # for the first convolution of the first repetition
            if i_repeat == 0:
                stride = 2
            else:
                stride = 1

            name = self.name_scope+'/%i/conv_in' % i_repeat
            with tf.variable_scope(name):
                w = tf.get_variable('w', shape=[1, 1, net.get_shape().as_list()[-1], self.bottleneck_size], initializer=tf.contrib.layers.xavier_initializer_conv2d())
                b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[self.bottleneck_size]))
                # w = tf.get_variable('w', initializer=tf.random_normal([1, 1, net.get_shape().as_list()[-1], self.bottleneck_size]))
                # b = tf.get_variable('b', initializer=tf.random_normal([self.bottleneck_size]))
                conv = conv_2D(net, w, b, stride=stride, padding='VALID', activation=tf.nn.relu)

            print_tensor_shape(conv, name)

            name = self.name_scope+'/%i/conv_bottleneck' % i_repeat
            with tf.variable_scope(name):
                w = tf.get_variable('w', shape=[3, 3, conv.get_shape().as_list()[-1], self.bottleneck_size],
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
                b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[self.bottleneck_size]))
                # w = tf.get_variable('w', initializer=tf.random_normal([3, 3, conv.get_shape().as_list()[-1], self.bottleneck_size]))
                # b = tf.get_variable('b', initializer=tf.random_normal([self.bottleneck_size]))
                conv = conv_2D(conv, w, b, stride=1, padding='SAME', activation=tf.nn.relu)

                print_tensor_shape(conv, name)

            name = self.name_scope+'/%i/conv_out' % i_repeat
            with tf.variable_scope(name):
                w = tf.get_variable('w', shape=[1, 1, conv.get_shape().as_list()[-1], self.num_filters],
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
                b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[self.num_filters]))
                # w = tf.get_variable('w', initializer=tf.random_normal([1, 1, conv.get_shape().as_list()[-1], self.num_filters]))
                # b = tf.get_variable('b', initializer=tf.random_normal([self.num_filters]))
                conv = conv_2D(conv, w, b, stride=1, padding='VALID', activation=None)
                print_tensor_shape(conv, name)

            if i_repeat == 0:
                net = conv + tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            else:
                net = conv + net

            net = tf.nn.relu(net)

        return net


def resnet(x):
    # reshape input
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # init block for each layer
    layer_1 = RepBlock(num_repeats=3, num_filters=128, bottleneck_size=32, name_scope='layer_1')
    layer_2 = RepBlock(num_repeats=3, num_filters=256, bottleneck_size=64, name_scope='layer_2')
    layer_3 = RepBlock(num_repeats=3, num_filters=512, bottleneck_size=128, name_scope='layer_3')
    layer_4 = RepBlock(num_repeats=3, num_filters=1024, bottleneck_size=256, name_scope='layer_4')

    layers = [layer_1, layer_2, layer_3, layer_4]

    # first layer
    with tf.variable_scope('conv_1'):
        w = tf.get_variable('w', shape=[7, 7, x.get_shape().as_list()[-1], 64],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # w = tf.get_variable('w', initializer=tf.random_normal([7, 7, x.get_shape().as_list()[-1], 64]))
        b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[64]))
        net = conv_2D(x, w, b, stride=1, padding='SAME', activation=tf.nn.relu)

    print_tensor_shape(net, 'conv_1')

    net = tf.nn.max_pool(net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    print_tensor_shape(net, 'After max pooling')

    with tf.variable_scope('conv_2'):
        w = tf.get_variable('w', shape=[1, 1, net.get_shape().as_list()[-1], layers[0].num_filters],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # w = tf.get_variable('w', initializer=tf.random_normal([1, 1, net.get_shape().as_list()[-1], layers[0].num_filters]))
        b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[layers[0].num_filters]))
        net = conv_2D(net, w, b, stride=1, padding='SAME', activation=tf.nn.relu)

    print_tensor_shape(net, 'conv_2')

    for i_layer, layer in enumerate(layers):
        print i_layer, layer

        # pass the net through all blocks of the layer
        net = layer.apply_block(net)

        print_tensor_shape(net, 'After block')

        try:
            # upscale (depth) to the next block size
            next_block = layers[i_layer+1]
            with tf.variable_scope('upscale_%i' % i_layer):
                w = tf.get_variable('w', shape=[1, 1, net.get_shape().as_list()[-1], next_block.num_filters],
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
                # w = tf.get_variable('w', initializer=tf.random_normal([1, 1, net.get_shape().as_list()[-1], next_block.num_filters]))
                b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[next_block.num_filters]))
                net = conv_2D(net, w, b, stride=1, padding='SAME', activation=tf.nn.relu)

            print_tensor_shape(net)

        except IndexError:
            pass

    # apply average pooling
    net = tf.nn.avg_pool(net, ksize=[1, net.get_shape().as_list()[1], net.get_shape().as_list()[2], 1],
                                     strides=[1, 1, 1, 1], padding='VALID')

    print_tensor_shape(net, msg='after average pooling')

    # fully connected layer
    with tf.variable_scope('fc'):
        w = tf.get_variable('w', shape=[256, 10],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # w = tf.get_variable('w', initializer=tf.random_normal([256, 10]))
        b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[10]))

    net = tf.reshape(net, shape=[-1, 256])
    net = tf.add(tf.matmul(net, w), b)

    print_tensor_shape(net, 'after fc')

    return net

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    Y_pred = resnet(X)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
    optim = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    correct_pred = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    session = tf.InteractiveSession()
    init_op = tf.initialize_all_variables()
    session.run(init_op)

    nb_epochs = 10
    batch_size = 128
    training_size = mnist.train.num_examples

    nb_mini_batches = training_size // batch_size

    # loop over epochs
    for i_epoch in range(nb_epochs):

        # loop over mini-batches
        for i_batch in range(nb_mini_batches):

            # get mini-batch
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            [_, cost_val, acc] = session.run([optim, cost, accuracy], feed_dict={X: batch_x, Y:batch_y})

            print('epoch %i - batch %i - cost=%f - accuracy=%f' % (i_epoch, i_batch, cost_val, acc))
