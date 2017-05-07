import tensorflow as tf
from constants import n_inputs, n_classes
from additional_funcs import define_variable

# define placeholders
x = tf.placeholder(tf.float32, shape=(None, n_inputs))
y = tf.placeholder(tf.float32, shape=(None, n_classes))

# weigths define...
weights2 = {
	'wc1': define_variable([n_inputs, 64], 'w21'), 
    'wc2': define_variable([64, 128], 'w22'),
    'wc3': define_variable([128, 1024], 'w23'),
    'out': define_variable([1024, n_classes], 'out2')
}

# biases define....
biases2 = {
    'bc1': define_variable([64], 'b21'),
    'bc2': define_variable([128], 'b22'),
    'bc3': define_variable([1024], 'b23'),
    'out': define_variable([n_classes], 'out2')
}

weights = {
    'wc1': define_variable([6, 6, 1, 40], 'wc1'), # 5x5 conv, 1 input, 40 outputs 
    'wc2': define_variable([6, 6, 40, 80], 'wc2'), # 5x5 conv, 40 inputs, 80 outputs
    'wc3': define_variable([3, 3, 80, 1024], 'wc3'), # 5x5 conv, 80 inputs, 500 outputs
    'out': define_variable([1024, n_classes], 'out') # 1024 inputs, 10 outputs (class prediction)
}
tf.summary.histogram("wc1", weights['wc1'])
tf.summary.histogram("wc2", weights['wc2'])
tf.summary.histogram("wc3", weights['wc3'])
tf.summary.histogram("out", weights['out'])

biases = {
    'bc1': define_variable([40], 'bc1'),
    'bc2': define_variable([80], 'bc2'),
    'bc3': define_variable([1024], 'bc3'),
    'out': define_variable([n_classes], 'out')
}
tf.summary.histogram("bc1", biases['bc1'])
tf.summary.histogram("bc2", biases['bc2'])
tf.summary.histogram("bc3", biases['bc3'])
tf.summary.histogram("out", biases['out'])