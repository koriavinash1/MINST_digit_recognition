import tensorflow as tf
from constants import n_inputs, n_classes
from additional_funcs import define_variable

# define placeholders
x = tf.placeholder(tf.float32, shape=(None, n_inputs))
y = tf.placeholder(tf.float32, shape=(None, n_classes))

# weigths define...
weights2 = {
	'wc1': define_variable([n_inputs, 32], 'w21'), 
    'wc2': define_variable([32, 64], 'w22'),
    'wc3': define_variable([64, 1024], 'w23'),
    'out': define_variable([1024, n_classes], 'out2')
}

# biases define....
biases2 = {
    'bc1': define_variable([32], 'b21'),
    'bc2': define_variable([64], 'b22'),
    'bc3': define_variable([1024], 'b23'),
    'out': define_variable([n_classes], 'out2')
}
