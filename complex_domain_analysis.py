import tensorflow as tf
import numpy as np
import cv2
import input_data
mnist = input_data.read_data_sets(one_hot = True, train_image_number=60000, test_image_number=1000)

learning_rate = 0.01
epoch = 200
test_examples = 256
batch_size = 128
display_step = batch_size/2
categories = 10
periodicity = 1

n_inputs = 784 
n_classes = 10

print "LEARNING RATE = {}".format(learning_rate)+", BATCH SIZE = {}".format(batch_size)+", EPOCHS = {}".format(epoch)
print "####################################################################"


image_real = tf.placeholder(dtype = tf.float32, shape = (None, n_inputs), name = "real_input")
image_imag = tf.placeholder(dtype = tf.float32, shape = (None, n_inputs), name = "imaginary_input")
label_real = tf.placeholder(dtype = tf.float32, shape = (None,n_classes), name = "expected_output_real")
label_imag = tf.placeholder(dtype = tf.float32, shape = (None,n_classes), name = "expected_output_imag")

# label = tf.placeholder(dtype = tf.float32, shape = (None,n_classes), name="expected_output_image")
# miscill... functions

def unroll_image(images):
    processed_images = []
    for image in images:
        channel_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(channel_1, (28, 28), interpolation = cv2.INTER_CUBIC)
        unrolled = np.array(resized, dtype="float").flatten()
        normalized = np.divide(unrolled, 255)
        complexdomain = np.fft.fft(normalized)
        processed_images.append(complexdomain)
    return (np.real(processed_images), np.imag(processed_images))

def activation(input_data, weights, biases):
    real = tf.add(tf.subtract(tf.matmul(input_data['real'], weights['real']), tf.matmul(input_data['imaginary'], weights['imaginary'])), biases['real'])
    imag = tf.add(tf.add(tf.matmul(input_data['real'], weights['imaginary']), tf.matmul(input_data['imaginary'], weights['real'])), biases['imaginary'])
    return {'real':real, 'imaginary': imag}

def nonlinear(activated_layer):
    real = tf.sigmoid(activated_layer['real'])
    imag = tf.sigmoid(activated_layer['imaginary'])
    return {'real': real, 'imaginary': imag}

def z2class(out):
    # angle = tf.cond(out['real']!=0, lambda: tf.divide(out['imaginary'], out['real']), lambda: 1)
    angle = tf.atan(tf.divide(out['imaginary'], out['real']+0.001))
    angle = tf.mod(tf.add(angle, 2*3.141), 2*3.141)
    return {'real':tf.cos(angle), 'imaginary':tf.sin(angle)}

def class2z(batch_y):
	real = []
	imag = []
	for a in batch_y:
		tmp = np.multiply(np.add(a, 0.5), np.dot(np.arange(categories), periodicity * 2 * 3.14))
		real.append(np.cos(tmp))
		imag.append(np.sin(tmp))
	return {'real': real, 'imaginary': imag}

def define_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape = shape, name = name))

weights_real = {
    'wih1': define_variable([n_inputs, 32], 'wr1'),
    'wh1h2': define_variable([32, 64], 'wr2'),
    'wh2h3': define_variable([64, 1024], 'wr3'),
    'wh3h4': define_variable([1024, 1024], 'wr4'),
    'wh4o': define_variable([1024, n_classes], 'wr5'),
}

weights_imaginary ={
    'wih1': define_variable([n_inputs, 32], 'wi1'),
    'wh1h2': define_variable([32, 64], 'wi2'),
    'wh2h3': define_variable([64, 1024], 'wi3'),
    'wh3h4': define_variable([1024, 1024], 'wi4'),
    'wh4o': define_variable([1024, n_classes], 'wi5'),
}

biases_real = {
    'bih1': define_variable([32], 'br1'),
    'bh1h2': define_variable([64], 'br2'),
    'bh2h3': define_variable([1024], 'br3'),
    'bh3h4': define_variable([1024], 'br4'),
    'bh4o': define_variable([n_classes], 'br5'),
}

biases_imaginary = {
    'bih1': define_variable([32], 'bi1'),
    'bh1h2': define_variable([64], 'bi2'),
    'bh2h3': define_variable([1024], 'bi3'),
    'bh3h4': define_variable([1024], 'bi4'),
    'bh4o': define_variable([n_classes], 'bi5'),
}

# main graph

def main_network(x, weights, biases):
    fc1 = activation(x, {'real':weights_real['wih1'], 'imaginary': weights_imaginary['wih1']}, {'real':biases_real['bih1'], 'imaginary': biases_imaginary['bih1']})
    fc1 = nonlinear(fc1)
    
    fc2 = activation(fc1, {'real':weights_real['wh1h2'], 'imaginary': weights_imaginary['wh1h2']}, {'real':biases_real['bh1h2'], 'imaginary': biases_imaginary['bh1h2']})
    fc2 = nonlinear(fc2)
    
    fc3 = activation(fc2, {'real':weights_real['wh2h3'], 'imaginary': weights_imaginary['wh2h3']}, {'real':biases_real['bh2h3'], 'imaginary': biases_imaginary['bh2h3']})
    fc3 = nonlinear(fc3)
    
    fc4 = activation(fc3, {'real':weights_real['wh3h4'], 'imaginary': weights_imaginary['wh3h4']}, {'real':biases_real['bh3h4'], 'imaginary': biases_imaginary['bh3h4']})
    fc4 = nonlinear(fc4)
    
    out = activation(fc4, {'real':weights_real['wh4o'], 'imaginary': weights_imaginary['wh4o']}, {'real':biases_real['bh4o'], 'imaginary': biases_imaginary['bh4o']})
    return out

prediction = main_network({'real':image_real, 'imaginary': image_imag}, {'real':weights_real, 'imaginary':weights_imaginary}, {'real':biases_real, 'imaginary':biases_imaginary})

# prediction = tf.sqrt(tf.add(tf.pow(prediction['real'], 2), tf.pow(prediction['imaginary'], 2)))

# define real and imaginary components of loss function...
real_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction['real'], labels = label_real))
imaginary_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction['imaginary'], labels = label_imag))

	# real_cost = tf.reduce_mean(tf.pow(label_real - prediction['real'], 2))
	# imaginary_cost = tf.reduce_mean(tf.pow(label_imag - prediction['imaginary'], 2))

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = label))



# optimizer for real and imaginary components of loss function...
real_optimizer = tf.train.AdamOptimizer(learning_rate).minimize((real_cost - imaginary_cost))
imaginary_optimizer = tf.train.AdamOptimizer(learning_rate).minimize((imaginary_cost + real_cost))

	# real_optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(real_cost - imaginary_cost)
	# imaginary_optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(imaginary_cost + real_cost)

	# optimizer = tf.train.AdamOptimizer(learning_rate).minimize((cost))

# model evaluation correct real and imaginary prediction ..
correct_real_prediction = tf.equal(tf.argmax(prediction['real'], 1), tf.argmax(label_real, 1))
correct_imag_prediction = tf.equal(tf.argmax(prediction['imaginary'], 1), tf.argmax(label_imag, 1))

# real and imaginary accuracy components...
real_acc = tf.reduce_mean(tf.cast(correct_real_prediction, tf.float32))
imag_acc = tf.reduce_mean(tf.cast(correct_imag_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < epoch * batch_size:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        Rbatch_x, Ibatch_x = unroll_image(batch_x) 
        Rbatch_y, Ibatch_y = np.real(np.fft.fft(batch_y)), np.imag(np.fft.fft(batch_y))

        # sess.run([real_optimizer, imaginary_optimizer], feed_dict={image_real:Rbatch_x, label_real: batch_y['real'], image_imag: Ibatch_x, label_imag: batch_y['imaginary']})
        
        sess.run([real_optimizer, imaginary_optimizer], feed_dict={image_real: Rbatch_x, image_imag: Ibatch_x, label_real: Rbatch_y, label_imag: Ibatch_y})

        if step % display_step == 0:
            Rloss, Racc, Iloss, Iacc = sess.run([real_cost, real_acc, imaginary_cost, imag_acc], feed_dict={image_real:Rbatch_x, image_imag: Ibatch_x, label_real: Rbatch_y, label_imag: Ibatch_y})
            print("loss= {:.6f}".format(Rloss*Iloss) + ", Accuracy= {:.5f}".format(Racc*Iacc))
        step += 1
        
    print("Optimization Finished!")
    
    tRimages, tIimages = unroll_image(mnist.test.images[:test_examples])
    tRlabels, tIlabels = np.real(np.fft.fft(mnist.test.labels[:test_examples])), np.imag(np.fft.fft(mnist.test.labels[:test_examples]))

    print("Testing Accuracy:", sess.run(real_acc, imag_acc, feed_dict={image_real: tRimages, image_imag: tIimages, label_imag: tIlabels, label_real: tRlabels }))