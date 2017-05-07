import numpy as np
from additional_funcs import unroll_image, initbiases, initweights, update_weights, find_error, activation
from constants import batch_size, training_iters, n_inputs, n_classes, periodicity, display_step
import input_data
mnist = input_data.read_data_sets(one_hot=True)

class network:
	def __init__(self, weightsList, biasesList, categories, periodicity):
		self.weightsList = weightsList
		self.biasesList = biasesList
		self.categories = categories
		self.periodicity = periodicity
		pass

	def z_to_class(self, z):
		angle = np.mod(np.angle(z) + 2*np.pi, 2*np.pi)
		p = np.floor (self.categories * self.periodicity * angle / (2*np.pi))
		p = np.mod(p, self.categories).T
		# print "shape: " + str(p.shape) + "input: " + str(z.shape)
		return p

	def class_to_angles(self, c):
		angles = (c + 0.5 + (self.periodicity * np.arange(self.categories))) / (self.categories * self.periodicity) * 2 * np.pi
		return angles

	def status(self):
		print "periodicity: " + str(self.periodicity) + " categories: " + str(self.categories)
		pass

	def train_network(self, train_data, train_labels):
		errorList = []
		for k in range(len(train_data)):
			activation_layers = []
			data = np.array(train_data[k], ndmin = 2, dtype='complex128').T
			activation_layers.append(data)
			for i in range(len(self.weightsList)):
				# print "b: " + str(biasesList[i].shape) + "w: " + str(weightsList[i].shape) + " data: "+ str(data.shape) + " act: "+str(activation_layers[i].shape)
				if i == 0:
					activation_layers.append(activation(np.dot(self.weightsList[i], data) + self.biasesList[i]))
				else:
					activation_layers.append(activation(np.dot(self.weightsList[i], activation_layers[i]) + self.biasesList[i]))
			# print "trainactivation: " + str(activation_layers[len(activation_layers) -1].shape)
			errors = find_error(self.weightsList, activation_layers[len(activation_layers) -1], self.class_to_angles(train_labels[k]))
			self.weightsList = update_weights(self.weightsList, activation_layers, errors)
			
			# print "error: "+str(np.array(errors,ndmin = 2).T.shape)
			errorList.append(np.array(errors,ndmin = 2).T/n_classes)
		return errorList

	def test_network(self, test_data, test_labels):
		errorsList = []
		output = []
		for k in range(len(test_data)):
			activation_layers = []
			data = np.array(test_data[k], ndmin = 2, dtype='complex128').T
			activation_layers.append(data)
			for i in range(len(self.weightsList)):
				if i == 0:
					activation_layers.append(activation(np.dot(self.weightsList[i], data) + self.biasesList[i]))
				else:
					activation_layers.append(activation(np.dot(self.weightsList[i], activation_layers[i]) + self.biasesList[i]))
			# print "testactivation: " + str(activation_layers[len(activation_layers) -1].shape)
			errorsList.append(find_error(self.weightsList, activation_layers[len(activation_layers)-1], self.class_to_angles(test_labels[k])))
			output.append({'label':test_labels[k], 'output': self.z_to_class(activation_layers[len(activation_layers)-1])})
		return (errorsList, output)
	pass

weightsList = [
	initweights(n_inputs, 64), # 784 * 64 weight matrix
	initweights(64, 32), # 64 * 32 weight matrix
	initweights(32, n_classes) # 32 * 10 weight matrix
	]

biasesList = [
	initbiases(64),
	initbiases(32),
	initbiases(n_classes)
	]

net = network(weightsList, biasesList, n_classes, periodicity)
net.status()

step = 1
while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = unroll_image(batch_x)
    err = net.train_network(batch_x, batch_y)
    if step % 50 == 0:
    	print "epoch.... " + str(step)
    # if step % display_step == 0:
    # 	print "loss= {:.6f}".format(np.sum(err)/len(err))
    step += 1
print("Optimization Finished!")

(testerr, out) = net.test_network(unroll_image(mnist.test.images[:100]), mnist.test.labels[:100])
# print("loss in test data = {:.6f}".format(np.sum(testerr)/len(testerr)))
print "--------------------------------------------------------------------"
for o in out:
	print o