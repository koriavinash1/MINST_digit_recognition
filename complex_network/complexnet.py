import numpy as np
from additional_funcs import unroll_image, initbiases, initweights, update_weights, find_error, activation
from constants import batch_size, training_iters, n_inputs, n_classes, periodicity, display_step
import input_data
mnist = input_data.read_data_sets(one_hot=False, train_image_number = 60000, test_image_number = 1000)

class network:
	def __init__(self, weightsList, categories, periodicity):
		self.weightsList = weightsList
		self.categories = categories
		self.periodicity = periodicity
		pass

	def z_to_class(self, z):
		angle = np.mod(np.angle(z) + 2*np.pi, 2*np.pi)
		p = int(np.floor (self.categories * self.periodicity * angle / (2*np.pi)))
		p = np.mod(p, self.categories)
		return p

	def class_to_angles(self, c):
		angles = (c + 0.5 + (self.categories * np.arange(self.periodicity))) / (self.categories * self.periodicity) * 2 * np.pi
		return angles

	def status(self):
		print "periodicity: " + str(self.periodicity) + " categories: " + str(self.categories)
		pass

	def train_network(self, train_data, train_labels):
		errorList = []
		# print "traindata length ", len(train_data)
		for k in range(len(train_data)):
			nodes = []
			train_data[k] = np.concatenate((train_data[k], [1.0]), axis=0)
			data = train_data[k].T
			nodes.append(data)
			for i in range(len(self.weightsList)):
				# print "w: " + str(weightsList[i].shape) + " data: "+ str(data.shape) + " act: "+ str(nodes[i].shape)
				if i == 0:
					nodes.append(activation(np.dot(self.weightsList[i], data)))
				else:
					nodes.append(activation(np.dot(self.weightsList[i], nodes[i])))
			# print np.subtract(nodes[len(nodes) - 1], self.class_to_angles(train_labels[k]))
			# print "trainactivation: " + str(nodes[len(nodes) -1].shape)
			errors = find_error(self.weightsList, nodes	, self.class_to_angles(train_labels[k]))
			# print errors
			self.weightsList = update_weights(self.weightsList, nodes, errors)
			# print "error: "+str(np.array(errors,ndmin = 2).T.shape)
			errorList.append(np.sum(errors[0])/10)
		return errorList

	def test_network(self, test_data, test_labels):
		errorsList = []
		output = []
		for k in range(len(test_data)):
			nodes = []
			test_data[k] = np.concatenate((test_data[k], [1.0]), axis=0)
			data = test_data[k].T
			nodes.append(data)
			for i in range(len(self.weightsList)):
				if i == 0:
					nodes.append(activation(np.dot(self.weightsList[i], data)))
				else:
					nodes.append(activation(np.dot(self.weightsList[i], nodes[i])))
			# print "testactivation: " + str(nodes[len(nodes) -1].shape)
			errorsList.append(np.sum(find_error(self.weightsList, nodes[len(nodes)-1], self.class_to_angles(test_labels[k]))[0])/10)
			output.append({'label':test_labels[k], 'output': self.z_to_class(nodes[len(nodes)-1])})
		return (errorsList, output)
	pass

weightsList = [
	initweights(n_inputs + 1, 10), # 784 + 1 * 64 weight matrix
	initweights(10, n_classes) # 32 * 10 weight matrix
	]

net = network(weightsList, n_classes, periodicity)
net.status()

step = 1
while step * batch_size <= training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = unroll_image(batch_x)
    batch_y = batch_y.T[1].T
    err = net.train_network(batch_x, batch_y)
    # print err
    # if step % 50 == 0:
    # 	print "epoch.... " + str(step)
    if step % display_step == 0:
    	print "loss= {:.6f}".format(np.abs(np.sum(err))/len(err))
    step += 1
print("Optimization Finished!")

(testerr, out) = net.test_network(unroll_image(mnist.test.images[:100]), mnist.test.labels[:100].T[1].T)
print("loss in test data = {:.6f}".format(np.abs(np.sum(testerr))/len(testerr)))
print "--------------------------------------------------------------------"
for o in out[:10]:
	print o