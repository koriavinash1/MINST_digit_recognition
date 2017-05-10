import numpy as np
from additional_funcs import unroll_image, initbiases, initweights, update_weights, find_error, activation
from constants import batch_size, training_iters, n_inputs, n_classes, periodicity, display_step
import input_data
mnist = input_data.read_data_sets(one_hot=True, train_image_number = 1, test_image_number = 1)

class network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, categories, periodicity):
        self.wih = np.random.uniform(-1.0, 1.0, (input_nodes, hidden_nodes)) + 1j * np.random.uniform(-1.0, 1.0, (input_nodes, hidden_nodes))
        self.who = np.random.uniform(-1.0, 1.0, (hidden_nodes, output_nodes)) + 1j * np.random.uniform(-1.0, 1.0, (hidden_nodes, output_nodes))
        self.categories = categories
        self.periodicity = periodicity
        pass

    def z_to_class(self, z):
        angle = np.mod(np.angle(z) + 2*np.pi, 2*np.pi)
        p = np.floor (self.categories * self.periodicity * angle / (2*np.pi))
        p = np.mod(p, self.categories).T
        return p[0]

    def class_to_angles(self, c):
        angles = (c + 0.5 + (self.periodicity * np.arange(self.categories))) / (self.categories * self.periodicity) * 2 * np.pi
        return angles

    def status(self):
        print "periodicity: " + str(self.periodicity) + " categories: " + str(self.categories)
        print "weights input_hidden", self.wih
        print "weights hidden_output", self.who
        pass

    def train_network(self, train_data, train_labels):
        train_data.append(1.0)



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
                    nodes.append(activation(np.dot(self.weightsList[i], data) + self.biasesList[i]))
                else:
                    nodes.append(activation(np.dot(self.weightsList[i], nodes[i]) + self.biasesList[i]))
            # print "testactivation: " + str(nodes[len(nodes) -1].shape)
            errorsList.append(find_error(self.weightsList, nodes[len(nodes)-1], self.class_to_angles(test_labels[k])))
            output.append({'label':test_labels[k], 'output': self.z_to_class(nodes[len(nodes)-1])})
        return (errorsList, output)
    pass

weightsList = [
    initweights(n_inputs + 1, 1), # 784 + 1 * 64 weight matrix
    initweights(1, n_classes) # 32 * 10 weight matrix
    ]

net = network(weightsList, n_classes, periodicity)
net.status()

step = 1
while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = unroll_image(batch_x)
    err = net.train_network(batch_x, batch_y)
    if step % 50 == 0:
        print "epoch.... " + str(step)
    # if step % display_step == 0:
    #   print "loss= {:.6f}".format(np.sum(err)/len(err))
    step += 1
print("Optimization Finished!")

(testerr, out) = net.test_network(unroll_image(mnist.test.images[:100]), mnist.test.labels[:100])
# print("loss in test data = {:.6f}".format(np.sum(testerr)/len(testerr)))
print "----------------------------------------------------------------------------------"
for o in out:
    print o