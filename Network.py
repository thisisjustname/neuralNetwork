import numpy as np
import scipy.special


class NeuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.synaptic_weight1 = np.random.normal(0.0, pow(self.inodes + 1, -0.5), (self.hnodes, self.inodes + 1))
        self.synaptic_weight2 = np.random.normal(0.0, pow(self.hnodes + 1, -0.5), (self.onodes, self.hnodes + 1))

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.vstack((np.array(inputs_list, ndmin=2).T, 1))

        answers = np.array(targets_list, ndmin=2).T

        hidden_layer = np.vstack((self.activation_function(np.dot(self.synaptic_weight1, inputs)), 1))

        outputs = self.activation_function(np.dot(self.synaptic_weight2, hidden_layer))

        output_errors = answers - outputs
        hidden_errors = np.dot(self.synaptic_weight2[0:, :self.hnodes].T, output_errors)

        self.synaptic_weight2 += self.lr * np.dot((output_errors * outputs * (1.0 - outputs)),
                                                  np.transpose(hidden_layer))
        self.synaptic_weight1 += self.lr * np.dot((hidden_errors * hidden_layer[0:self.hnodes] * (1.0 - hidden_layer[0:self.hnodes])),
                                                  np.transpose(inputs))

        return output_errors

    def query(self, inputs_list):
        inputs = np.vstack((np.array(inputs_list, ndmin=2).T, 1))

        hidden_layer = np.vstack((self.activation_function(np.dot(self.synaptic_weight1, inputs)), 1))

        outputs = self.activation_function(np.dot(self.synaptic_weight2, hidden_layer))

        return outputs
