import numpy as np
import scipy.special


class NeuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.synaptic_weight1 = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.synaptic_weight2 = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        answers = np.array(targets_list, ndmin=2).T

        hidden_layer = self.activation_function(np.dot(self.wih, inputs))

        final_outputs = self.activation_function(np.dot(self.who, hidden_layer))

        outputs = self.activation_function(np.dot(self.who, hidden_layer))

        output_errors = answers - outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.synaptic_weight2 += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_layer))
        self.synaptic_weight1 += self.lr * np.dot((hidden_errors * hidden_layer * (1.0 - hidden_layer)), np.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_layer = self.activation_function(np.dot(self.wih, inputs))

        outputs = self.activation_function(np.dot(self.who, hidden_layer))

        return outputs
