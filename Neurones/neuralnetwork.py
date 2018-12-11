# coding: utf-8

from mnist import MNIST
import numpy as np
import math
import os


DATASETS_PREFIX    = '../Datasets/MNIST'
mndata             = MNIST(DATASETS_PREFIX)
TRAINING_IMAGES, TRAINING_LABELS  = mndata.load_training()
TESTING_IMAGES , TESTING_LABELS   = mndata.load_testing()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.dot(x.T, 1.0 - x)

class NeuralNetwork(object):
    """
    This is a 2-layer neural network.
    @_input   : input layer
    @_weights1: weights between layer 1 and layer 2      (matrix shape (input.shape[1], 4))
    @_weights2: weights between layer 2 and output layer (matrix shape (4, 1))
    @_y       : output
    @_output  : computed output
    @_alpha   : learning rate
    """
    def __init__(self, xshape, yshape):
        self._input     = None
        self._weights1  = np.random.rand(xshape, 4)
        self._weights2  = np.random.rand(4, 1)
        self._y         = None
        self._output    = np.zeros(yshape)
        self._alpha     = 1.0

    def Train(self, xs, ys):
        for i in range(len(xs)):
            self._input   = np.mat(xs[i])
            self._y       = np.mat(ys[i])
            #import pdb; pdb.set_trace()
            self.feedforward()
            self.backpropagation()

    def Predict(self, image):
        self._input = image
        self.feedforward()
        return self._output
    
    def feedforward(self):
        self._layer1 = sigmoid(np.dot(self._input, self._weights1))
        self._output = sigmoid(np.dot(self._layer1, self._weights2))

    def backpropagation(self):
        d_weights2 = np.dot(
            self._layer1.T,
            2 * (self._y - self._output) * sigmoid_derivative(self._output)
        )
        d_weights1 = np.dot(
            self._input.T,
            np.dot(
                2 * (self._y - self._output) * sigmoid_derivative(self._output),
                self._weights2.T
            ) * sigmoid_derivative(self._layer1)
        )
        self._weights1 += self._alpha * d_weights1
        self._weights2 += self._alpha * d_weights2

if __name__ == '__main__':
    neural_network = NeuralNetwork(len(TRAINING_IMAGES[0]), 1)
    print('* trainig neural network')
    neural_network.Train(TRAINING_IMAGES, TRAINING_LABELS)
    print('* testing neural network')
    count = 0
    for i in range(len(TESTING_IMAGES)):
        image       = np.mat(TESTING_IMAGES[i])
        expected    = TESTING_LABELS[i]
        prediction  = neural_network.Predict(image)[0,0]
        if i % 100 == 0: print(expected, prediction)
        if expected == prediction: count += 1
    print(f'* results: {count} / {len(TESTING_IMAGES)}')
