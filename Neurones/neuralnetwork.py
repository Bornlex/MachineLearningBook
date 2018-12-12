# coding: utf-8

from mnist import MNIST
import numpy as np
import math
import os
import pdb


DATASETS_PREFIX    = '../Datasets/MNIST'
mndata             = MNIST(DATASETS_PREFIX)
TRAINING_IMAGES, TRAINING_LABELS  = mndata.load_training()
TESTING_IMAGES , TESTING_LABELS   = mndata.load_testing()

### UTILS

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x.T * (1 - x)
    #return np.dot(x.T, 1.0 - x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#def d_softmax(x):
#    tmp = x.reshape((-1,1))
#    return np.diag(x) - np.dot(tmp, tmp.T)

def d_softmax(x):
    #This function has not yet been tested.
    return x.T * (1 - x)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - x.T * x

def normalize(image):
    return image / (255.0 * 0.99 + 0.01)

### !UTILS

class NeuralNetwork(object):
    """
    This is a 3-layer neural network (1 hidden layer).
    @_input   : input layer
    @_weights1: weights between input layer and hidden layer  (matrix shape (input.shape[1], 4))
    @_weights2: weights between hidden layer and output layer (matrix shape (4, 1))
    @_y       : output
    @_output  : computed output
    @_alpha   : learning rate
    """
    def __init__(self, xshape, yshape):
        self._neurones_nb = 20
        self._input       = None
        self._weights1    = np.random.randn(xshape, self._neurones_nb)
        self._weights2    = np.random.randn(self._neurones_nb, yshape)
        self._y           = np.mat(np.zeros(yshape))
        self._output      = np.mat(np.zeros(yshape))
        self._alpha1      = 0.01
        self._alpha2      = 0.01
        self._function    = sigmoid
        self._derivative  = d_sigmoid
        self._epoch       = 1

    def Train(self, xs, ys):
        for j in range(self._epoch):
            for i in range(len(xs)):
                self._input = normalize(np.mat(xs[i]))
                self._y[0, ys[i]] = 1
                self.feedforward()
                self.backpropagation()
                self._y[0, ys[i]] = 0

    def Predict(self, image):
        self._input = normalize(image)
        self.feedforward()
        return self._output
    
    def feedforward(self):
        self._layer1 = self._function(np.dot(self._input, self._weights1))
        self._output = self._function(np.dot(self._layer1, self._weights2))

    def backpropagation(self):
        d_weights2 = np.dot(
            self._layer1.T,
            2 * (self._y - self._output) * self._derivative(self._output)
        )
        d_weights1 = np.dot(
            self._input.T,
            np.dot(
                2 * (self._y - self._output) * self._derivative(self._output),
                self._weights2.T
            ) * self._derivative(self._layer1)
        )
        self._weights1 += self._alpha1 * d_weights1
        self._weights2 += self._alpha2 * d_weights2

if __name__ == '__main__':
    neural_network = NeuralNetwork(len(TRAINING_IMAGES[0]), 10)
    print('* training neural network')
    neural_network.Train(TRAINING_IMAGES, TRAINING_LABELS)
    print('* testing neural network')
    count = 0
    for i in range(len(TESTING_IMAGES)):
        image       = np.mat(TESTING_IMAGES[i])
        expected    = TESTING_LABELS[i]
        prediction  = neural_network.Predict(image)
        if i % 100 == 0: print(expected, prediction)
    #print(f'* results: {count} / {len(TESTING_IMAGES)}')
