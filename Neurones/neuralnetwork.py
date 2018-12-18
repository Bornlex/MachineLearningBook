# coding: utf-8

from mnist import MNIST
import numpy as np
import math
import os
import pdb
import multiprocessing, joblib


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
    return e_x / np.sum(e_x)

def d_softmax(x):
    tmp = x.reshape((-1,1))
    return np.diag(x) - np.dot(tmp, tmp.T)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1.0 - (np.tanh(x).T * np.tanh(x))

def normalize(image):
    return image / 255.0

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
        ### NEURAL NETWORK PARAMETERS ###
        self._hidden_layers = 3
        self._neurones_nb   = 50
        self._input         = None
        self._weights_in    = np.random.randn(xshape, self._neurones_nb)
        self._weights_hi    = [np.random.randn(self._neurones_nb, self._neurones_nb)] * (self._hidden_layers - 1)
        self._weights_out   = np.random.randn(self._neurones_nb, yshape)
        self._weights       = [self._weights_in] + self._weights_hi + [self._weights_out]
        self._y             = np.mat(np.zeros(yshape))
        self._output        = np.mat(np.zeros(yshape))
        self._layers        = []
        ### COMPUTATIONAL PARAMETERS ###
        self._learning_rate = 0.01
        self._epoch         = 1

    def Train(self, xs, ys):
        for j in range(self._epoch):
            self._train(xs, ys, j)

    def Predict(self, image):
        self._input = normalize(image)
        out = self.feedforward()
        return out

    def _train(self, xs, ys, epoch):
        print(f'* epoch {epoch} / {self._epoch}')
        for i in range(len(xs)):
            if i % 5000 == 0: print(f'{i} / {len(xs)}')
            self._input = normalize(np.mat(xs[i]))
            self._y[0, ys[i]] = 1
            self.feedforward()
            self.backpropagation()
            self._y[0, ys[i]] = 0
    
    def feedforward(self):
        current_layer = self._input.copy()
        self._layers = []
        for index in range(self._hidden_layers):
            current_layer = tanh(np.dot(current_layer, self._weights[index]))
            self._layers.append(current_layer)
        self._output = softmax(np.dot(current_layer, self._weights[self._hidden_layers]))
        return self._output

    def backpropagation(self):
        err = 2 * (self._y - self._output) * d_softmax(self._output)
        d_weights = [None] * len(self._weights)
        d_weights[-1] = self._layers[-1].T * err
        wei = None
        for index in range(self._hidden_layers):
            out = self._layers[-(index + 2)] if (-index - 2) >= -self._hidden_layers else self._input
            wei = self._weights[-(index + 1)]
            err = err * wei.T
            d_weights[-(index + 2)] = d_tanh(out.T) * err
            """
            There is something to fix here...
            d_tanh should not be used like that.
            """
        self._weights = [self._weights[i] + (self._learning_rate * d_weights[i]) for i in range(len(self._weights))]

if __name__ == '__main__':
    neural_network = NeuralNetwork(len(TRAINING_IMAGES[0]), 10)
    print('* training neural network')
    neural_network.Train(TRAINING_IMAGES, TRAINING_LABELS)
    print('* testing neural network')
    count = 0
    import pdb; pdb.set_trace()
    for i in range(len(TESTING_IMAGES)):
        image       = np.mat(TESTING_IMAGES[i])
        expected    = TESTING_LABELS[i]
        prediction  = neural_network.Predict(image)
        if i % 100 == 0: print(expected, prediction)
    #print(f'* results: {count} / {len(TESTING_IMAGES)}')
