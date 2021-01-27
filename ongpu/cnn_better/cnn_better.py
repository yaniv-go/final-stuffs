import layers_better as layers
import numpy as np
import cupy as cp
import copy
import sys
import os

class CNN:
    def __init__(self, input_shape, loss='cross-entropy', optimizer='nadam'):
        optimizers = {'nadam' : self.adam_momentum, 'sgd' : self.sgd_nesterov_momentum}
        losses = {'cross-entropy' : self.cross_entropy}
        assert optimizer in optimizers, 'incorrect optimizer'
        assert isinstance(input_shape, tuple), 'insert input shape tuple'
        assert len(input_shape) == 3, 'insert tuple of correct length'
        for num in input_shape:
            assert isinstance(num, int), 'insert valid integer dimension'
            assert num > 0, 'insert valid positive dimensions'

        self.curr_output = input_shape
        self.fit = optimizers[optimizer]
        self.cost = losses[loss]
        self.optimizer = optimizer
        self.nn = []
    
    def adam_momentum(self, epochs, tx, ty, vx, vy, e, d1=0.9, d2=0.999, wd=0):
        train_loss, validation_loss = [], []
        t = 0
    
    def sgd_nesterov_momentum(self, tx, ty, vx, vy, e, d1=0.9, wd=0):
        train_loss, validation_loss = [], []
        t = 0
    
    def cross_entropy(self, o, y):
        n = o.shape[0]
        return -cp.sum(y * cp.log2(o + 1e-10)) / n
    