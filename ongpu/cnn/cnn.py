from layers import *
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sk
import random
import copy

class CNN:
    def __init__(self):
        self.nn = []
        self.g = {ConvLayer: 2, MaxPool:0, ReluLayer:0, SoftMaxLayer:0, Fc:2}

    def get_batches(self, x, y, k): 
        p = cp.random.permutation(x.shape[0])
        x, y = x[p], y[p]

        if not (a:= x.shape[0] % k) == 0:
            x = cp.concatenate((x, x[:k - a]))
            y = cp.concatenate((y, y[:k - a]))
        
        return x.reshape(-1, k, x.shape[1]), y.reshape(-1, k, y.shape[1])

    def add_softmax_layer(self):
        self.nn.append(SoftMaxLayer())

    def add_relu_layer(self):
        self.nn.append(ReluLayer())

    def add_fc_layer(self, row, column, x=0):
        self.nn.append(Fc(row, column, x))

    def add_pool_layer(self, size=2, stride=2):
        self.nn.append(MaxPool(size, stride))

    def add_conv_layer(self, size=3, amount=2, pad=1, stride=1, channels=1):
        self.nn.append(ConvLayer(size, amount, pad, stride, channels))

    def cost(self, x, y):
        a = 10 ** -8
        n = x.shape[0]
        return -cp.sum(y * cp.log(x + (1 * a))) / n
    
    def forward(self, x):
        for l in self.nn:
            x = l.forward(x)
        return x    

    def back(self, x, y):
        do = x - y
        for l in self.nn[-2::-1]:
            do = l.backprop(do)
        
        return do
    
    def test(self, x, y):
        yes, no = 0, 0
        o = self.forward(x)
        for ot, yt in zip(cp.argmax(o, axis=1), y):
            if ot == yt: yes += 1
            else: no += 1

        print('yes: ', yes)
        print ('no: ', no)
        print ('per: ', yes/ x.shape[0])

    
a = CNN()