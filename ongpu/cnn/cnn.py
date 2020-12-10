from keras.datasets import mnist
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
        
        return x.reshape(-1, k, x.shape[1], x.shape[2], x.shape[3]), y.reshape(-1, k, y.shape[1])

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

    def get_learning_rate(self, e0, et, t, n):
        if (k := n / t) < 1: return e0 * (1 - k) + et * t
        return et

    def sgd(self, epochs, x, y, xv, yv, e0=0.01, t=100, et=0, wd=0.01, k=16):
        if et == 0: et = e0 / 100
        j, jv = [], []
        best = cp.inf

        for ep in range(epochs):
            e = self.get_learning_rate(e0, et, t, ep)

            xb, yb = self.get_batches(x, y, k)

            for n in range(xb.shape[0]):
                p = cp.random.randint(xb.shape[0] - 1)

                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                cost = self.cost(o, yt)
                if cost < best: 
                    best_model = copy.deepcopy(self.nn)
                    best = cost
                j.append(cost)

                self.back(o, yt)
                for l in self.nn:
                    l.mem = [e * x for x in l.mem]
                    l.update(wd * e)
            
            xv ,yv = self.get_batches(vx, vy, k)
            validate = self.cost(self.forward(vx), vy)
            jv.append(cp.sum(validate) / validate.shape[0])
    
        return j, jv

def get_one_hot(targets, nb_classes):
    res = cp.eye(nb_classes)[cp.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

(tx, ty), (vx, vy) = mnist.load_data()
tx = tx.astype('float32') / 255.
tx = tx.reshape(tx.shape[0], 1, tx.shape[1], tx.shape[2])
vx = vx.reshape(vx.shape[0], 1, vx.shape[1], vx.shape[2])

tx = cp.array(tx)
ty = cp.array(ty)
vx = cp.array(vx)
vy = cp.array(vy)

ty = get_one_hot(ty, 10)
vy = get_one_hot(vy, 10)

c = CNN()
c.add_conv_layer(3, 16, 1, 1)
c.add_relu_layer()
c.add_conv_layer(3, 32, 1, 1, 16)
c.add_relu_layer()
c.add_pool_layer()
c.add_fc_layer(6272, 100, 1)
c.add_relu_layer()
c.add_fc_layer(100, 10, 0)
c.add_softmax_layer()

j, jv = c.sgd(1, tx[:2000], ty[:2000], vx[:2000], vy[:2000], e0=1e-3, wd=1e-8, k=32)
fig, axs = plt.subplots(2)
axs[0].plot(range(len(j)), j)
axs[1].plot(range(len(jv)), jv)