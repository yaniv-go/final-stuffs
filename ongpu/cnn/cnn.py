from keras.datasets import mnist
from layers import *
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sk
import random
import copy
import cProfile
import os

class CNN:
    def __init__(self):
        self.nn = []
        self.g = {ConvLayer: 2, MaxPool:0, ReluLayer:0, SoftMaxLayer:0, Fc:2, BN_layer:2}

    def get_batches(self, x, y, k): 
        p = cp.random.permutation(x.shape[0])
        x, y = x[p], y[p]

        if not x.shape[0] % k == 0:
            x = cp.concatenate((x, x[:k - x.shape[0] % k]))
            y = cp.concatenate((y, y[:k - x.shape[0] % k]))
        
        return x.reshape(-1, k, x.shape[1], x.shape[2], x.shape[3]), y.reshape(-1, k, y.shape[1])

    def add_bn_layer(self, exp_shape):
        self.nn.append(BN_layer(exp_shape))

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
        if (n / t) < 1: return e0 * (1 - n / t) + et * t
        return et

    def sgd(self, epochs, x, y, xv, yv, e0=0.01, t=100, et=0, wd=0.01, k=16):
        if et == 0: et = e0 / 100
        xv, yv = self.get_batches(xv, yv, k)
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
                    self.best_model = copy.deepcopy(self.nn)
                    best = cost
                j.append(cost)

                self.back(o, yt)
                for l in self.nn:
                    l.mem = [e * x for x in l.mem]
                    l.update(wd * e)
        
            # make validate run on batches
            validate = [self.cost(self.forward(xv[1]), yv[1]), self.cost(self.forward(xv[0]), yv[0])]
            for a, b in zip(xv[2:], yv[2:]):
                c = self.cost(self.forward(a), b)
                validate.append(c)
            
            jv.append(np.sum(validate) / len(validate))
    
        return j, jv

    def sgd_momentum(self, epochs, x, y, xv, yv, e0=0.01, t=100, et=0, wd=0.01, k=16, m=9):
        if et == 0: et = e0 / 100
        xv, yv = self.get_batches(xv, yv, k)
        vv = [[0] * self.g[type(x)] for x in self.nn]
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
                    self.best_model = copy.deepcopy(self.nn)
                    best = cost
                j.append(cost)

                self.back(o, yt)
                for l, v in zip(self.nn, vv):
                    v = [m * c + e * dx for c, dx in zip(v, l.mem)]
                    l.mem = v
                    l.update(wd * e)

            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o, yt))

        return j, jv

    def sgd_momentum_nesterov(self, epochs, x, y, xv, yv, e0=0.01, t=100, et=0, wd=0.01, k=16, m=9):
        if et == 0: et = e0 / 100
        xv, yv = self.get_batches(xv, yv, k)
        vv = [[0] * self.g[type(x)] for x in self.nn]
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
                    self.best_model = copy.deepcopy(self.nn)
                    best = cost
                j.append(cost)

                self.back(o, yt)
                for l, v in zip(self.nn, vv):
                    v = [m *  (m * c + e * dx) + e * dx for c, dx in zip(v, l.mem)]
                    l.mem = v
                    l.update(wd * e)

            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o, yt))

        return j, jv

    def rmsprop(self, epochs, x, y, xv, yv, e=0.01, wd=0.01, k=32, d=0.9):
        if et == 0: et = e / 100
        xv, yv = self.get_batches(xv, yv, k)
        rr = [[0] * self.g[type(x)] for x in self.nn]
        j, jv = [], []
        best = cp.inf

        for ep in range(epochs):
            xb, yb = self.get_batches(x, y, k)

            for n in range(xb.shape[0]):
                p = cp.random.randint(xb.shape[0] - 1)

                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                cost = self.cost(o, yt)
                if cost < best: 
                    self.best_model = copy.deepcopy(self.nn)
                    best = cost
                j.append(cost)

                self.back(o, yt)
                for l, r in zip(self.nn, rr):
                    r = [e * (1. / cp.sqrt(1e-8 + (d * c + (1 - d) * dx * dx))) * dx for c, dx in zip(r, l.mem)]
                    l.mem = r
                    l.update(wd * e)

            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o, yt))

        return j, jv

    def rmsprop_momentum(self, epochs, x, y, xv, yv, e=0.01, wd=0.01, k=32, d=0.9, m=0.9):
        if et == 0: et = e / 100
        xv, yv = self.get_batches(xv, yv, k)
        rr = [[0] * self.g[type(x)] for x in self.nn]
        vv = [[0] * self.g[type(x)] for x in self.nn]
        j, jv = [], []
        best = cp.inf

        for ep in range(epochs):
            xb, yb = self.get_batches(x, y, k)

            for n in range(xb.shape[0]):
                p = cp.random.randint(xb.shape[0] - 1)

                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                cost = self.cost(o, yt)
                if cost < best: 
                    self.best_model = copy.deepcopy(self.nn)
                    best = cost
                j.append(cost)

                self.back(o, yt)
                for l, r in zip(self.nn, rr):
                    r = [e * (1. / cp.sqrt(1e-8 + (d * c + (1 - d) * dx * dx))) * dx for c, dx in zip(r, l.mem)]
                    v = [m * c + e * dx for c, dx in zip(v, r)]
                    l.mem = v
                    l.update(wd * e)

            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o, yt))

        return j, jv

    def adam(self, epochs, x, y, xv, yv, e=0.01, wd=0.01, k=32, d=0.999, m=0.9):
        if et == 0: et = e / 100
        xv, yv = self.get_batches(xv, yv, k)
        rr = [[0] * self.g[type(x)] for x in self.nn]
        ss = [[0] * self.g[type(x)] for x in self.nn]
        j, jv = [], []
        best = cp.inf

        t = 0

        for ep in range(epochs):
            xb, yb = self.get_batches(x, y, k)

            for n in range(xb.shape[0]):
                p = cp.random.randint(xb.shape[0] - 1)

                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                cost = self.cost(o, yt)
                if cost < best: 
                    self.best_model = copy.deepcopy(self.nn)
                    best = cost
                j.append(cost)

                self.back(o, yt)
                t += 1

                for l, r in zip(self.nn, rr):
                    s = [m * c + (1 - m) * dx for c, dx in zip(s, l.mem)]
                    r = [d * c + (1 - d) * dx * dx for c, dx in zip(r, l.mem)]
                    
                    x = 1 - cp.power(m, t)
                    sh = [c / x for c in s]
                    
                    x = 1 - cp.power(d, t)
                    rh = [c / x for c in r]

                    l.mem = [(y / (cp.sqrt(x) + 1e-9)) * e for y, x in zip(sh, rh)] 
                    l.update(wd * e)

            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o, yt))

        return j, jv

    def adam_momentum(self, epochs, x, y, xv, yv, e=0.01, wd=0.01, k=32, d=0.999, m=0.9):
        xv, yv = self.get_batches(xv, yv, k)
        rr = [cp.array([0] * self.g[type(x)]) for x in self.nn]
        ss = [cp.array([0] * self.g[type(x)]) for x in self.nn]
        
        j, jv = [], []
        best = cp.inf

        t = 0

        for ep in range(epochs):
            xb, yb = self.get_batches(x, y, k)

            for n in range(xb.shape[0]):
                p = cp.random.randint(xb.shape[0] - 1)

                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                cost = self.cost(o, yt)
                if cost < best: 
                    self.best_model = copy.deepcopy(self.nn)
                    best = cost
                j.append(cost)

                self.back(o, yt)
                t += 1

                for l, i in zip(self.nn, range(len(ss))):
                    ss[i] = [m * c + (1 - m) * dx for c, dx in zip(ss[i], l.mem)]
                    rr[i] = [d * c + (1 - d) * dx * dx for c, dx in zip(rr[i], l.mem)]

                    a = 1 - cp.power(m, t)
                    sh = [m * (c / a) + (1-m) * dx / a for c, dx in zip(ss[i], l.mem)]
                    
                    a = 1 - cp.power(d, t)
                    rh = [c / a for c in rr[i]]

                    l.mem = [(y / (cp.sqrt(x) + 1e-9)) * e for y, x in zip(sh, rh)] 
                    l.update(wd * e)

            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o, yt))

        return j, jv

def get_one_hot(targets, nb_classes):
    res = cp.eye(nb_classes)[cp.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def get_mnist():
        
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

    return tx, ty, vx, vy

c = CNN()
c.add_conv_layer(3, 16, 1, 1)
c.add_bn_layer((16, 28, 28))
c.add_relu_layer()
c.add_pool_layer()
c.add_conv_layer(3, 16, 1, 1, 16)
c.add_bn_layer((16, 14, 14))
c.add_relu_layer()
c.add_pool_layer()
c.add_fc_layer(784, 100, 1)
c.add_bn_layer((100,))
c.add_relu_layer()
c.add_fc_layer(100, 10, 0)
c.add_softmax_layer()

#cProfile.run('c.sgd(1, tx, ty, vx, vy, e0=1e-3, wd=1e-8, k=2500)')

cProfile.run('j, jv = c.adam_momentum(100, tx, ty, vx, vy, e=3e-3, wd=0, k=1000)')
fig, axs = plt.subplots(2)
axs[0].plot(range(len(j)), j)
axs[1].plot(range(len(jv)), jv)

plt.show()
