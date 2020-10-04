import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sk

class MLP:
    def __init__(self, layers, a_foo, o_foo, reg, cost):
        assert isinstance(layers, list)
        self.layers = l_tuple(layers, 0)
        self.logits = {'r':self.relu, 's':self.sigmoid, 'so':self.softmax}
        self.regulizeres = {'l2': self.weight_decay}
        self.costs = {'co':self.cross_entropy}
        self.d = len(self.layers)
        self.h = []
        self.b = []

        if not a_foo in self.logits: self.f = self.relu
        else: self.f = self.logits[a_foo]
        if not o_foo in self.logits: self.f_o = self.softmax
        else: self.f_o = self.logits[o_foo] 
        if not reg in self.regulizeres: self.reg = self.weight_decay
        else: self.reg = self.regulizeres[reg]
        if not cost in self.costs: self.cost = self.cross_entropy
        else: self.cost = self.costs[cost]

        if self.f == self.relu:
            for l in self.layers:
                self.h.append(np.random.rand(l[0], l[1]) * np.sqrt(2./l[0]))
                self.b.append(np.zeros(l[1]))
        else: 
            for l in self.layers:
                self.h.append(np.random.rand(l[0], l[1]) * np.sqrt(6/(l[0] + l[1])))
                self.b.append(np.zeros(l[1]))

    def feed_forward(self, x):
        o = [x]
        for l in range(self.d - 1):
            o.append(o[l] @ self.h[l] + self.b[l])
            o[l + 1] = self.f(o[l + 1]) 
        o.append(o[self.d - 1] @ self.h[self.d - 1] + self.b[self.d - 1])
        o[self.d] = self.f_o(o[self.d])

        return o
    
    def gradient_descent(self, niter, x, y, e, wd=0.1):
        j = []
        for n in range(niter):
            p = np.random.choice(x)
            x_T, y_T = x[p], self.get_one_hot(y[p], x.shape[1])

            o = self.feed_forward(x_T)
            j.append(self.cost(o[self.d], y_T) + self.weight_decay(self.h, wd))

            de = o[self.d] - y_T
            for l in range(self.d-1, -1, -1):
                dw = o[l].T @ de
                db = de.flatten()
                de = de @ self.h[l].T
                self.h[l] -= e * (dw + wd * self.h[l])
                self.b[l] -= db * e
            
        return j

    def relu(self, x):
        return np.greater(x, 0).astype(int) * x

    def d_relu(self, x):
        return np.greater(x, 0).astype(int)

    def sigmoid(self, x):  
        return np.exp(-np.logaddexp(0, -x))

    def d_sigmoid(self, x):
        y = self.sigmoid(x)
        return y * (1 - y)

    def softmax(self, z):
        if not len(z.shape) == 2: z = np.array([z])
        s = np.array([np.max(z,axis=1)]).T
        e_z = np.exp(z - s)

        return e_z / np.array([np.sum(e_z, axis=1)]).T

    def cross_entropy(self, x, y):
        a = 10 ** -8
        n = x.shape[0]
        return -np.sum(y * np.log(x + (1 * a))) / n

    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])
    
    def weight_decay(self, h, wd):
        if self.h == h: 
            h = [np.power(l, 2) for l in h]
            return np.sum([np.sum(l) for l in h]) * wd / 2
        else: return wd * h

def l_tuple(layers, i):
    try:
        layers[i] = (layers[i], layers[i + 1]) ; return(l_tuple(layers, i + 1))
    except IndexError:
        layers.pop(i) ; return layers

nn = MLP([2,3,2], 'r', 'so', 'l2', 'co')
x = np.array([[1, 0],[0, 1], [0, 1], [1, 0]])
y = np.array([0, 1, 1, 0])
print (nn.feed_forward(x))