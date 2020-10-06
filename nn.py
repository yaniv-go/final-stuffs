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
            p = np.random.choice(x.shape[0])
            x_T, y_T = x[p], self.get_one_hot(y[p], self.h[self.d - 1].shape[1])

            o = self.feed_forward(x_T)
            j.append(self.cost(o[self.d], y_T) + self.weight_decay(self.h, wd))
            o = [np.array([i]) if len(i.shape) < 2 else i for i in o]

            de = o[self.d] - y_T
            for l in range(self.d-1, -1, -1):                
                dw = o[l].T @ de
                db = de.flatten()
                de = de @ self.h[l].T
                self.h[l] -= e * (dw + wd * self.h[l])
                self.b[l] -= db * e
            
        return j
    
    def sgd(self, niter, x, y, e0=0.01, t=500, et=0, wd=0.01, k=32):
        if et == 0: et = e0 / 100
        xb, yb = self.get_batches(x, y, k)
        j = []        
        
        for n in range(niter):
            p = np.random.choice(len(xb))
            xt, yt = xb[p], yb[p]

            o = self.feed_forward(xt)
            j.append(self.cost(o[self.d], yt) + self.weight_decay(self.h, wd))

            e = self.get_learning_rate(e0, et, t, n)
            de = o[self.d] - yt
            for l in range(self.d - 1, -1, -1):
                dw = o[l].T @ de / k
                db = np.sum(de, axis=0) / k
                de = de @ self.h[l].T
                self.h[l] -= e * (dw + wd * self.h[l])
                self.b[l] -= db * e
        
        return j

    def sgd_with_momentum(self, niter, x, y, e0=0.01, et=0, t=500, wd=0.01, k=32, m=0.5):
        if et == 0: et = e0 / 100
        xb, yb = self.get_batches(x, y, k)
        v = np.zeros((32, 2))
        j = []

        for n in range(niter):
            p = np.random.choice(len(xb))
            xt, yt = xb[p], yb[p]

            o = self.feed_forward(xt)
            j.append(self.cost(o[self.d], yt) + self.weight_decay(self.h, wd))

            e = self.get_learning_rate(e0, et, t, n)
            de = o[self.d] - yt ; v = m*v - de * e
            for l in range(self.d - 1, -1, -1):
                dw = o[l].T @ de / k
                db = np.sum(de, axis=0) / k
                de = de @ self.h[l].T
                self.h[l] -= e * (dw + wd * self.h[l])
                self.b[l] -= db * e

        return j

    def sgd_with_nesterov_momentum(self, niter, x, y, e0=0.01, et=0, t=500, wd=0.01, k=32, m=0.5):
        if et == 0: et = e0 / 100
        xb, yb = self.get_batches(x, y, k) 
        v = np.zeros((32, 2))
        j = []

        for n in range(niter):
            p = np.random.choice(len(xb))
            xt, yt = xb[p], yb[p]

            o = self.feed_forward(xt)
            j.append(self.cost(o[self.d], yt) + self.weight_decay(self.h, wd))

            e = self.get_learning_rate(e0, et, t, n)
            de = o[self.d] - yt
            v = m*v + e*de 
            de += v
            for l in range(self.d - 1, -1, -1):
                dw = o[l].T @ de / k
                db = np.sum(de, axis=0) / k
                de = de @ self.h[l].T
                self.h[l] -= e * (dw + wd * self.h[l])
                self.b[l] -= db * e
        
        return j

    def get_batches(self, x, y, k):
        p = np.random.permutation(len(x))
        x, y = x[p], y[p]
        y, p = [self.get_one_hot(i, self.h[self.d - 1].shape[1]) for i in y], len(x) // k

        xb, yb = np.append(x, x[:k - (len(x) - p * k)], axis=0), np.append(np.array(y), y[:k - (len(y) - p * k)], axis=0)
        return np.split(xb, p + 1), np.split(yb, p + 1)
    
    def get_learning_rate(self, e0, et, t, n):
        if (k := n / t) < 1: return e0 * (1 - k) + et * t
        return et

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

nn = MLP([20, 14, 13, 10, 8 ,2], 'r', 'so', 'l2', 'co')
x, y = sk.make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2
                           , n_repeated=0, n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=1.0)

j = nn.sgd_with_nesterov_momentum(2000, x, y, e0=0.002, et=0.00005, m=0.9, t=300)
plt.plot(range(len(j)), j)
plt.show() 