import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sk
import random

class MLP:
    def __init__(self, layers, batchnorm=0):
        assert isinstance(layers, list)
        layers = self.l_tuple(layers, 0)
        self.nn = {}
        self.bn = batchnorm
        self.d = len(layers)

        for i in range(self.d):
            self.nn['W%d' % i] = np.random.rand(layers[i][0], layers[i][1]) * np.sqrt(2. / layers[i][0])
            self.nn['b%d' % i] = np.zeros(layers[i][1])
            if batchnorm == 1: self.nn['gamma%d' % i], self.nn['beta%d' % i] = 1, 0

    def forward(self, x):
        o = [x]
        for l in range(self.d - 1):
            o.append(self.relu(o[l] @ self.nn['W%d' % l] + self.nn['b%d' % l]))
        o.append(self.softmax(o[self.d - 1] @ self.nn['W%d' % (self.d - 1)] + self.nn['b%d' % (self.d - 1)]))

        return o
    
    def forward_bn(self, x):
        o = {'i0' : x}
        for l in range(self.d - 1):
            o['o%d' % l] = o['i%d' % l] @ self.nn['W%d' % l] + self.nn['b%d' % l]
            o['mean%d' % l], o['var%d' % l], o['i_hat%d' % l] = self.BN( o['o%d' % l])
            o['i%d' % (l + 1)] = self.relu(self.nn['gamma' % l] * o['i_hat%d' % l] + self.nn['beta%d' % l])
        
        return o

    def BN(self, x):
        mean = np.sum(x, axis=0) / x.shape[0]
        var = np.sum(np.power(x - mean, 2), axis=0) / x.shape[0]
        x_hat = (x - mean) / np.sqrt(var + 1e-8)

        return mean, var, x_hat

    def backprop(self, x, y, o, k):
        g = {}
        de = o[self.d] - y

        for l in range(self.d - 1, -1, -1):
            g['W%d' % l] = (o[l].T @ de) / k
            g['b%d' % l] = np.sum(de, axis=0) / k
            de = de @ self.nn['W%d' % l].T
            de = de * self.d_relu(o[l])

        return g

    def l_tuple(self, layers, i):
        try:
            layers[i] = (layers[i], layers[i + 1]) ; return(l_tuple(layers, i + 1))
        except IndexError:
            layers.pop(i) ; return layers

    def get_batches(self, x, y, k):
        p = np.random.permutation(len(x))
        x, y = x[p], y[p]
        y, p = [self.get_one_hot(i, self.nn['W%d' % (self.d - 1)].shape[1]) for i in y], len(x) // k

        xb, yb = np.append(x, x[:k - (len(x) - p * k)], axis=0), np.append(np.array(y), y[:k - (len(y) - p * k)], axis=0)
        return np.split(xb, p + 1), np.split(yb, p + 1)
    
    def get_learning_rate(self, e0, et, t, n):
        if (k := n / t) < 1: return e0 * (1 - k) + et * t
        return et

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return np.greater(x, 0).astype(int)

    def softmax(self, z):
        if not len(z.shape) == 2: z = np.array([z])
        s = np.array([np.max(z,axis=1)]).T
        e_z = np.exp(z - s)

        return e_z / np.array([np.sum(e_z, axis=1)]).T

    def cost(self, x, y):
        a = 10 ** -8
        n = x.shape[0]
        return -np.sum(y * np.log(x + (1 * a))) / n

    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])

    def sgd(self, epochs, x, y, e0=0.01, t=100, et=0, wd=0.01, k=32):
        if et == 0: et = e0 / 100
        rxb, ryb = self.get_batches(x, y, k)
        xv, yv = [], []
        j, jv = [], []

        for ep in range(epochs):
            e = self.get_learning_rate(e0, et, t, ep)

            xb, yb = rxb.copy(), ryb.copy()
            xv, yv = [], []

            if ((p := len(xb) // 5) >= 1):
                for i in range(p):
                    xv.append(xb.pop()) ; yv.append(yb.pop())
            else: xv = xb.pop() ; yv = yb.pop() 

            for n in range(len(xb)):
                p = np.random.choice(len(xb))
                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                j.append(self.cost(o[self.d], yt))

                g = self.backprop(xt, yt, o, k)
                for l in range(self.d):
                    self.nn['W%d' % l] -= e * (g['W%d' % l] + wd * self.nn['W%d' % l])
                    self.nn['b%d' % l] -= e * g['b%d' % l]
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o[self.d], yt))

        return j, jv
    
    def sgd_momentum(self, epochs, x, y, e0=0.01, t=100, et=0, wd=0.01, k=32, m=0.9):
        if et == 0: et = e0 / 100
        rxb, ryb = self.get_batches(x, y, k)
        xv, yv = [], []
        j, jv = [], []
        v = {}
        for i in self.nn.keys():
            v[i] = 0

        for ep in range(epochs):
            e = self.get_learning_rate(e0, et, t, ep)

            xb, yb = rxb.copy(), ryb.copy()
            xv, yv = [], []

            if ((p := len(xb) // 5) >= 1):
                for i in range(p):
                    xv.append(xb.pop()) ; yv.append(yb.pop())
            else: xv = xb.pop() ; yv = yb.pop() 

            for n in range(len(xb)):
                p = np.random.choice(len(xb))
                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                j.append(self.cost(o[self.d], yt))

                g = self.backprop(xt, yt, o, k)
                for i in g.keys():
                    v[i] = m * v[i] + e * g[i]
                for l in range(self.d):
                    self.nn['W%d' % l] -= ((e * wd * self.nn['W%d' % l]) + v['W%d' % l])
                    self.nn['b%d' % l] -= v['b%d' % l]
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o[self.d], yt))
        return j, jv

    def sgd_momentum_nesterov(self, epochs, x, y, e0=0.01, t=100, et=0, wd=0.01, k=32, m=0.9):
        if et == 0: et = e0 / 100
        rxb, ryb = self.get_batches(x, y, k)
        j, jv = [], []
        v = {}
        for i in self.nn.keys():
            v[i] = 0

        for ep in range(epochs):
            e = self.get_learning_rate(e0, et, t, ep)

            xb, yb = rxb.copy(), ryb.copy()
            xv, yv = [], []

            if ((p := len(xb) // 5) >= 1):
                for i in range(p):
                    xv.append(xb.pop()) ; yv.append(yb.pop())
            else: xv = xb.pop() ; yv = yb.pop() 

            for n in range(len(xb)):
                p = np.random.choice(len(xb))
                xt, yt = xb[p], yb[p]
                
                nn_c = self.nn.copy()
                for i in self.nn.keys():
                    self.nn[i] -= m * v[i] 
                o = self.forward(xt)
                j.append(self.cost(o[self.d], yt))

                g = self.backprop(xt, yt, o, k)
                self.nn = nn_c
                for i in g.keys():
                    v[i] = m * v[i] + e * g[i]
                for l in range(self.d):
                    self.nn['W%d' % l] -= ((e * wd * self.nn['W%d' % l]) + v['W%d' % l])
                    self.nn['b%d' % l] -= v['b%d' % l]
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o[self.d], yt))
        return j, jv       
    
    def rmsprop(self, epochs, x, y, e=0.01, wd=0.01, k=32, d=0.9):
        rxb, ryb = self.get_batches(x, y, k)
        j, jv = [], []
        r = {}
        for i in self.nn.keys():
            r[i] = 0

        for ep in range(epochs):
            xb, yb = rxb.copy(), ryb.copy()
            xv, yv = [], []

            if ((p := len(xb) // 5) >= 1):
                for i in range(p):
                    xv.append(xb.pop()) ; yv.append(yb.pop())
            else: xv = xb.pop() ; yv = yb.pop() 
                       
            for n in range(len(xb)):
                p = np.random.choice(len(xb))
                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                j.append(self.cost(o[self.d], yt))

                g = self.backprop(xt, yt, o, k)
                for i in g.keys():
                    r[i] = d * r[i] + (1 - d) * g[i] * g[i]
                    g[i] = e * (1 / np.sqrt(1e-8 + r[i])) * g[i]
                for l in range(self.d):
                    self.nn['W%d' % l] -= e * (wd * self.nn['W%d' % l]) + g['W%d' % l]
                    self.nn['b%d' % l] -= g['b%d' % l]
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o[self.d], yt))
                
        return j, jv

    def rmsprop_momentum(self, epochs, x, y ,e=0.01, wd=0.01, k=32, d=0.9, m=0.9):
        rxb, ryb = self.get_batches(x, y, k)
        j, jv = [], []
        r, v = {}, {}
        
        for i in self.nn.keys():
            r[i], v[i] = 0, 0

        for ep in range(epochs):
            xb, yb = rxb.copy(), ryb.copy()
            xv, yv = [], []

            if ((p := len(xb) // 5) >= 1):
                for i in range(p):
                    xv.append(xb.pop()) ; yv.append(yb.pop())
            else: xv = xb.pop() ; yv = yb.pop()

            for n in range(len(xb)):
                p = np.random.choice(len(xb))
                xt, yt = xb[p], yb[p]

                nn_c = self.nn.copy()
                for i in self.nn.keys():
                    self.nn[i] -= m * v[i] 

                o = self.forward(xt)
                j.append(self.cost(o[self.d], yt))

                g = self.backprop(xt, yt, o, k)
                self.nn = nn_c
                for i in g.keys():
                    r[i] = d * r[i] + (1 - d) * g[i] * g[i]
                    v[i] = v[i] * m + e * (((r[i] + 1e-8) ** (-1./2.)) * g[i])
                for l in range(self.d):
                    self.nn['W%d' % l] -= e * (wd * self.nn['W%d' % l]) + v['W%d' % l]
                    self.nn['b%d' % l] -= v['b%d' % l]
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o[self.d], yt))
                
        return j, jv

    def adam(self, epoch, x, y, e=0.01, wd=0.01, k=32, m=0.9, d=0.999):
        rxb, ryb = self.get_batches(x, y, k)
        j, jv = [], []
        r, s = {}, {}
        s_hat, r_hat = {}, {}

        for i in self.nn.keys():
            r[i], s[i] = 0, 0
            s_hat[i], r_hat[i] = 0, 0

        t = 0 

        for ep in range(epoch):
            xb, yb = rxb.copy(), ryb.copy()
            xv, yv = [], []

            if ((p := len(xb) // 5) >= 1):
                for i in range(p):
                    xv.append(xb.pop()) ; yv.append(yb.pop())
            else: xv = xb.pop() ; yv = yb.pop()

            for n in range(len(xb)):
                p = np.random.choice(len(xb))
                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                j.append(self.cost(o[self.d], yt))

                g = self.backprop(xt, yt, o, k)
                t += 1

                for i in g.keys():
                    s[i] = m * s[i] + (1 - m) * g[i]
                    r[i] = d * r[i] + (1 - d) * g[i] * g[i]
                    s_hat[i] = s[i] / (1 - np.power(m, t))
                    r_hat[i] = r[i] / (1 - np.power(d, t))
                    g[i] = s_hat[i] / (np.sqrt(r_hat[i]) + 1e-9)

                for l in range(self.d):
                    self.nn['W%d' % l] -=  e * (self.nn['W%d' % l] * wd + g['W%d' % l])
                    self.nn['b%d' % l] -= e * g['b%d' % l]

            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o[self.d], yt))

        return j, jv

    def adam_momentum(self, epoch, x, y, e=0.01, wd=0.01, k=32, m=0.9, d=0.999):
        rxb, ryb = self.get_batches(x, y, k)
        j, jv = [], []
        r, s = {}, {}
        s_hat, r_hat = {}, {}

        for i in self.nn.keys():
            r[i], s[i] = 0, 0
            s_hat[i], r_hat[i] = 0, 0

        t = 0 

        for ep in range(epoch):
            xb, yb = rxb.copy(), ryb.copy()
            xv, yv = [], []

            if ((p := len(xb) // 5) >= 1):
                for i in range(p):
                    xv.append(xb.pop()) ; yv.append(yb.pop())
            else: xv = xb.pop() ; yv = yb.pop()

            for n in range(len(xb)):
                p = np.random.choice(len(xb))
                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                j.append(self.cost(o[self.d], yt))

                g = self.backprop(xt, yt, o, k)
                t += 1

                for i in g.keys():
                    s[i] = m * s[i] + (1 - m) * g[i]
                    r[i] = d * r[i] + (1 - d) * g[i] * g[i]
                    s_hat[i] = m * (s[i] / (1 - np.power(m, t))) + ((1 - m) * g[i] / (1 - np.power(m, t)))
                    r_hat[i] = r[i] / (1 - np.power(d, t))
                    g[i] = s_hat[i] / (np.sqrt(r_hat[i]) + 1e-9)

                for l in range(self.d):
                    self.nn['W%d' % l] -=  e * (self.nn['W%d' % l] * wd + g['W%d' % l])
                    self.nn['b%d' % l] -= e * g['b%d' % l]

            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o[self.d], yt))

        return j, jv
        
def l_tuple(layers, i):
    try:
        layers[i] = (layers[i], layers[i + 1]) ; return(l_tuple(layers, i + 1))
    except IndexError:
        layers.pop(i) ; return layers
nn = MLP([20, 22, 20, 16, 10 ,2])
x, y = sk.make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2
                           , n_repeated=0, n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=1.0)

j, jv = nn.rmsprop_momentum(100, x, y, e=1e-4, wd=1e-4)
fig, axs = plt.subplots(2)
axs[0].plot(range(len(j)), j)
axs[1].plot(range(len(jv)), jv)
plt.show()