import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sk


class MLP:
    def __init__(self, layers, batchnorm=0):
        assert isinstance(layers, list)
        layers = self.l_tuple(layers, 0)
        self.nn = {}
        self.BN = batchnorm
        self.d = len(layers)

        for i in range(self.d):
            self.nn['W%d' % i] = np.random.rand(layers[i][0], layers[i][1]) * np.sqrt(2. / layers[i][0])
            self.nn['b%d' % i] = np.zeros(layers[i][1])
            if batchnorm == 1: self.nn['BN%d' % i] = [1, 0]

    def forward(self, x):
        o = [x]
        if self.BN == 0:
            for l in range(self.d - 1):
                o.append(self.relu(o[l] @ self.nn['W%d' % l] + self.nn['b%d' % l]))
            o.append(self.softmax(o[self.d - 1] @ self.nn['W%d' % (self.d - 1)] + self.nn['b%d' % (self.d - 1)]))
        
        return o

    def backprop(self, x, y, o, k):
        g = {}
        de = o[self.d] - y

        for l in range(self.d - 1, -1, -1):
            g['W%d' % l] = (o[l].T @ de) / k
            g['b%d' % l] = np.sum(de, axis=0) / k
            de = de @ g['W%d' % l].T
        
        return g

    def l_tuple(self, layers, i):
        try:
            layers[i] = (layers[i], layers[i + 1]) ; return(l_tuple(layers, i + 1))
        except IndexError:
            layers.pop(i) ; return layers

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
    
    def weight_decay(self, h, wd):
        if self.h == h: 
            h = [np.power(l, 2) for l in h]
            return np.sum([np.sum(l) for l in h]) * wd / 2
        else: return wd * h

class MLP1:
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
    
    def backprop(self, x, y, o, k, e):
        g = {}
        de = o[self.d] - y

        for l in range(self.d - 1, -1, -1):
            g['W%d' % l] = (o[l].T @ de) / k
            g['b%d' % l] = np.sum(de, axis=0) / k
            de = de @ self.h[l].T
        
        return g

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
            g = self.backprop(xt, yt, o, k, e)
            for l in range(self.d):
                self.h[l] -= e * (g['W%d' % l] + wd * self.h[l])
                self.b[l] -= e * g['b%d' % l]
        
        return j

    def sgd_with_momentum(self, niter, x, y, e0=0.01, et=0, t=500, wd=0.01, k=32, m=0.9):
        if et == 0: et = e0 / 100
        xb, yb = self.get_batches(x, y, k)
        v = {}
        j = []

        for i in range(self.d):
            v['W%d' % i] = 0
            v['b%d' % i] = 0

        for n in range(niter):
            p = np.random.choice(len(xb))
            xt, yt = xb[p], yb[p]

            o = self.feed_forward(xt)
            j.append(self.cost(o[self.d], yt) + self.weight_decay(self.h, wd))

            e = self.get_learning_rate(e0, et, t, n)
            g = self.backprop(xt, yt, o, k, e)
            for l in range(self.d):
                self.h[l] -= v['W%d' % l] + wd * e * self.h[l]
                self.b[l] -= v['b%d' % l] 
            for i in g.keys(): 
                v[i] = m * v[i] + e * g[i]

        return j

    def sgd_with_nesterov_momentum(self, niter, x, y, e0=0.01, et=0, t=500, wd=0.01, k=32, m=0.5):
        if et == 0: et = e0 / 100
        xb, yb = self.get_batches(x, y, k) 
        v = {}
        j = []

        for i in range(self.d):
            v['W%d' % i] = 0
            v['b%d' % i] = 0

        for n in range(niter):
            p = np.random.choice(len(xb))
            xt, yt = xb[p], yb[p]

            o = self.feed_forward(xt)
            j.append(self.cost(o[self.d], yt) + self.weight_decay(self.h, wd))

            e = self.get_learning_rate(e0, et, t, n)
            g = self.backprop(xt, yt, o, k, e)
            for i in g.keys(): 
                v[i] = m * v[i] + e * g[i]      
            for l in range(self.d):
                self.h[l] -= v['W%d' % l] + e * (wd * self.h[l] + g['W%d' % l] )
                self.b[l] -= v['b%d' % l] + e * g['b%d' % l]        
        
        return j

    def rmsprop(self, niter, x, y, e=0.01, wd=0.01, k=32, d=0.9):
        xb, yb = self.get_batches(x, y, k) 
        r = {}
        j = []

        for i in range(self.d):
            r['W%d' % i] = 0
            r['b%d' % i] = 0

        for n in range(niter):
            p = np.random.choice(len(xb))
            xt, yt = xb[p], yb[p]

            o = self.feed_forward(xt)
            j.append(self.cost(o[self.d], yt))

            g = self.backprop(xt, yt, o, k, e)
            for i in g.keys():
                r[i] = d * r[i] + (1 - d) * g[i] * g[i]
                g[i] = g[i] * (1 / np.sqrt(1e-6 + r[i])) * e
            for l in range(self.d):
                self.h[l] -= g['W%d' % l] + e * wd * self.h[l]
                self.b[l] -= g['b%d' % l] 

        return j

    def rmsprop_with_momentum(self, niter, x, y, e=0.001, wd=0.01, k=32, m=0.9, d=0.9):
        xb, yb = self.get_batches(x, y, k) 
        r = {}
        v = {}
        j = []

        for i in range(self.d):
            r['W%d' % i] = 0
            r['b%d' % i] = 0
            v['W%d' % i] = 0
            v['b%d' % i] = 0
        
        for n in range(niter):
            p = np.random.choice(len(xb))
            xt, yt = xb[p], yb[p]

            o = self.feed_forward(xt)
            j.append(self.cost(o[self.d], yt))
            g = self.backprop(xt, yt, o, k, e)
            for i in g.keys():
                r[i] = r[i] * d + (1 - d) * g[i] * g[i]
                g[i] = g[i] * (e / np.sqrt(1e-6 + r[i]))
                v[i] = v[i] * m + g[i]
            for l in range(self.d):
                self.h[l] -= v['W%d' % l] + e * wd * self.h[l] + g['W%d' % l] 
                self.b[l] -= v['b%d' % l] + g['b%d' % l]        
            
        return j

    def adam(self, niter, x, y, e=0.01, wd=0.01, k=32, d1=0.9, d2=0.999):
        xb, yb = self.get_batches(x, y, k) 
        r = {} ; r_hat = {}
        s = {} ; s_hat = {}
        j = []

        for i in range(self.d):
            r['W%d' % i] = 0 ; r_hat['W%d' % i] = 0
            r['b%d' % i] = 0 ; r_hat['b%d' % i] = 0
            s['W%d' % i] = 0 ; s_hat['W%d' % i] = 0 
            s['b%d' % i] = 0 ; s_hat['b%d' % i] = 0
        
        for n in range(niter):
            p = np.random.choice(len(xb))
            xt, yt = xb[p], yb[p]

            o = self.feed_forward(xt)
            j.append(self.cost(o[self.d], yt))
            g = self.backprop(xt, yt, o, k, e)
            for i in g.keys():
                s[i] = d1*s[i] + (1 - d1) * g[i]
                r[i] = d2*r[i] + (1 - d2) * g[i] * g[i]
                s_hat[i] = s[i] / (1 - d1 ** (n+1))
                r_hat[i] = r[i] / (1 - d2 ** (n+1))
            for l in range(self.d):
                self.h[l] -= e * (s_hat['W%d' % l] / np.sqrt(r_hat['W%d' % l] + 1e-8) + wd * self.h[l])
                self.b[l] -= e * (s_hat['b%d' % l] / np.sqrt(r_hat['b%d' % l] + 1e-8))
        
        return j

    def adam_with_momentum(self, niter, x, y, e=0.01, wd=0.01, k=32, m=0.9, d=0.999):
        xb, yb = self.get_batches(x, y, k)
        r = {} ; r_hat = {}
        v = {} ; v_hat = {}
        j = []

        for i in range(self.d):
            r['W%d' % i] = 0 ; r_hat['W%d' % i] = 0
            r['b%d' % i] = 0 ; r_hat['b%d' % i] = 0
            v['W%d' % i] = 0 ; v_hat['W%d' % i] = 0 
            v['b%d' % i] = 0 ; v_hat['b%d' % i] = 0

        for n in range(niter):
            p = np.random.choice(len(xb))
            xt, yt = xb[p], yb[p]

            o = self.feed_forward(xt)
            j.append(self.cost(o[self.d], yt))
            g = self.backprop(xt,yt, o, k, e)
            for i in g.keys():
                v[i] = m * v[i] + (1-m) * g[i]
                r[i] = d * r[i] + (1-d) * g[i] * g[i]
                v_hat[i] = ((m * v[i]) / (1 - m ** (n + 1))) + (((1 - m) * g[i]) / (1 - m ** (n + 1))) 
                r_hat[i] = (d * r[i]) / (1 - d ** (n + 1))
            for l in range(self.d):
                self.h[l] -= e * (v_hat['W%d' % l] * (1 / np.sqrt(r_hat['W%d' % l] + 1e-8)) + wd * self.h[l])
                self.b[l] -= e * (v_hat['b%d' % l] * (1 / np.sqrt(r_hat['b%d' % l] + 1e-8)))
            
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

nn = MLP([20, 22, 20, 16, 10 ,2])
x, y = sk.make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2
                           , n_repeated=0, n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=1.0)

xt, yt = x[:5], [nn.get_one_hot(i, 2) for i in y[:5]]
o = nn.forward(xt)
print(nn.backprop(xt, yt, o, 5))