import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sk


class MLPBN:
    def __init__(self, layers):
        assert isinstance(layers, list)
        layers = self.l_tuple(layers, 0)
        self.nn = {} ; self.d = len(layers)

        for l in range(self.d):
            self.nn['W%d' % l] = np.random.rand(layers[l][0], layers[l][1]) * np.sqrt(2. / layers[l][0])
            self.nn['b%d' % l] = np.zeros(layers[l][1])
            self.nn['gamma%d' % l] = 1 ; self.nn['beta%d' % l]
    
    def backprop(self, x, y ,o, k):
        g = {}
        de = o['i%d' % self.d] - y

        for l in range(self.d, -1, -1, -1):
            g['dbeta%d' % l] = np.sum(de, axis=0)
            g['dgamma%d' % l] = np.sum(()) 

    def forward(self, x):
        o = {'i0' : x}
        for l in range(self.d - 1):
            o['o%d' % l] = o['i%d' % l] @ self.nn['W%d' % l] + self.nn['b%d' % l]
            o['mean%d' % l], o['var%d' % l], o['i_hat%d' % l] = self.BN( o['o%d' % l])
            o['i%d' % (l + 1)] = self.relu(self.nn['gamma' % l] * o['i_hat%d' % l] + self.nn['beta%d' % l])
        
        return o

    def bn(self, x):
        mean = np.sum(x, axis=0) / x.shape[0]
        var = np.sum(np.power(x - mean, 2), axis=0) / x.shape[0]
        x_hat = (x - mean) * (var + 1e-8) ** (-1./2.)

        return mean, var, x_hat

    def l_tuple(self, layers, i):
        try:
            layers[i] = (layers[i], layers[i + 1]) ; return(self.l_tuple(layers, i + 1))
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
