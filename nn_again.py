import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MLPBN:
    def __init__(self, layers):
        assert isinstance(layers, list)
        layers = self.l_tuple(layers, 0)
        self.nn = {} ; self.d = len(layers)

        for l in range(self.d):
            self.nn['W%d' % l] = np.random.rand(layers[l][0], layers[l][1]) * np.sqrt(2. / layers[l][0])
            self.nn['b%d' % l] = np.zeros(layers[l][1])
            if l == self.d - 1: break
            self.nn['gamma%d' % l] = np.ones(layers[l][1]) ; self.nn['beta%d' % l] = np.zeros(layers[l][1])
    
    def rmsprop_momentum(self, epochs, x, y, e=0.01, wd=1e-4, k=32, d=0.9, m=0.9):
        j, jv = [], []
        r, v = {}, {}
        
        for i in self.nn.keys():
            r[i], v[i] = 0, 0

        for ep in range(epochs):
            xb, yb = self.get_batches(x, y, k)
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
                j.append(self.cost(o['o%d' % (self.d - 1)], yt))

                g = self.backprop(xt, yt, o, k)
                self.nn = nn_c
                for i in g.keys():
                    r[i] = d * r[i] + (1 - d) * g[i] * g[i]
                    v[i] = v[i] * m + e * (((r[i] + 1e-8) ** (-1./2.)) * g[i])
                for l in range(self.d - 1):
                    self.nn['W%d' % l] -= v['W%d' % l] + (e * wd * self.nn['W%d' % l])
                    self.nn['b%d' % l] -= v['b%d' % l]
                    self.nn['gamma%d' % l] -= v['gamma%d' % l]
                    self.nn['beta%d' % l] -= v['beta%d' % l]
                self.nn['W%d' % (self.d - 1)] -= v['W%d' % (self.d - 1)] + (e * wd * self.nn['W%d' % (self.d - 1)])
                self.nn['b%d' % (self.d - 1)] -= v['b%d' % (self.d - 1)]
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o['o%d' % (self.d - 1)], yt))
        
        return j, jv
                

    def rmsprop(self, epochs, x, y, e=0.01, wd=1e-5, k=32, d=0.9):
        rxb, ryb = self.get_batches(x, y, k)
        j, jv = [], []
        r = {}
        for i in self.nn.keys():
            r[i] = 0

        for ep in range(epochs):
            xb, yb = rxb.copy(), ryb.copy()
            xv, yv= [], []

            if ((p := len(xb) // 5) >= 1):
                for i in range(p):
                    xv.append(xb.pop()) ; yv.append(yb.pop())
            else: xv = xb.pop() ; yv = yb.pop()

            for n in range(len(xb)):
                p = np.random.choice(len(xb))
                xt, yt = xb[p], yb[p]

                o = self.forward(xt)
                j.append(self.cost(o['o%d' % (self.d - 1)], yt))

                g = self.backprop(xt, yt, o, k)
                for i in g.keys():
                    r[i] = d * r[i] + (1- d) * g[i] * g[i]
                    g[i] = e * (1e-8 + r[i]) ** (-1./2.) * g[i]
                for l in range(self.d - 1):
                    self.nn['W%d' % l] -= g['W%d' % l] + (e * wd * self.nn['W%d' % l])
                    self.nn['b%d' % l] -= g['b%d' % l]
                    self.nn['gamma%d' % l] -= g['gamma%d' % l]
                    self.nn['beta%d' % l] -= g['beta%d' % l]
                self.nn['W%d' % (self.d - 1)] -= g['W%d' % (self.d - 1)] + (e * wd * self.nn['W%d' % (self.d - 1)])
                self.nn['b%d' % (self.d - 1)] -= g['b%d' % (self.d - 1)]
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o['o%d' % (self.d - 1)], yt))

        return j, jv

    def sgd_momentum_nesterov(self, epochs, x, y, e0=0.01, t=100, et=0, wd=0.01, k=32, m=0.9):
        if et == 0: e0 / 100
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
                j.append(self.cost(o['o%d' % (self.d - 1)], yt))

                g = self.backprop(xt, yt, o, k)
                self.nn = nn_c
                for i in g.keys():
                    v[i] = m * v[i] + e * g[i]
                for l in range(self.d - 1):
                    self.nn['W%d' % l] -= v['W%d' % l] + (e * wd * self.nn['W%d' % l])
                    self.nn['b%d' % l] -= v['b%d' % l]
                    self.nn['gamma%d' % l] -= v['gamma%d' % l]
                    self.nn['beta%d' % l] -= v['beta%d' % l]
                self.nn['W%d' % (self.d - 1)] -= v['W%d' % (self.d - 1)] + (e * wd * self.nn['W%d' % (self.d - 1)])
                self.nn['b%d' % (self.d - 1)] -= v['b%d' % (self.d - 1)]
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o['o%d' % (self.d - 1)], yt))
        
        return j, jv
                

    def sgd_momentum(self, epochs, x, y ,e0=0.01, t=100, et=0, wd=1e-8, k=32, m=0.9):
        if et==0: et = e0 / 100
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
                j.append(self.cost(o['o%d' % (self.d - 1)], yt))

                g = self.backprop(xt, yt, o, k)
                for i in g.keys():
                    v[i] = m * v[i] + e * g[i]
                for l in range(self.d - 1):
                    self.nn['W%d' % l] -= v['W%d' % l] + (e * wd * self.nn['W%d' % l])
                    self.nn['b%d' % l] -= v['b%d' % l]
                    self.nn['gamma%d' % l] -= v['gamma%d' % l]
                    self.nn['beta%d' % l] -= v['beta%d' % l]
                self.nn['W%d' % (self.d - 1)] -= v['W%d' % (self.d - 1)] + (e * wd * self.nn['W%d' % (self.d - 1)])
                self.nn['b%d' % (self.d - 1)] -= v['b%d' % (self.d - 1)]
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o['o%d' % (self.d - 1)], yt))

        return j, jv

    def sgd(self, epochs, x, y, e0=0.01, t=100, et=0, wd=1e-3, k=32):
        if et==0: et = e0 / 100
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
                j.append(self.cost(o['o%d' % (self.d - 1)], yt))

                g = self.backprop(xt, yt, o, k)
                for l in range(self.d - 1):
                    self.nn['W%d' % l] -= e * (g['W%d' % l] + wd * self.nn['W%d' % l])
                    self.nn['b%d' % l] -= e * g['b%d' % l]
                    self.nn['gamma%d' % l] -= e * g['gamma%d' % l]
                    self.nn['beta%d' % l] -= e * g['beta%d' % l]
                self.nn['W%d' % (self.d - 1)] -= e * (g['W%d' % (self.d - 1)] + wd * self.nn['W%d' % (self.d - 1)])
                self.nn['b%d' % (self.d - 1)] -= e * g['b%d' % (self.d - 1)]
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o['o%d' % (self.d - 1)], yt))

        return j, jv

    def backprop(self, x, y ,o, k):
        g = {} ; d = self.d - 1
        do = o['o%d' % d] - y

        g['W%d' % d] = (o['i%d' % d].T @ do) / k
        g['b%d' % d] = np.sum(do, axis=0) / k
        do = do @ self.nn['W%d' % d].T
        do = do * self.d_relu(o['i%d' % d])

        for l in range(d - 1, -1, -1):
            dohat = do * self.nn['gamma%d' % l]
            dvar = dohat * (o['o%d' % l] - o['m%d' % l]) * (-1. / 2.) * (o['v%d' % l] + 1e-8) ** (-3. / 2.)
            dm = dohat * (((o['v%d' % l] + 1e-8) ** (-1./2)) * -1) + dvar * ((-2 * (o['o%d' % l] - o['m%d' % l])) / k)
            di = dohat * (o['v%d' % l] + 1e-8) ** (-1./2) + dvar * (2 * (o['o%d' % l] - o['m%d' % l]) /k) + dm / k
            
            g['gamma%d' % l] = np.sum(do * o['ohat%d' % l], axis=0)
            g['beta%d' % l] = np.sum(do, axis=0)
            g['W%d' % l] = (o['i%d' % l].T @ di) / k
            g['b%d' % l] = np.sum(di, axis=0) / k
            do = di @ self.nn['W%d' % l].T
            do = do * self.d_relu(o['i%d' % l])
        return g
    
    def forward(self, x):
        o = {'i0' : x}
        d = self.d - 1
        for l in range(d):
            o['o%d' % l] = o['i%d' % l] @ self.nn['W%d' % l] + self.nn['b%d' % l]
            o['m%d' % l], o['v%d' % l], o['ohat%d' % l] = self.bn(o['o%d' % l])
            o['i%d' % (l + 1)] = self.relu(self.nn['gamma%d' % l] * o['ohat%d' % l] + self.nn['beta%d' % l])
        o['o%d' % d] = self.softmax(o['i%d' % d] @ self.nn['W%d' % d] + self.nn['b%d' % d])

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
        p = len(x) // k

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

nn = MLPBN([78, 120, 100, 90,80 ,58])
x, y = np.load('C:\\Users\\yaniv\\Documents\\GitHub\\final-stuffs\\input-data.npy'), np.load('C:\\Users\\yaniv\\Documents\\GitHub\\final-stuffs\\output-data.npy') 

j, jv = nn.rmsprop_momentum(100, x, y, e=4e-4, wd=0)
fig, axs = plt.subplots(2)
axs[0].plot(range(len(j)), j)
axs[1].plot(range(len(jv)), jv)
plt.show()