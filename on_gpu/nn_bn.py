import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import os

class MLPBN:
    def __init__(self, layers):
        assert isinstance(layers, list)
        layers = self.l_tuple(layers, 0)
        self.nn = {} ; self.d = len(layers)

        for l in range(self.d):
            self.nn['W%d' % l] = cp.random.rand(layers[l][0], layers[l][1]) * cp.sqrt(2. / layers[l][0])
            self.nn['b%d' % l] = cp.zeros(layers[l][1])
            if l == self.d - 1: break
            self.nn['gamma%d' % l] = cp.ones(layers[l][1]) ; self.nn['beta%d' % l] = cp.zeros(layers[l][1])

    def rmsprop_momentum(self, epochs, x, y, e=0.01, wd=1e-4, k=32, d=0.9, m=0.9):
        j, jv = [], []
        r, v = {}, {}
        
        for i in self.nn.keys():
            r[i], v[i] = 0, 0

        for ep in range(epochs):
            xb, yb = self.get_batches(x, y, k)

            if ((p := len(xb) // 5) >= 1):
                xv, yv = xb[:p], yb[:p]
                xb, yb = xb[p:], yb[p:]
            else:
                xv, yv = xb[:1], yb[:1]
                xb, yb = xb[1:], yb[1:]

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
            
            g['gamma%d' % l] = cp.sum(do * o['ohat%d' % l], axis=0)
            g['beta%d' % l] = cp.sum(do, axis=0)
            g['W%d' % l] = (o['i%d' % l].T @ di) / k
            g['b%d' % l] = cp.sum(di, axis=0) / k
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
        mean = cp.sum(x, axis=0) / x.shape[0]
        var = cp.sum(cp.power(x - mean, 2), axis=0) / x.shape[0]
        x_hat = (x - mean) * (1 / cp.sqrt(var + 1e-8))

        return mean, var, x_hat

    def l_tuple(self, layers, i):
        try:
            layers[i] = (layers[i], layers[i + 1]) ; return(self.l_tuple(layers, i + 1))
        except IndexError:
            layers.pop(i) ; return layers

    def get_batches(self, x, y ,k):
        p = cp.random.permutation(len(x))
        x, y = x[p], y[p]
        p = len(x) // k

        xb, yb = cp.concatenate((x, x[:k - (len(x) - p * k)]), axis=0), cp.concatenate((y, y[:k - (len(y) - p * k)]), axis=0)
        return cp.split(xb, p + 1), cp.split(yb, p + 1)

    def d_relu(self, x):
        return cp.greater(x, 0).astype(cp.int8)

    def relu(self, x):
        return cp.maximum(0, x)

    def softmax(self, z): 
        if not len(z.shape) == 2: z = cp.array([z])
        s = cp.array([cp.max(z, axis=1)]).T
        e_z = cp.exp(z - s)

        return e_z / cp.array([cp.sum(e_z, axis=1)]).T

    def cost(self, x, y):
        a = 10 ** -8
        n = x.shape[0]
        return -cp.sum(y * cp.log(x + (1 * a))) / n
    
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '1'

nn = MLPBN([78, 220, 180, 180, 150, 150, 150, 58])
x, y = cp.load('C:\\Users\\yaniv\\Documents\\GitHub\\final-stuffs\\taki\\input-data.npy'), cp.load('C:\\Users\\yaniv\\Documents\\GitHub\\final-stuffs\\taki\\output-data.npy') 

j, jv = nn.rmsprop_momentum(100, x, y, e=5e-3, wd=0)
fig, axs = plt.subplots(2)
axs[0].plot(range(len(j)), j)
axs[1].plot(range(len(jv)), jv)
plt.show()

print (nn.forward(x[0])['o%d' % (nn.d - 1)])
print (y[0])