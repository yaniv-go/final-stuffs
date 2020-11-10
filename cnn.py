from keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sk
import random


class SoftmaxLayer():
    def __init__(self):
        self.mem = 0

    def forward(self, x):
        self.mem = self.softmax(x)
        return self.mem
    
    def softmax(self, z):
        if len(z.shape) < 2: z = np.array([z])
        s = np.array([np.max(z,axis=1)]).T
        e_z = np.exp(z - s)

        return e_z / np.array([np.sum(e_z, axis=1)]).T
    
    def backprop(self, y):
        de = y - self.mem
        self.mem = []
        return de

    def update(self, wd=0):
        pass

class ReluLayer():
    def __init__(self):
        self.mem = 0

    def forward(self, x):
        self.mem = x
        return self.relu(x)

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return np.greater(x, 0).astype(int)

    def backprop(self, x):
        y = self.mem
        self.mem = []
        return self.d_relu(y) * x 
    
    def update(self, wd=0):
        pass

class Fc():
    def __init__(self, row, column, prev_shape=0):
        self.prev_shape = prev_shape
        self.mem = 0
        self.row = row
        self.col = column

        self.w = np.random.rand(self.row, self.col) * np.sqrt(2./self.col)
        self.b = np.zeros(self.col)
        if self.prev_shape: self.forward = self.ff ; self.backprop = self.backk
        else: self.forward = self.f ; self.backprop = self.back

    def f(self, x):
        self.mem = x
        o = x @ self.w + self.b
        return o

    def ff(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, c * h * w)
        self.mem = x
        o = x @ self.w 
        o += self.b

        self.prev_shape = (n, c, h, w)

        return o
    
    def back(self, dx):
        dw = self.mem.T @ dx
        db = np.sum(dx, axis=0)
        de = dx @ self.w.T

        self.mem = dw, db

        return de

    def backk(self, dx):
        dw = self.mem.T @ dx
        db = np.sum(dx, axis=0)
        de = dx @ self.w.T

        self.mem = dw, db

        return de.reshape(self.prev_shape)

    def update(self, wd=0):
        self.w -= (self.mem[0] + wd * self.w)
        self.b -= (self.mem[1] + wd * self.b)

class MaxPool():
    def __init__(self, size=2, stride=2, padding=0):
        self.mem = 0
        self.ks = size
        self.s = stride
        self.p = padding
    
    def forward(self, x):
        n, cp, hp, wp = x.shape
        c = cp
        h = int(((hp + 2 * self.p - self.ks) / self.s) + 1)
        w = int(((wp + 2 * self.p - self.ks) / self.s) + 1)

        x_reshaped = x.reshape(n * c, 1, hp, wp)
        x_col = im2col(x_reshaped, self.ks, self.ks, self.p, self.s)
    
        max_idx = np.argmax(x_col, axis=0)

        out = x_col[max_idx, range(max_idx.size)]
        out = out.reshape(h, w, n, c)
        out = out.transpose(2, 3, 0, 1)

        self.mem = max_idx, x_col.shape, x.shape

        return out

    def backprop(self, do): 
        mmax, shape, ishape = self.mem

        dx_col = np.zeros(shape)

        do_f = do.transpose(2, 3, 0, 1).ravel()
        dx_col[mmax, range(mmax.size)] = do_f

        n, c, h, w = ishape

        dx = col2im(dx_col, (n * c, 1, h, w), self.ks, self.ks, self.p, self.s)
        dx = dx.reshape(ishape)

        self.mem = []

        return dx

    def update(self, wd=0):
        pass

class ConvLayer:
    def __init__(self, size=3, amount=2,  pad=1, stride=1, channels=1):
        self.ks = size ; self.p = pad ; self.s = stride ; self.a = amount ; self.c = channels
        self.k = np.random.rand(self.a, self.c, self.ks, self.ks) * np.sqrt(2./self.ks)
        self.b = np.zeros((self.a, 1))

    def forward(self, x):
        n, cp, hp, wp = x.shape
        c = self.a
        h = int(((hp + 2 * self.p - self.ks) / self.s) + 1)
        w = int(((wp + 2 * self.p - self.ks) / self.s) + 1)
        
        k_col = self.k.reshape(self.a, -1)

        xcol = im2col(x, self.ks, self.ks, self.p, self.s)
        o = k_col @ xcol + self.b

        o = o.reshape(self.a, h, w, n)
        o = o.transpose(3, 0, 1, 2)

        self.mem = x.shape, xcol

        return o

    def backprop(self, do):
        db = np.sum(do, axis=(0, 2, 3))
        db = db.reshape(self.a, -1)

        do = do.transpose(1, 2, 3, 0).reshape(self.a, -1)
        dw = do @ self.mem[1].T 
        dw = dw.reshape(self.k.shape)

        dxcol = self.k.T @ do 
        dx = col2im(dxcol, self.mem[0], self.ks, self.ks, self.p, self.s)

        self.mem = dw, db

        return dx

    def update(self, wd):
        self.k -= (self.mem[0] + wd * self.k)
        self.b -=  (self.mem[1] + wd * self.b)

class CNN:
    def __init__(self):
        self.nn = []
        self.g = {ConvLayer: 2, MaxPool:0, ReluLayer:0, SoftmaxLayer:0, Fc:2}
    
    def get_batches(self, x, y, k):
        p = np.random.permutation(len(x))
        x, y = x[p], y[p]
        y, p = [self.get_one_hot(i, self.nn[-2].col) for i in y], len(x) // k

        xb, yb = np.append(x, x[:k - (len(x) - p * k)], axis=0), np.append(np.array(y), y[:k - (len(y) - p * k)], axis=0)
        return np.split(xb, p + 1), np.split(yb, p + 1)

    def add_softmax_layer(self):
        self.nn.append(SoftmaxLayer())

    def add_relu_layer(self):
        self.nn.append(ReluLayer())

    def add_fc_layer(self, row, column, x=0):
        self.nn.append(Fc(row, column, x))

    def add_pool_layer(self, size=2, stride=2):
        self.nn.append(MaxPool(size, stride))

    def add_conv_layer(self, size=3, amount=2, pad=1, stride=1, channels=1):
        self.nn.append(ConvLayer(size, amount, pad, stride, channels))

    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])

    def cost(self, x, y):
        a = 10 ** -8
        n = x.shape[0]
        return -np.sum(y * np.log(x + (1 * a))) / n

    def forward(self, x):
        for l in self.nn:
            x = l.forward(x)
        return x            
    
    def back(self, x, y):
        do = x - y
        for l in self.nn[-2::-1]:
            do = l.backprop(do)
        
        return do
    
    def sgd(self, epochs, x, y, e0=0.01, t=100, et=0, wd=0.01, k=16):
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
                j.append(self.cost(o, yt))

                self.back(o, yt)
                for l in self.nn:
                    g = [dx * e for dx in l.mem]
                    l.mem = g
                    l.update(wd)
            
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o, yt))
        
        return j, jv

    def sgd_momentum(self, epochs, x ,y, e0=0.01, t= 100, et=0, wd=0.01, k=16, m=0.9):
        if et == 0: et = e0 / 100
        rxb, ryb = self.get_batches(x, y, k)
        xv, yv = [], []
        j, jv = [], []
        vv = [[0] * self.g[type(x)] for x in self.nn]

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
                j.append(self.cost(o, yt))

                self.back(o, yt)
                for l, v in zip(self.nn, vv):
                    v = [m * c + e * dx for c, dx in zip(v, l.mem)]
                    l.mem = v
                    l.update(wd)
            
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o, yt))
        
        return j, jv

    def sgd_momentum_nesterov(self, epochs, x ,y, e0=0.01, t= 100, et=0, wd=0.01, k=16, m=0.9):
        if et == 0: et = e0 / 100
        rxb, ryb = self.get_batches(x, y, k)
        xv, yv = [], []
        j, jv = [], []
        vv = [[0] * self.g[type(x)] for x in self.nn]

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
                j.append(self.cost(o, yt))

                self.back(o, yt)
                for l, v in zip(self.nn, vv):
                    v = [m * (m * c + e * dx) + e * dx for c, dx in zip(v, l.mem)]
                    l.mem = v
                    l.update(wd)
            
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o, yt))
        
        return j, jv

    def rmsprop(self, epochs, x, y, e=0.01, wd=0.01, k=32, d=0.9)
        

    def get_learning_rate(self, e0, et, t, n):
        if (k := n / t) < 1: return e0 * (1 - k) + et * t
        return et


def get_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = int((H + 2 * padding - field_height) / stride + 1)
  out_width = int((W + 2 * padding - field_width) / stride + 1)

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)

def im2col(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_indices(x.shape, field_height, field_width, padding,
                                stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C  , -1)
    return cols

def col2im(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_indices(x_shape, field_height, field_width, padding,
                                stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

(tx, ty), (vx, vy) = mnist.load_data()
tx = tx.astype('float32') / 255.
print (tx[0].shape)
tx = tx.reshape(tx.shape[0], 1, tx.shape[1], tx.shape[2])

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

j, jv = c.rmsprop(1, tx[:6000], ty[:6000], e=0.01, wd=1e-5)
fig, axs = plt.subplots(2)
axs[0].plot(range(len(j)), j)
axs[1].plot(range(len(jv)), jv)
plt.show()