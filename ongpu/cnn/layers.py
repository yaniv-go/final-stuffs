import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import copy

class SoftMaxLayer():
    def __init(self):
        self.mem = 0

    def forward(self, x):
        self.mem = self.softmax(x)
        return self.mem

    def softmax(self, z):
        if len(z.shape) < 2: z = cp.array([z])
        s = cp.array([cp.max(z, axis=1)]).T
        e_z = cp.exp(z - s)

        return e_z / cp.array([cp.sum(e_z, axis=1)]).T

    def backprop(self, y):
        de = y - self.mem
        self.mem - []
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
        return cp.maximum(0, x)
    
    def d_relu(self, x):
        return cp.greater(x, 0).astype(cp.int8)
    
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

        self.w = cp.random.rand(self.row, self.col) * cp.sqrt(2./self.col)
        self.b = cp.zeros(self.col)
        if self.prev_shape: self.forward = self.fprev_conv ; self.backprop = self.bprev_conv
        else: self.forward = self.fprev_fc ; self.backprop = self.bprev_fc

    def fprev_fc(self, x):
        self.mem = x
        o = cp.dot(x, self.w) + self.b

        return o

    def fprev_conv(self, x):
        n, c, w, h = x.shape
        x = x.reshape(n, c * h * w)
        self.mem = x
        o = cp.dot(x, self.w) + self.b

        self.prev_shape = (n, c, h, w)

        return o
    
    def bprev_fc(self, dx):
        dw = cp.dot(self.mem.T, dx)
        db = cp.sum(dx, axis=0)
        de = cp.dot(dx, self.w.T)

        self.mem = dw, db

        return de
        
    def bprev_conv(self, dx):
        dw = cp.dot(self.mem.T, dx)
        db = cp.sum(dx, axis=0)
        de = cp.dot(dx, self.w.T)

        self.mem = dw, db

        return de.reshape(self.prev_shape)

    def update(self, wd=0):
        self.w -= (self.mem[0] + wd * self.w)
        self.b -= (self.mem[1] + wd * self.b)

class MaxPool():
    def __init__(self, size=2, stride=2, paddding=0):
        self.mem = 0 
        self.ks = size
        self.s = stride
        self.p = paddding

    def forward(self, x):
        n, cp, hp, wp = x.shape
        c = cp
        h = int(((hp + 2 * self.p - self.ks) / self.s) + 1)
        w = int(((wp + 2 * self.p - self.ks) / self.s) + 1)

        x_reshaped = x.reshape(n * c, 1, hp, wp)\
        x_col = im2col(x_reshaped, self.ks, self.ks, self.p, self.s)

        max_idx = cp.argmax(x_col, axis=0)

        out = x_col[max_idx, rang(max_idx.size)]
        out = out.reshape(h, w, n, c)
        out = out.transpose(2, 3, 0, 1)

        self.mem = max_idx, x_col.shape, x.shape

        return out
    
    def backprop(self, do):
        maxi, shape, xshape = self.mem
        
        dx_col = cp.zeros(shape)

        do_f = do.transpose(2, 3, 0, 1).ravel()
        dx_col[maxi, range(maxi.size)] = do_f

        n, c, h, w = ishape

        dx = col2im(dx_col, (n * c, 1, h, w), self.ks, self.ks, self.p, self.s)
        dx = dx.reshape(ishape)

        self.mem = []

        return dx

    def update(self, wd=0):
        pass

class ConvLayer:
    def __init__(self, size=3, amount=2, pad=1, stride=1, channels=1):
        self.ks = size ; self.p = pad ; self.s = stride 
        self.a = amount ; self.c = channels

        self.k = cp.random.rand(self.a, self.c, self.ks, self.ks) * cp.sqrt(2./self.ks)
        self.b = cp.zeros((self.a, 1))
    
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
        db = 