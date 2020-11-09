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

    def update(self):
        self.w -= self.mem[0]
        self.b -= self.mem[1]

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
        x_col = im2col(x_reshaped, self.ks, self.ks, self.s,self.p)
    
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

        dx = col2im(dx_col, (n * c, 1, h, w), self.ks, self.ks, self.s, self.p)
        dx = dx.reshape(ishape)

        self.mem = []

        return dx

    def update(self):
        pass

class ConvLayer:
    def __init__(self, size=3, amount=2,  pad=1, stride=1):
        self.ks = size ; self.p = pad ; self.s = stride ; self.a = amount
        self.kernels = np.random.rand(self.a, self.ks * self.ks) * np.sqrt(2./self.ks)
        self.b = np.zeros((self.a, 1))

    def forward(self, x):
        n, cp, hp, wp = x.shape
        c = self.a
        h = int(((hp + 2 * self.p - self.ks) / self.s) + 1)
        w = int(((wp + 2 * self.p - self.ks) / self.s) + 1)
        
        xcol = im2col(x, self.ks, self.ks, self.s, self.p)
        o = self.kernels @ xcol + self.b

        o = o.reshape(c, h, w, n)
        o = o.transpose(3, 0, 1, 2)

        self.mem = x.shape, xcol

        return o

    def backprop(self, do):
        db = np.sum(do, axis=(0, 2, 3)) / (do.shape[0] * do.shape[2] * do.shape[3])
        db = db.reshape(self.a, -1)

        do = do.transpose(1, 2, 3, 0).reshape(self.a, -1)
        dw = do @ self.mem[1].T 
        dw = dw.reshape(self.kernels.shape) / self.mem[1].shape[1]

        dxcol = self.kernels.T @ do 
        dx = col2im(dxcol, self.mem[0], self.ks, self.ks, self.s, self.p)

        self.mem = dw, db

        return dx

    def update(self):
        self.kernels -= self.mem[0]
        self.b -=  self.mem[1]

class CNN:
    def __init__(self):
        self.nn = []
    
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

    def add_conv_layer(self, size=3, amount=2, pad=1, stride=1):
        self.nn.append(ConvLayer(size, amount, pad, stride))

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
                    l.update()
            
            for xt, yt in zip(xv, yv):
                o = self.forward(xt)
                jv.append(self.cost(o, yt))
        
        return j, jv

    def get_learning_rate(self, e0, et, t, n):
        if (k := n / t) < 1: return e0 * (1 - k) + et * t
        return et


def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.
        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.
        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d. 
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
  
    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----
    
    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d

def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.
        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols

def col2im(dX_col, X_shape, HF, WF, stride, pad):
    """
        Transform our matrix back to the input image.
        Parameters:
        - dX_col: matrix with error.
        - X_shape: input image shape.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -x_padded: input image with error.
    """
    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))
    
    # Index matrices, necessary to transform our input image into a matrix. 
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[:, :, pad:-pad, pad:-pad]

(tx, ty), (vx, vy) = mnist.load_data()
tx = tx.astype('float32') / 255.
print (tx[0].shape)
tx = tx.reshape(tx.shape[0], 1, tx.shape[1], tx.shape[2])

c = CNN()
c.add_conv_layer(3, 32, 1, 1)
c.add_relu_layer()
c.add_pool_layer()
c.add_fc_layer(6272, 100, 1)
c.add_relu_layer()
c.add_fc_layer(100, 10, 0)
c.add_softmax_layer()

j, jv = c.sgd(1, tx, ty, e0=1e-4)
fig, axs = plt.subplots(2)
axs[0].plot(range(len(j)), j)
axs[1].plot(range(len(jv)), jv)
plt.show()