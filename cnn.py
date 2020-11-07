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

class Fc():
    def __init__(self, row, column, prev_shape=0):
        self.prev_shape = prev_shape
        self.mem = 0
        self.row = row
        self.col = column

        self.w = np.random.rand(self.row, self.col) * np.sqrt(2./self.col)
        self.b = np.zeros(self.col)
        if self.prev_shape: self.forward = self.ff
        else: self.forward = self.f

    def f(self, x):
        self.mem = x
        o = x @ self.w + self.b
        return o

    def ff(self, x):
        self.prev_shape = x.shape
        x = x.flatten()
        self.mem = x
        o = x @ self.w + self.b
        return o

class MaxPool():
    def __init__(self, size=2, stride=1, padding=0):
        self.mem = 0
        self.ks = 2
        self.s = stride
        self.p = padding
    
    def forward(self, x):
        self.mem = x

        n, cp, hp, wp = x.shape
        c = cp
        h = int(((hp + 2 * self.p - self.ks) / self.s) + 1)
        w = int(((wp + 2 * self.p - self.ks) / self.s) + 1)

        xcol = im2col(x, self.ks, self.ks, self.s, self.p) 
        xcol = xcol.reshape((c, xcol.shape[0] // c, -1))
        mp = np.max(xcol, axis=1)
        mp = np.array(np.hsplit(mp, n))
        mp = mp.reshape((n, c, h, w))

        return mp

class ConvLayer:
    def __init__(self, size=3, amount=2,  pad=1, stride=1):
        self.ks = size ; self.p = pad ; self.s = stride ; self.a = amount
        self.kernels = np.random.rand(self.a, self.ks * self.ks) * np.sqrt(2./self.ks)

    def forward(self, x):
        n, cp, hp, wp = x.shape
        c = self.a
        h = int(((hp + 2 * self.p - self.ks) / self.s) + 1)
        w = int(((wp + 2 * self.p - self.ks) / self.s) + 1)
        
        xcol = im2col(x, self.ks, self.ks, self.s, self.p)
        o = self.kernels @ xcol
        o = np.array(np.hsplit(o, n)).reshape((n, c, h, w))
        
        self.cache = x, xcol, self.kernels

        return o

class CNN:
    def __init__(self):
        self.nn = []
    
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
        return X_padded[pad:-pad, pad:-pad, :, :]

c = CNN()
c.add_conv_layer(3, 1, 1, 1)
c.add_relu_layer()
c.add_pool_layer()
c.add_fc_layer(9, 5, 1)
c.add_fc_layer(5, 4)
c.add_softmax_layer()
print (c.forward(np.arange(36).reshape((1, 1, 6, 6))))