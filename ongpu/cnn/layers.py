import cupy as cp
import cupyx
import copy
import sys

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
        n, cx, hp, wp = x.shape
        c = cx
        h = int(((hp + 2 * self.p - self.ks) / self.s) + 1)
        w = int(((wp + 2 * self.p - self.ks) / self.s) + 1)

        x_reshaped = x.reshape(n * c, 1, hp, wp)
        x_col = im2col(x_reshaped, self.ks, self.ks, self.p, self.s)

        max_idx = cp.argmax(x_col, axis=0)

        out = cp.amax(x_col, 0)
        out = out.reshape(h, w, n, c)
        out = out.transpose(2, 3, 0, 1)

        self.mem = max_idx, x_col.shape, x.shape

        return out
    
    def backprop(self, do):
        maxi, shape, xshape = self.mem
        
        dx_col = cp.zeros(shape)

        do_f = do.transpose(2, 3, 0, 1).ravel()

        for a, b, c in zip(dx_col.T, maxi, do_f):
            a[b] = c

        n, c, h, w = xshape

        dx = col2im(dx_col, (n * c, 1, h, w), self.ks, self.ks, self.p, self.s)
        dx = dx.reshape(xshape)

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
        n, cx, hp, wp = x.shape
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
        db = cp.sum(do, axis=(0, 2, 3))
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
        self.b -= (self.mem[1] + wd * self.b)



def get_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = int((H + 2 * padding - field_height) / stride + 1)
  out_width = int((W + 2 * padding - field_width) / stride + 1)

  i0 = cp.repeat(cp.arange(field_height), field_width)
  i0 = cp.tile(i0, C)
  i1 = stride * cp.repeat(cp.arange(out_height), out_width)
  j0 = cp.tile(cp.arange(field_width), field_height * C)
  j1 = stride * cp.tile(cp.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = cp.repeat(cp.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)

def im2col(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

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
    x_padded = cp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_indices(x_shape, field_height, field_width, padding,
                                stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    cupyx.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]