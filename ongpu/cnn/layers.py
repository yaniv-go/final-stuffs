import cupy as cp
import cupyx
import copy
import sys

class BN_layer():
    def __init__(self, exp_shape):
        assert isinstance(exp_shape, tuple)
        if len(exp_shape) == 1: 
            self.forward = self.forward_prev_fc
            self.backprop = self.back_prev_fc

            self.gamma = cp.ones(exp_shape[0])
            self.beta = cp.zeros(exp_shape[0])
        elif len(exp_shape) == 3: 
            self.forward = self.forward_prev_conv
            self.backprop = self.back_prev_conv

            c, w, h = exp_shape
            self.gamma = cp.ones((c, 1, w * h))
            self.beta = cp.zeros((c, 1, w * h))
        else:
            raise Exception("incorrect exp_shape input")

    def forward_prev_conv(self, x):
        n, c, w, h = x.shape

        # normalize x
        x_reshaped = x.transpose(1, 0, 2, 3).reshape(c, n, w * h)
        mean = ((cp.sum(x_reshaped, axis=1)) / n).reshape(c, 1, w * h)
        var = ((cp.sum((x_reshaped - mean) ** 2, axis=1)) / n).reshape(c, 1, w * h)
        x_hat = (x_reshaped - mean) / (cp.sqrt(var + 1e-10))

        y = self.gamma * x_hat + self.beta      

        self.mem = x_reshaped, mean, var

        return y.transpose(1, 0, 2).reshape(n, c, w, h)

    def forward_prev_fc(self, x):
        n = x.shape[0]
        mean = cp.sum(x, axis=0) / n
        var = cp.sum((x - mean) ** 2, axis=0) / n
        x_hat = (x - mean) / cp.sqrt(var + 1e-10)

        y = self.gamma * x_hat + self.beta

        self.mem = x, mean, var

        return y

    def back_prev_conv(self, do):
        x_reshaped, mean, var = self.mem
        x_hat = (x_reshaped - mean) / (cp.sqrt(var + 1e-10))
        n, c, w, h = do.shape
        do = do.transpose(1, 0, 2, 3).reshape(c, n, w * h)

        var = cp.sqrt(var + 1e-9)

        """
        this is what i am doing in the following line i had to join it together for 
        memory effeciency concerns
        
        xminusm = (x_reshaped - mean)
        
        dx_hat = do * self.gamma
        dvar = dx_hat * xminusm * (var ** -3.) / -2.
        dmean = (dx_hat * (-1/ var)) + dvar * -2 * xminusm 
        
        dx_reshaped = (dx_hat / var) +  dvar * 2 * xminusm / n + dmean / n
        """

        dx_reshaped = (do * self.gamma / var + 
            -1 * do * self.gamma * ((x_reshaped - mean) ** 2) * (var ** -3) / n + 
            (-1 * do * self.gamma / var + do * self.gamma * ((x_reshaped - mean) ** 2) * (var ** -3)) / n)

        dgamma = cp.sum(do * x_hat, axis=1).reshape(c, 1, h * w)
        dbeta = cp.sum(do, axis=1).reshape(c, 1, h * w)

        self.mem = dgamma, dbeta

        return dx_reshaped.transpose(1, 0, 2).reshape(n, c, w, h)

    def back_prev_fc(self, do):
        x, mean, var = self.mem
        x_hat = (x - mean) / cp.sqrt(var + 1e-10)
        n = do.shape[0]

        var = cp.sqrt(var + 1e-9)
        
        """
        this is what i am doing in the following line i had to join it together for 
        memory effeciency concerns

        xminusm = x - mean

        dx_hat = do * self.gamma
        dvar = dx_hat * xminusm * (var ** -3.) / -2.
        dmean = (dx_hat * (-1/ var)) + dvar * -2 * xminusm 
        
        dx = (dx_hat / var) +  dvar * 2 * xminusm / n + dmean / n
        """

        dx = (do * self.gamma / var + 
            -1 * do * self.gamma * ((x - mean) ** 2) * (var ** -3) / n + 
            (-1 * do * self.gamma / var + do * self.gamma * ((x - mean) ** 2) * (var ** -3)) / n)

        dgamma = cp.sum(do * x_hat, axis=0)
        dbeta = cp.sum(do, axis=0)

        self.mem = dgamma, dbeta

        return dx

    def update(self, wd=0):
        self.gamma -= (wd * self.gamma + self.mem[0])
        self.beta -= (wd * self.beta + self.mem[1])

        self.mem = [None]
     
class SoftMaxLayer():
    def __init(self):
        self.mem = [None]

    def forward(self, x):
        self.mem = self.softmax(x)
        return self.mem

    def softmax(self, z):
        if  z.ndim < 2: z = cp.array([z])
        s = cp.max(z, axis=1).T
        e_z = cp.exp(z - s.reshape(-1, 1))

        return e_z / cp.sum(e_z, axis=1).T.reshape(-1, 1)

    def backprop(self, y):
        de = y - self.mem
        self.mem = [None]
        return de

    def update(self, wd=0):
        pass

class ReluLayer():
    def __init__(self):
        self.mem = [None]

    def forward(self, x):
        self.mem = x
        return self.relu(x)
    
    def relu(self, x):
        return cp.maximum(0, x)
    
    def d_relu(self, x):
        return cp.greater(x, 0).astype(cp.int8)
    
    def backprop(self, x):
        y = self.mem
        self.mem = [None]

        return self.d_relu(y) * x
    
    def update(self, wd=0):
        pass

class Fc():
    def __init__(self, row, column, prev_shape=0):
        self.prev_shape = prev_shape
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

        self.mem = [None]

class MaxPool():
    def __init__(self, size=2, stride=2, paddding=0):
        self.ks = size
        self.s = stride
        self.p = paddding

    def forward(self, x):
        n, c, hp, wp = x.shape
        h = int(((hp + 2 * self.p - self.ks) / self.s) + 1)
        w = int(((wp + 2 * self.p - self.ks) / self.s) + 1)

        x_reshaped = x.reshape(n * c, 1, hp, wp)
        if not cp.may_share_memory(x_reshaped, x):
            print('shit its forward maxpool')
        del x
        x_col = im2col(x_reshaped, self.ks, self.ks, self.p, self.s)

        max_idx = cp.argmax(x_col, axis=0)

        out = x_col[max_idx, cp.arange(max_idx.size)]
        out = out.reshape(h, w, n, c)
        out = out.transpose(2, 3, 0, 1)

        self.mem = max_idx, x_col.shape, (n, c, hp, wp)

        return out
    
    def backprop(self, do):
        maxi, shape, xshape = self.mem
        
        dx_col = cp.zeros(shape)

        do = do.transpose(2, 3, 0, 1).ravel()
        if not cp.may_share_memory(do, do):
            print('shit its maxpool backprop')

        dx_col[maxi, cp.arange(maxi.size)] = do

        n, c, h, w = xshape

        dx = col2im(dx_col, (n * c, 1, h, w), self.ks, self.ks, self.p, self.s)
        dx = dx.reshape(xshape)

        self.mem = [None]

        return dx

    def update(self, wd=0):
        pass

class ConvLayer:
    def __init__(self, size=3, amount=2, pad=1, stride=1, channels=1):
        self.ks = size ; self.p = pad ; self.s = stride 
        self.a = amount ; self.c = channels

        self.k = cp.random.rand(self.a, self.c, self.ks, self.ks) * cp.sqrt(2./(self.ks ** 2))
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

        k_col = self.k.reshape(self.a, -1)
        dxcol = k_col.T @ do
        dx = col2im(dxcol, self.mem[0], self.ks, self.ks, self.p, self.s)

        self.mem = dw, db

        return dx
    
    def update(self, wd):
        self.k -= (self.mem[0] + wd * self.k)
        self.b -= (self.mem[1] + wd * self.b)
        self.mem = [None]

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
