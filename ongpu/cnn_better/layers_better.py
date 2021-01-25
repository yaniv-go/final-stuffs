import cupy as cp
import cupyx
import copy
import sys

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

class Relu():
    def __init__(self, optimizer):
        optimizers = {'nadam' : self.adam_momentum, 'sgd-m' : self.sgd_momentum}
        self.mem = [None]
        self.optimizer = optimizers[optimizer]
    
    def forward(self, x):
        self.mem = cp.greater(x, 0).astype(cp.int8)
        return x * self.mem

    def bprop(self, do):
        y = self.mem
        self.mem = [None]
        return y * do

    def update(self):
        pass

    def adam_momentum(self, e, t, d1=0.9, d2=0.999, wd=0):
        pass

    def sgd_momentum(self, e, m=0.9, wd=0):
        pass

class Softmax():
    def __init__(self, optimizer):
        optimizers = {'nadam' : self.adam_momentum, 'sgd-m' : self.sgd_momentum}
        self.mem = [None]
        self.optimizer = optimizers[optimizer]

    def forward(self, x):
        e_x = cp.exp(x - cp.max(x))
        self.mem = softmaxed = e_x / e_x.sum(axis=1).reshape((-1, 1))
        return softmaxed

    def bprop(self, y):
        de = y - self.mem
        self.mem = [None]
        return de
    
    def update(self):
        pass

    def adam_momentum(self, e, t, d1=0.9, d2=0.99, wd=0):
        pass

    def sgd_momentum(self, e, m=0.9, wd=0):
        pass

class Fc():
    def __init__(self, row, col, optimizer, bias=None):
        optimizers = {'nadam' : self.adam_momentum, 'sgd-m' : self.sgd_momentum}

        self.w = (cp.random.rand(row, col) * cp.sqrt(2./col)).astype('float32')
        self.bias = bias
        if bias: self.b = cp.zeros(col).astype('float32')
        self.mem = [None]

        self.optimizer = optimizers[optimizer]

    def forward(self, x):
        self.mem = x
        if len(x.shape) > 2: x = x.reshape((x.shape[0], -1))

        o = x @ self.w
        if self.bias: o += self.b

        return o
    
    def bprop(self, do):
        x = self.mem
        self.mem = x.shape
        if len(x.shape) > 2: x = x.reshape((x.shape[0], -1))
        

        dw = x.T @ do
        dx = do @ self.w.T
        dx = dx.reshape(self.mem)

        if self.bias:
            db = cp.sum(do, axis=0)
            self.mem = dw / self.mem[0], db / self.mem[0]
        else: self.mem = dw / self.mem[0]

        return dx
    
    def update(self):
        if self.bias:
            self.w -= self.mem[0]
            self.b -= self.mem[1]
        else:
            self.w -= self.mem
        
        self.mem = [None]
    
    def adam_momentum(self, e, t, d1=0.9, d2=0.999, wd=0):
        if self.bias:
            try:
                self.first_arm == 'hello'
            except AttributeError:
                self.first_arm = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
            finally:
                self.first_arm = [m * d1 + (1 - d1) * gradient for m, gradient in zip(self.first_arm, self.mem)]

            try:
                self.second_arm == 'hello'
            except AttributeError:
                self.second_arm = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
            finally:
                self.second_arm = [m * d2 + (1 - d2) * gradient ** 2 for m, gradient in zip(self.second_arm, self.mem)]

            first_stabilize = 1 - cp.power(d1, t)
            second_stabilize = 1 - cp.power(d2, t)

            first_arm_norm = [(m * d1) / first_stabilize + ((1 - d1) * gradient) / first_stabilize for m, gradient in (zip(self.first_arm, self.mem))]
            second_arm_norm = [(m * d2) / second_stabilize for m in self.second_arm]

            gradients = [e * m / (cp.sqrt(n + 1e-9)) for m, n in zip(first_arm_norm, second_arm_norm)]

            self.mem = gradients
            

    def sgd_momentum(self, e, m=0.9, wd=0):
        pass
