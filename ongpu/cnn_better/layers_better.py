import cupy as cp
import cupyx
import copy
import sys

def get_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  del N
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

class GlobalAveragePool:
    def __init__(self):
        pass
    
    def test(self, x):
        self.mem = n, c, w, h = x.shape
        x = x.reshape(n, c, w * h)
        o = cp.mean(x, axis=2, dtype='float32').reshape((n, c, 1, 1))

        self.mem = [None]

        return o

    def forward(self, x):
        self.mem = n, c, w, h = x.shape
        x = x.reshape(n, c, w * h)
        o = cp.mean(x, axis=2, dtype='float32').reshape((n, c, 1, 1))

        return o
    
    def bprop(self, do):
        n, c, w, h = self.mem

        dx = cp.zeros((n, c, w, h))
        dx[::] = do

        self.mem = [None]

        return dx

    def optimize(self, *args, **kwargs):
        pass

class Relu:
    def __init__(self):
        self.mem = [None]
    
    def test(self, x):
        a = cp.greater(x, 0).astype(cp.int8)
        self.mem = [None]
        return x * a

    def forward(self, x):
        self.mem = cp.greater(x, 0).astype(cp.int8)
        return x * self.mem

    def bprop(self, do):
        y = self.mem
        self.mem = [None]
        return y * do

    def optimize(self, *arg, **kwarg):
        pass

class Softmax:
    # maybe change to have cost types as optimizers 
    def __init__(self):
        self.mem = [None]

    def test(self, x):
        e_x = cp.exp(x - cp.max(x))
        softmaxed = e_x / e_x.sum(axis=1).reshape((-1, 1))
        self.mem = [None]
        return softmaxed

    def forward(self, x):
        e_x = cp.exp(x - cp.max(x))
        self.mem = softmaxed = e_x / e_x.sum(axis=1).reshape((-1, 1))
        return softmaxed

    def bprop(self, y):
        de = y - self.mem
        self.mem = [None]
        return de

    def optimize(self, *args, **kwargs):
        pass

class Fc:
    def __init__(self, row, col, optimizer='nadam', bias=True):
        optimizers = {'nadam' : self.adam_momentum, 'sgd' : self.sgd_nesterov_momentum}

        self.w = (cp.random.uniform(-1, 1, (row, col)) * cp.sqrt(2./col)).astype('float32')
        self.bias = bias
        if bias: self.b = cp.zeros(col).astype('float32')
        self.mem = [None]

        self.optimize = optimizers[optimizer]

    def test(self, x):
        self.mem = [None]
        if len(x.shape) > 2: x = x.reshape((x.shape[0], -1))

        o = x @ self.w
        if self.bias: o += self.b

        return o       

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
        else:
            self.mem = dw / self.mem[0]

        return dx
    
    def adam_momentum(self, e, t, d1=0.9, d2=0.999, wd=0):
        if self.bias:
            try:
                self.first_arm[0]
            except AttributeError:
                self.first_arm = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
            finally:
                self.first_arm = [m * d1 + (1 - d1) * gradient for m, gradient in zip(self.first_arm, self.mem)]

            try:
                self.second_arm[0]
            except AttributeError:
                self.second_arm = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
            finally:
                self.second_arm = [m * d2 + (1 - d2) * (gradient ** 2) for m, gradient in zip(self.second_arm, self.mem)]

            first_stabilize = 1 - cp.power(d1, t)
            second_stabilize = 1 - cp.power(d2, t)

            first_arm_norm = [(m * d1) / first_stabilize + ((1 - d1) * gradient) / first_stabilize for m, gradient in (zip(self.first_arm, self.mem))]
            second_arm_norm = [(m * d2) / second_stabilize for m in self.second_arm]

            gradients = [(e * m) / (cp.sqrt(n + 1e-9)) for m, n in zip(first_arm_norm, second_arm_norm)]

            self.w -= (e * wd * self.w + gradients[0])
            self.b -= (e * wd * self.b + gradients[1])
        else:
            gradient = self.mem
            try:
                self.first_arm[0]
            except AttributeError:
                self.first_arm = cp.zeros_like(gradient, dtype='float32')
            finally:
                self.first_arm = self.first_arm * d1 + (1 - d1) * self.mem
            
            try:
                self.second_arm[0]
            except AttributeError:
                self.second_arm = cp.zeros_like(gradient, dtype='float32')
            finally:
                self.second_arm = self.second_arm * d2 + (1 - d2) * (gradient ** 2)
            
            first_stabilize = 1 - cp.power(d1, t)
            second_stabilize = 1 - cp.power(d2, t)

            first_arm_norm = (self.first_arm * d1) / first_stabilize + ((1 - d1) * gradient) / first_stabilize
            second_arm_norm = (self.second_arm * d2) / second_stabilize

            gradient = (e * first_arm_norm) / cp.sqrt(second_arm_norm + 1e-9)

            self.w -= (e * wd * self.w + gradient)

        self.mem = [None]

    def sgd_nesterov_momentum(self, e, d1=0.9, wd=0):
        if self.bias:
            try:
                self.momentum[0]
            except AttributeError:
                self.momentum = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
            finally:
                self.momentum = [d1 * m + e * gradient for m, gradient in zip(self.momentum, self.mem)]
            
            gradients = [d1 * m + e * gradient for m, gradient in zip(self.momentum, self.mem)]

            self.w -= (e * wd * self.w + gradients[0])
            self.b -= (e * wd * self.b + gradients[1])
        else:
            gradient = self.mem
            
            try:
                self.momentum[0]
            except AttributeError:
                self.momentum = cp.zeros_like(gradient, dtype='float32')
            finally:
                self.momentum = d1 * self.momentum + e * gradient
            
            gradient = d1 * self.momentum + e * gradient

            self.w -= (e * wd * self.w + gradient)

        self.mem = [None]

class MaxPool:
    def __init__(self, kernel_size=2, stride=2, padding=0):
        self.ks = kernel_size
        self.stride = stride
        self.pad = padding     

    def test(self, x):
        n, c, ph, pw = x.shape
        h = int(((ph + 2 * self.pad - self.ks) / self.stride) + 1)
        w = int(((pw + 2 * self.pad - self.ks) / self.stride) + 1)

        x_reshaped = x.reshape(n * c, 1, ph, pw)
        del x

        xcol = im2col(x_reshaped, self.ks, self.ks, self.pad, self.stride)
        max_idx = cp.argmax(xcol, axis=0)

        out = xcol[max_idx, cp.arange(max_idx.size, dtype='int8')]
        out = out.reshape((h, w, n, c))
        out = out.transpose(2, 3, 0, 1)

        self.mem = [None]

        return out

    def forward(self, x):
        n, c, ph, pw = x.shape
        h = int(((ph + 2 * self.pad - self.ks) / self.stride) + 1)
        w = int(((pw + 2 * self.pad - self.ks) / self.stride) + 1)

        x_reshaped = x.reshape(n * c, 1, ph, pw)
        del x

        xcol = im2col(x_reshaped, self.ks, self.ks, self.pad, self.stride)
        max_idx = cp.argmax(xcol, axis=0)

        out = xcol[max_idx, cp.arange(max_idx.size, dtype='int8')]
        out = out.reshape((h, w, n, c))
        out = out.transpose(2, 3, 0, 1)

        self.mem = max_idx, xcol.shape, (n, c, ph, pw)

        return out
    
    def bprop(self, do):
        max_idx, col_shape, (n, c, h, w) = self.mem
        
        dxcol = cp.zeros(col_shape, dtype='float32')

        do = do.transpose(2, 3, 0, 1).ravel()

        dxcol[max_idx, cp.arange(max_idx.size)] = do

        dx = col2im(dxcol, (n * c, 1, h, w), self.ks, self.ks, self.pad, self.stride)
        dx = dx.reshape(n, c, h, w)
        
        self.mem = [None]

        return dx
    
    def optimize(self, *args, **kwargs):
        pass

class ConvLayer:                     
    def __init__(self, optimizer='nadam', kernel_heigth=3, kernel_width=3, output_channels=16, pad=1, stride=1, input_channels=3, bias=True):
        optimizers = {'nadam' : self.adam_momentum, 'sgd' : self.sgd_nesterov_momentum}
        self.optimize = optimizers[optimizer]
        
        self.oc = output_channels ; self.ic = input_channels
        self.kh = kernel_heigth ; self.kw = kernel_width        
        self.pad = pad ; self.stride = stride
        self.bias = bias

        self.k = cp.random.uniform(-1, 1, (self.oc, self.ic, self.kw, self.kh), dtype='float32') * cp.sqrt(2./(self.kh * self.kw))
        if self.bias: self.b = cp.zeros((self.oc, 1), dtype='float32')

    def test(self, x):
        n, pc, pw, ph = x.shape

        del pc

        w = int(((pw + 2 * self.pad - self.kw) / self.stride) + 1)
        h = int(((ph + 2 * self.pad - self.kh) / self.stride) + 1)

        kcol = self.k.reshape((self.oc, -1))

        xcol = im2col(x, self.kh, self.kw, self.pad, self.stride)
        o = kcol @ xcol
        if self.bias: o += self.b

        o = o.reshape((self.oc, w, h, n)).transpose(3, 0, 1, 2)

        self.mem = [None]

        return o

    def forward(self, x):
        n, pc, pw, ph = x.shape

        del pc

        w = int(((pw + 2 * self.pad - self.kw) / self.stride) + 1)
        h = int(((ph + 2 * self.pad - self.kh) / self.stride) + 1)

        kcol = self.k.reshape((self.oc, -1))

        xcol = im2col(x, self.kh, self.kw, self.pad, self.stride)
        o = kcol @ xcol
        if self.bias: o += self.b

        o = o.reshape((self.oc, w, h, n)).transpose(3, 0, 1, 2)

        self.mem = x.shape, xcol

        return o

    def bprop(self, do):
        if self.bias: 
            db = cp.sum(do, axis=(0, 2, 3)) / (do.shape[0] * do.shape[2] * do.shape[3])
            db = db.reshape((-1, 1))

        do = do.transpose(1, 2, 3, 0).reshape(self.oc, -1)
        dw = (do @ self.mem[1].T) / self.mem[0][0]
        dw = dw.reshape(self.k.shape)

        kcol = self.k.reshape((self.oc, -1))
        dxcol = kcol.T @ do
        dx = col2im(dxcol, self.mem[0], self.kh, self.kw, self.pad, self.stride)

        if self.bias:
            self.mem = dw, db
        else:
            self.mem = dw

        return dx

    def adam_momentum(self, e, t, d1=0.9, d2=0.999, wd=0):
        if self.bias:
            try:
                self.first_arm[0]
            except AttributeError:
                self.first_arm = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
            finally:
                self.first_arm = [m * d1 + (1 - d1) * gradient for m, gradient in zip(self.first_arm, self.mem)]

            try:
                self.second_arm[0]
            except AttributeError:
                self.second_arm = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
            finally:
                self.second_arm = [m * d2 + (1 - d2) * (gradient ** 2) for m, gradient in zip(self.second_arm, self.mem)]

            first_stabilize = 1 - cp.power(d1, t)
            second_stabilize = 1 - cp.power(d2, t)

            first_arm_norm = [(m * d1) / first_stabilize + ((1 - d1) * gradient) / first_stabilize for m, gradient in (zip(self.first_arm, self.mem))]
            second_arm_norm = [(m * d2) / second_stabilize for m in self.second_arm]

            gradients = [(e * m) / (cp.sqrt(n + 1e-9)) for m, n in zip(first_arm_norm, second_arm_norm)]

            self.k -= (e * wd * self.k + gradients[0])
            self.b -= (e * wd * self.b + gradients[1])

        else:
            gradient = self.mem
            try:
                self.first_arm[0]
            except AttributeError:
                self.first_arm = cp.zeros_like(gradient, dtype='float32')
            finally:
                self.first_arm = self.first_arm * d1 + (1 - d1) * self.mem
            
            try:
                self.second_arm[0]
            except AttributeError:
                self.second_arm = cp.zeros_like(gradient, dtype='float32')
            finally:
                self.second_arm = self.second_arm * d2 + (1 - d2) * (gradient ** 2)
            
            first_stabilize = 1 - cp.power(d1, t)
            second_stabilize = 1 - cp.power(d2, t)

            first_arm_norm = (self.first_arm * d1) / first_stabilize + ((1 - d1) * gradient) / first_stabilize
            second_arm_norm = (self.second_arm * d2) / second_stabilize

            gradient = (e * first_arm_norm) / cp.sqrt(second_arm_norm + 1e-9)

            self.k -= (e * wd * self.k + gradient)

        self.mem = [None]

    def sgd_nesterov_momentum(self, e, d1=0.9, wd=0):
        if self.bias:
            try:
                self.momentum[0]
            except AttributeError:
                self.momentum = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
            finally:
                self.momentum = [d1 * m + e * gradient for m, gradient in zip(self.momentum, self.mem)]
            
            gradients = [d1 * m + e * gradient for m, gradient in zip(self.momentum, self.mem)]

            self.k -= (e * wd * self.k + gradients[0])
            self.b -= (e * wd * self.b + gradients[1])
        else:
            gradient = self.mem
            
            try:
                self.momentum[0]
            except AttributeError:
                self.momentum = cp.zeros_like(gradient, dtype='float32')
            finally:
                self.momentum = d1 * self.momentum + e * gradient
            
            gradient = d1 * self.momentum + e * gradient

            self.k -= (e * wd * self.k + gradient)
        
        self.mem = [None]

class BatchNorm:
    def __init__(self, expected, optimizer='nadam'):
        optimizers = {'nadam' : self.adam_momentum, 'sgd' : self.sgd_nesterov_momentum}
        self.optimize = optimizers[optimizer]

        if isinstance(expected, int):
            self.gamma = cp.ones(expected, dtype='float32')
            self.beta = cp.zeros(expected, dtype='float32')

            self.forward = self.forward_fc
            self.bprop = self.bprop_fc
            self.test = self.test_fc
        elif isinstance(expected, tuple):
            assert len(expected) == 3, 'incorrect tuple'
            c, w, h = expected

            self.gamma = cp.ones((c, 1, w * h), dtype='float32')
            self.beta = cp.zeros((c, 1, w * h), dtype='float32')

            self.forward = self.forward_conv
            self.bprop = self.bprop_conv
            self.test = self.test_conv
        else:
           self.forward = None
           sys.exit('InputError: incorrect input') 

    def test_conv(self, x):
        n, c, w, h = x.shape

        x_reshaped = x.reshape((n, c, w * h)).transpose(1, 0, 2)
        mean = cp.mean(x_reshaped, axis=1, dtype='float32', keepdims=True)
        var = cp.sum((x_reshaped - mean) ** 2, axis=1, dtype='float32', keepdims=True) / n

        xhat = (x_reshaped - mean) / cp.sqrt(var + 1e-9)

        o = self.gamma * xhat + self.beta
        o = o.transpose(1, 0, 2).reshape((n, c, w, h))

        self.mem = [None]

        return o

    def forward_conv(self, x):
        n, c, w, h = x.shape

        x_reshaped = x.reshape((n, c, w * h)).transpose(1, 0, 2)
        mean = cp.mean(x_reshaped, axis=1, dtype='float32', keepdims=True)
        var = cp.sum((x_reshaped - mean) ** 2, axis=1, dtype='float32', keepdims=True) / n

        xhat = (x_reshaped - mean) / cp.sqrt(var + 1e-9)

        o = self.gamma * xhat + self.beta
        o = o.transpose(1, 0, 2).reshape((n, c, w, h))

        self.mem = x_reshaped, mean, var

        return o
    
    def forward_fc(self, x):
        n = x.shape[0]

        mean = cp.mean(x, axis=0, dtype='float32', keepdims=True)
        var = cp.sum((x - mean) ** 2, axis=0, dtype='float32', keepdims=True) / n

        xhat = (x - mean) / cp.sqrt(var + 1e-9)

        o = self.gamma * xhat + self.beta

        self.mem = x, mean, var

        return o
    
    def test_fc(self, x):
        n = x.shape[0]

        mean = cp.mean(x, axis=0, dtype='float32', keepdims=True)
        var = cp.sum((x - mean) ** 2, axis=0, dtype='float32', keepdims=True) / n

        xhat = (x - mean) / cp.sqrt(var + 1e-9)

        o = self.gamma * xhat + self.beta

        self.mem = [None]

        return o

    def bprop_conv(self, do):
        x_reshaped, mean, var = self.mem
        
        n, c, w, h = do.shape
        do = do.reshape((n, c, w * h)).transpose(1, 0, 2)

        dxhat = do * self.gamma
        dvar = cp.sum((dxhat * (x_reshaped - mean) * (((var + 1e-9) ** (-3./2.)) / -2)), axis=1, keepdims=True)
        dmean = cp.sum((dxhat * (-1. / cp.sqrt(var + 1e-9))), axis=1, keepdims=True) + dvar * (cp.sum((-2 * (x_reshaped - mean)), axis=1, keepdims=True) / n)
        
        dx = dxhat * (1. / cp.sqrt(var + 1e-9)) + dvar * ((2 * (x_reshaped - mean)) / n) + dmean / n
        dgamma = cp.sum((do * ((x_reshaped - mean) / cp.sqrt(var + 1e-9))), axis=1, keepdims=True)
        dbeta = cp.sum(do, axis=1, keepdims=True)

        self.mem = dgamma, dbeta

        dx = dx.transpose(1, 0, 2).reshape((n, c, w, h))

        return dx

    def bprop_fc(self, do):
        x, mean, var = self.mem
        n = x.shape[0]

        dxhat = do * self.gamma
        dvar = cp.sum((dxhat * (x - mean) * (((var + 1e-9) ** (-3/2.)) / -2.)), axis=0, keepdims=True)
        dmean = cp.sum((dxhat * (-1. / cp.sqrt(var + 1e-9))), axis=0, keepdims=True) + dvar * (cp.sum((-2 * (x - mean)), axis=0, keepdims=True) / n)
        
        dx = dxhat * (1. / cp.sqrt(var + 1e-9)) + dvar * ((2 * (x - mean)) / n) + dmean / n
        dgamma = cp.sum((do * ((x - mean) / cp.sqrt(var + 1e-9))), axis=0)
        dbeta = cp.sum(do, axis=0)

        self.mem = dgamma, dbeta

        return dx

    def adam_momentum(self, e, t, d1=0.9, d2=0.999, wd=0):
        try:
            self.first_arm[0]
        except AttributeError:
            self.first_arm = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
        finally:
            self.first_arm = [m * d1 + (1 - d1) * gradient for m, gradient in zip(self.first_arm, self.mem)]

        try:
            self.second_arm[0]
        except AttributeError:
            self.second_arm = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
        finally:
            self.second_arm = [m * d2 + (1 - d2) * (gradient ** 2) for m, gradient in zip(self.second_arm, self.mem)]

        first_stabilize = 1 - cp.power(d1, t)
        second_stabilize = 1 - cp.power(d2, t)

        first_arm_norm = [(m * d1) / first_stabilize + ((1 - d1) * gradient) / first_stabilize for m, gradient in (zip(self.first_arm, self.mem))]
        second_arm_norm = [(m * d2) / second_stabilize for m in self.second_arm]

        gradients = [(e * m) / (cp.sqrt(n + 1e-9)) for m, n in zip(first_arm_norm, second_arm_norm)]

        self.gamma -= (e * wd * self.gamma + gradients[0])
        self.beta -= (e * wd * self.beta + gradients[1])

        self.mem = [None]

    def sgd_nesterov_momentum(self, e, d1=0.9, wd=0):
        try:
            self.momentum[0]
        except AttributeError:
            self.momentum = [cp.zeros_like(gradient, dtype='float32') for gradient in self.mem]
        finally:
            self.momentum = [d1 * m + e * gradient for m, gradient in zip(self.momentum, self.mem)]
        
        gradients = [d1 * m + e * gradient for m, gradient in zip(self.momentum, self.mem)]

        self.gamma -= (e * wd * self.gamma + gradients[0])
        self.beta -= (e * wd * self.beta + gradients[1])

        self.mem = [None]

class ResidualBlock:
    def __init__(self, *layers):
        self.first = layers[0]
        self.layers = layers[1:]
    
    def forward(self, x):
        self.mem = o = self.first.forward(x)

        for l in self.layers:
            o = l.forward(o)
        
        o += self.mem
        self.mem = [None]

        return o
    
    def bprop(self, do):
        self.mem = do

        for l in self.layers[::-1]:
            do = l.bprop(do)
        
        do += self.mem
        self.mem = [None]

        return self.first.bprop(do)

    def optimize(self, *args, **kwargs):
        for l in self.layers:
            l.optimizer(*args, **kwargs)
        
        self.first.optimizer(*args, **kwargs)
    
    def test(self, x):
        self.mem = o = self.first.forward(x)

        for l in self.layers:
            o = l.forward(o)
        
        o += self.mem
        self.mem = [None]

        return o