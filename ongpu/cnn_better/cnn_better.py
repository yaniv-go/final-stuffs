import layers_better as layers
import numpy as np
import cupy as cp
import copy
import sys
import os

class CNN:
    def __init__(self, input_shape, loss='cross-entropy', optimizer='nadam', pre_proc_x='none', pre_proc_y='one-hot'):
        optimizers = {'nadam' : self.adam_momentum, 'sgd' : self.sgd_nesterov_momentum}
        losses = {'cross-entropy' : self.cross_entropy}
        assert optimizer in optimizers, 'incorrect optimizer'
        assert isinstance(input_shape, tuple), 'insert input shape tuple'
        assert len(input_shape) == 3, 'insert tuple of correct length'
        for num in input_shape:
            assert isinstance(num, int), 'insert valid integer dimension'
            assert num > 0, 'insert valid positive dimensions'

        self.fit = optimizers[optimizer]
        self.curr_output = input_shape
        self.pre_proc_x = pre_proc_x
        self.pre_proc_y = pre_proc_y
        self.optimizer = optimizer
        self.cost = losses[loss]
        self.nn = []
  
    def adam_momentum(self, epochs, tx, ty, vx, vy, e, d1=0.9, d2=0.999, wd=0):
        train_loss, validation_loss = [], []
        train_acc, validation_acc = [], []
        t = 0
    
        for ep in range(epochs):
            j, jv = [], []
            a, av = [], []
            for n in range(tx.shape[0]):
                t += 1

                p = np.random.permutation(tx.shape[0])
                xb, yb = cp.asarray(tx[p], dtype='float32'), cp.asarray(ty[p], dtype='float32')

                xb = self.pre_proc_x(xb)
                yb = self.pre_proc_y(yb)

                o = self.forward(xb)
                j.append(self.cost(o, yb))
                a.append(self.accuracy(o, yb))

                self.bprop(yb)
                self.optimize(e, t, d1, d2, wd)
            
            train_loss.append(sum(j) / tx.shape[0])
            train_acc.append(sum(a) / tx.shape[0])

            for xb, yb in zip(vx, vy):
                xb, yb = cp.asarray(tx[p], dtype='float32'), cp.asarray(ty[p], dtype='float32')

                xb = self.pre_proc_x(xb)
                yb = self.pre_proc_y(yb)

                o = self.test(xb)
                jv.append(self.cost(o, yb))
                av.append(self.accuracy(o, yb))
            
            validation_loss.append(sum(jv) / vx.shape[0])
            validation_acc.append(sum(av) / vx.shape[0])

            print('epoch ')

        return train_loss, validation_loss, train_acc, validation_acc

    def sgd_nesterov_momentum(self, tx, ty, vx, vy, e, d1=0.9, wd=0):
        train_loss, validation_loss = [], []
        t = 0
    
    def test(self, o):
        for l in self.nn:
            o = l.test(o)

        return o

    def optimize(self, *args, **kwargs):
        for l in self.nn:
            l.optimize(*args, **kwargs)

    def bprop(self, do):
        for l in self.nn:
            do = l.bprop(do)

        return do

    def forward(self, o):
        for l in self.nn:
            o = l.forward(o)
        
        return o

    def cross_entropy(self, o, y):
        n = o.shape[0]
        return -cp.sum(y * cp.log2(o + 1e-10)) / n
    
    def one_hot(self, x):
        try:
            self.nb_classes
        except AttributeError:
            self.nb_classes = int(cp.max(x) + 1)
        finally:
            y = cp.zeros((x.size, self.nb_classes))
            y[cp.arange(x.size), x] = 1
        
        return y

    def norm_cent(self, x):
        mean = cp.mean(x, axis=(2, 3), dtype='float64', keepdims=True)
        x = (x - mean).astype('float32')

        return x
    
    def fc(self, output, activation='relu', batch_norm=False, after=True):
        assert isinstance(output, int), 'insert valid output'
        
        if isinstance(self.curr_output, tuple):
            self.curr_output = sum(self.curr_output)
        activations = {'relu' : layers.Relu}
        try:
            activation = activations[activation]
        except KeyError:
            raise KeyError('insert valid activation')
        
        if batch_norm:
            self.nn.append(layers.Fc(self.curr_output, output, self.optimizer, bias=False))
            if after:
                self.nn.append(activation())
                self.nn.append(layers.BatchNorm(output, self.optimizer))
            else:
                self.nn.append(layers.BatchNorm(output, self.optimizer))
                self.nn.append(activation())
        else:
            self.nn.append(layers.Fc(self.curr_output, output, self.optimizer, bias=True))
            self.nn.append(activation())
            
        
        self.curr_output = output
    
    def convLayer(self, num_maps, kernel_width=3, kernel_heigth=3, pad=1, stride=1, activation='relu', batch_norm=False, after=True):
        assert not isinstance(self.curr_output, int), 'cannot insert conv after fc'
        
        c, pw, ph = self.curr_output
        
        assert (pw + 2 * pad - kernel_width) % stride == 0, 'invalid kernel width'
        assert (ph + 2 * pad - kernel_heigth) % stride == 0, 'invalid kernel height'

        w = int((pw + 2 * pad - kernel_width) / stride + 1)
        h = int((ph + 2 * pad - kernel_heigth) / stride + 1)

        activations = {'relu' : layers.Relu}
        try:
            activation = activations[activation]
        except KeyError:
            raise KeyError('insert valid activation')
            
        if batch_norm:
            self.nn.append(layers.ConvLayer(self.optimizer, kernel_heigth, kernel_width, num_maps, pad, stride, c, False))
            if after:            
                self.nn.append(activation())
                self.nn.append(layers.BatchNorm((num_maps, w, h), self.optimizer))
            else:
                self.nn.append(layers.BatchNorm((num_maps, w, h), self.optimizer))
                self.nn.append(activation())
        else:
            self.nn.append(layers.ConvLayer(self.optimizer, kernel_heigth, kernel_width, num_maps, pad, stride, c, True))
            self.nn.append(activation())
        
        self.curr_output = (num_maps, w, h)

    def globalAveragePool(self):
        assert isinstance(self.curr_output, tuple), 'incorrect input for average pool'
        self.nn.append(layers.GlobalAveragePool())
        self.curr_output = (self.curr_output[0], 1, 1)

    def Softmax(self):
        assert isinstance(self.curr_output, int), 'invalid input for softmax'
        self.nn.append(layers.Softmax())
    
    def maxPool(self, kernel_size=2, stride=2, pad=0):
        assert isinstance(self.curr_output, tuple), 'invalid input shape for maxpool'

        c, pw, ph = self.curr_output

        assert (pw + 2 * pad - kernel_size) % stride == 0, 'invalid kernel width'
        assert (ph + 2 * pad - kernel_size) % stride == 0, 'invalid kernel height'

        w = int((pw + 2 * pad - kernel_size) / stride + 1)
        h = int((ph + 2 * pad - kernel_size) / stride + 1)

        self.nn.append(layers.MaxPool(kernel_size, stride, pad))

        self.curr_output = c, w, h

    def ResidualBlock(self, output_channels, depth=3, activation='relu', after=True):
        assert isinstance(self.curr_output, tuple), 'incorrect input for residual block'
        assert isinstance(output_channels, int), 'insert valid output channel'
        assert isinstance(depth, int), 'insert valid depth'

        activations = {'relu' : layers.Relu}
        
        pc, pw, ph = self.curr_output

        try: 
            activation = activations[activation]
        except KeyError:
            raise KeyError('insert valid activation')
            
        res_block = [layers.ConvLayer(optimizer=self.optimizer, output_channels=output_channels, input_channels=pc, bias=False)]
        for i in range(depth - 1):
            if after:
                res_block.append(activation())
                res_block.append(layers.BatchNorm((output_channels, pw, ph), optimizer=self.optimizer))
            else:
                res_block.append(layers.BatchNorm((output_channels, pw, ph), optimizer=self.optimizer))
                res_block.append(activation())
            res_block.append(layers.ConvLayer(optimizer=self.optimizer, output_channels=output_channels, input_channels=output_channels, bias=False))
        
        self.curr_output = (output_channels, pw, ph)
        self.nn.append(layers.ResidualBlock(*res_block))
        if after:
            self.nn.append(activation())
            self.nn.append(layers.BatchNorm(self.curr_output, optimizer=self.optimizer))
        else:
            self.nn.append(layers.BatchNorm(self.curr_output, optimizer=self.optimizer))
            self.nn.append(activation())

    def accuracy(self, o, y):
        n = o.shape[0]
        o = cp.argmax(o, axis=1, dtype='float32')
        yes = 0
        for output, arr in zip(o, y):
            if y[output] == 1: yes += 1

        return yes / n

    @property
    def pre_proc_x(self):
        return self._pre_proc_x
    
    @pre_proc_x.setter
    def pre_proc_x(self, x):
        pre_procs = {'norm-cent' : self.norm_cent, 'none' : lambda x: x}
        self._pre_proc_x = pre_procs[x]
    
    @property
    def pre_proc_y(self):
        return self._pre_proc_y

    @pre_proc_y.setter
    def pre_proc_y(self, x):
        pre_procs = {'one-hot' : self.one_hot, 'none' : lambda x: x}
        self._pre_proc_y = pre_procs[x]
    