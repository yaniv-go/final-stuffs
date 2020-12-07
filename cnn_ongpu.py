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