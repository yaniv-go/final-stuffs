import timeit
import cupy as cp
import numpy as np
from layers import *

a = cp.arange(2*2*2*2).reshape(2,2,2,2)
print(a)

c = BN_layer(a.shape)
print(c.forward(a))
print(c.back(a))
