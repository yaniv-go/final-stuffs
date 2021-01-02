import timeit
import cupy as cp
import numpy as np
from layers import *

a = cp.arange(4*4*3*2).reshape(3,2,4,4)
print (a)

print ('and now the transposed: \n')
a = a.transpose(1, 0, 2, 3)
print (a)

print('and now the vectorized: \n')
a = a.reshape(a.shape[0], a.shape[1], -1)
print(a)