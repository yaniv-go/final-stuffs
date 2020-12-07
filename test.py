import timeit
import cupy as cp
import numpy as np

def normal(a, b):
    a + b

def numpy(a, b):
    np.add(a, b)

def cupy(a, b):
    cp.add(a, b)


a = np.arange(1e7).reshape((100, -1))
b = a + 37

ca = cp.array(a)
cb = cp.array(b)

print(timeit.timeit('normal(a, b)', globals=globals()))