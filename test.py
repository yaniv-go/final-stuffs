import timeit
import cupy as cp
import numpy as np

def normal(a, b):
    a + b

def numpy(a, b):
    np.dot(a, b)

def cupy(a, b):
    cp.add(a, b)


a = np.arange(1e7).reshape((100, -1))
b = a + 37
b = b

ca = cp.array(a)
cb = cp.array(b)


normal(ca, cb)
cupy(ca, cb)

print(timeit.timeit('normal(ca, cb)', globals=globals(), number=10000))
print(timeit.timeit('cupy(ca, cb)', globals=globals(), number=10000)) 