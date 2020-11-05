import numpy as np

"""
a = np.arange(9) + 1
a = np.array((a, np.roll(a, 1), np.roll(a, 2))).T
k = 3
p = 0 ; c = 3
x = np.zeros(a.shape).T

for j in range(a.shape[1]):
    for i in range(a.shape[0]):    
        p = i // k 

        m = i - 3 * p
        n = j * c + p
        x[m, n] = a[i, j]
        print (x)

        if i == 8: p = 0
print (a)
print (x)
"""
"""
a = np.arange(9) + 1
a = np.array((a, np.roll(a, 1), np.roll(a, 2)))
x = np.concatenate((np.arange(0, 27, 3), np.arange(1, 27, 3), np.arange(2, 27, 3)), axis=0)
a = np.ravel(a)[x].reshape(3, 9)
print (x)
print (np.arange(0, 27, 3).dtype)
"""
a = np.arange(18)
print (a)
b = np.zeros(18).astype(bool)
b = np.packbits(b)
print (b)

for i in range(a.shape[0]):
    x = i % 3 ; y = i // 3
    n = 6 * x + y 
    j = n // 8
    np.unpackbits

