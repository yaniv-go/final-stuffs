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

a = np.arange(9) + 1
a = np.array((a, np.roll(a, 1), np.roll(a, 2)))
k = 3 ; c = 0
i = 0 ; j = 1
x = a[i, j]

while c != 26 - 1:
    p = j // k
    m = j - 3 * p
    n = i * 3 + p
    a[m, n], x = x, a[m, n]
    i = m ; j = n
    print (a)