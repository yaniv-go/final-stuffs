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
best memory wise
data = np.arange(16 * 3)
b = np.ones(data.shape).astype(bool)
i = 0 ; x = data[0] ; k = 4 ; f = 3
while np.any(b):
    if b[i] == 0: i = np.argmax(b > 0) ; x = data[i] ; continue
    b[i] = 0
    j = i % k * k * f + i // k
    data[j], x = x, data[j]     
    i = j
print (data.reshape(4, 12))
""" 
"""
best speed
data = np.arange(27)
k = 3
x = np.arange(0, data.shape[0], k)
x = np.concatenate((x, x + 1, x + 2))
data = data[x]
print (data.reshape(3, 9))
"""
