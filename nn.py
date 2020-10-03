import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn.datasets as sk

def relu(x):
    return np.greater(x, 0).astype(int)

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def cross_entropy(x, y):
    a = 10 ** -8
    n = x.shape[0]
    return -np.sum(y * np.log(x + (1 * a))) / n


X, y = sk.make_classification(n_samples=1000, n_features=2, n_informative=2,  n_redundant=0, n_repeated=0
                              , n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=3)

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, edgecolor='k')
plt.show()

# creating network
h_layers = []
l = [2, 4, 4, 2] 

l = [(l[x], l[x+1]) for x in range(len(l) - 1)]

for c, x in enumerate(l[:-1]):    
    h_layers.append(np.random.rand(x[0], x[1]))

o_layer = np.random.rand(l[-1][0], l[-1][1])

o = [X[:5]]
for h in h_layers:
    o.append(o[-1] @ h)
    o[-1] = relu(o[-1]) * o[-1]
o.append(o[-1] @ o_layer)

for x in o[-1]:
    print (x, '\n')
o[-1] = softmax(o[-1])
for x in o[-1]:
    print (x, '\n')

