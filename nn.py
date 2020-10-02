import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sklearn.datasets as sk

def relu(x):
    return np.greater(x, 0).astype(int)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy(x, y):
    a = 10 ** -8
    return -np.mean(y * np.log(x + (1 * a)))


X, y = sk.make_classification(n_samples=1000, n_features=3, n_informative=3,  n_redundant=0, n_repeated=0
                              , n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=3)

layers = []
l = [3, 3, 4, 2] 
l = [(l[x], l[x+1]) for x in range(len(l) - 1)]

for c, x in enumerate(l):    
    layers.append(np.random.rand(x[0], x[1]) * 5)

o = [X[0]]
for l in layers:
    o.append(l @ o[-1])
    o[-1] = relu(o[-1]) * o[-1]
print (o)
