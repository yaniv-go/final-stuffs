import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import sklearn.datasets as sk

def relu(x):
    return x if x > 0 else 0

def d_relu(x):
    return np.greater(x, 0).astype(int)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy(x, y):
    a = 10 ** -8
    return -np.mean(y * np.log(x + (1 * a)))

def create_weights(layers):
    if not isinstance(layers, list): return
    weights = []
    for x, y in zip(layers, layers[1:]):
        weights.append(np.random.rand(x, y))
    return weights

def 

x, y = sk.make_classification(n_samples=1000, n_features=3, n_informative=3,  n_redundant=0, n_repeated=0
                              , n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=2)

layers = [2, 3, 2]
weights = create_weights(layers)
x = np.array([[1,1], [2,2]])
o = [x]
for w in weights:
    o.append(o[-1] @ w)
print (o[-1])
