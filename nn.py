import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sk

def create_layer(input, output):
    return np.random.rand(input, output)

def create_weights(layers):
    weights = []
    for x,y in zip(layers,layers[1:]):
        weights.append([create_layer(x, y), np.random.rand(y)])
    return weights

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

x, y = sk.make_classification(n_samples=1000, n_features=6, n_informative=6,  n_redundant=0, n_repeated=0
                              , n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=2)
print (x)
print (y)
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, edgecolor='k')
plt.show()