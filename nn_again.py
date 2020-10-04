import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def l_tuple(layers, i):
    try:
        layers[i] = (layers[i], layers[i + 1]) ; return(l_tuple(layers, i + 1))
    except IndexError:
        layers.pop(i) ; return layers


X, y = sk.make_classification(n_samples=1000, n_features=3, n_informative=3,  n_redundant=0, n_repeated=0
                              , n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=2)


layers = [3, 4 ,4, 2] ; layers = l_tuple(layers, 0)
hidden = []
bias = [np.zeros(l[1]) + 0.01 for l in layers]
for l in layers:
    hidden.append(np.random.rand(l[0], l[1]))

j = []

for n in range(0):
    r = np.random.choice(X.shape[0], 1)
    o = [X[r]]
    
    #forwards
    for h in hidden[:-1]:
        o.append(o[-1] @ h)    
        o[-1] = o[-1] * relu(o[-1])
    o.append(o[-1] @ hidden[-1])
    o[-1] = softmax(o[-1])

    #error
    j.append(cross_entropy(o[-1], get_one_hot(y[r], 2)))
    de = o[-1] - get_one_hot(y[r], 2)
    for h in hidden[::-1]:
        dw = o[hidden.index(h)].T @ de
        db = de
