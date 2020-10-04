import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sk

def relu(x):
    return np.greater(x, 0).astype(int)

def softmax(z):
    assert len(z.shape) == 2
    s = np.array([np.max(z,axis=1)]).T
    e_z = np.exp(z - s)

    return e_z / np.array([np.sum(e_z, axis=1)]).T

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


X, y = sk.make_classification(n_samples=1000, n_features=4, n_informative=4,  n_redundant=0, n_repeated=0
                              , n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=2)


layers = [4, 4, 3, 2] ; layers = l_tuple(layers, 0)
hidden = []
bias = [np.zeros(l[1]) for l in layers]
e0 = 0.01
et = 0.0000001
t = 1000
for l in layers:
    hidden.append(np.random.rand(l[0], l[1]) * np.sqrt(2./l[0]))

j = [] ; d = len(hidden)
wd = 0.05

for n in range(3000):
    r = np.random.choice(X.shape[0], 1)
    o = [X[r]]
    
    #forwards
    for i in range(d - 1):
        o.append(o[-1] @ hidden[i] + bias[i])
        o[-1] = relu(o[-1]) * o[-1]
    o.append(softmax(o[-1] @ hidden[d - 1]))    

    #error
    w = [np.power(h, 2) for h in hidden]
    w = np.sum([np.sum(i) for i in w])
    j.append(cross_entropy(o[-1], get_one_hot(y[r], 2)))

    # backwards
    e = e0 * (1 - n/t) + et
    de = o[-1] - get_one_hot(y[r], 2)
    for i in range(d-1, -1, -1):
        dw = o[i].T @ de
        db = de.flatten()
        de = de @ hidden[i].T
        hidden[i] -= (dw + wd * hidden[i] * 2) * e
        bias[i] -= db * e

plt.plot(range(len(j)), j) ; plt.show()
o = [X[:5]]
for i in range(d - 1):
    o.append(o[-1] @ hidden[i] + bias[i])
    o[-1] = relu(o[-1]) * o[-1]
o.append(softmax(o[-1] @ hidden[d - 1]))
print (o[-1] - get_one_hot(y[0], 2))