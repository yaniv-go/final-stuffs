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

def poop():
    X, y = sk.make_classification(n_samples=1000, n_features=2, n_informative=2,  n_redundant=0, n_repeated=0
                              , n_classes=2, n_clusters_per_class=2, flip_y=0.01, class_sep=3)
    # creating network
    h_layers = []
    l = [2, 4, 4, 2] 

    l = [(l[x], l[x+1]) for x in range(len(l) - 1)]

    for c, x in enumerate(l[:-1]):    
        h_layers.append(np.random.rand(x[0], x[1]))

    o_layer = np.random.rand(l[-1][0], l[-1][1])

    # forwards
    o = [X[:5]]

    for h in h_layers:
        o.append(o[-1] @ h)
        o[-1] = relu(o[-1]) * o[-1]
    o.append(o[-1] @ o_layer)
    o[-1] = softmax(o[-1])

    # backwards
    e = 0.5
    y = get_one_hot(y, 2)
    error = cross_entropy(o[-1], y[:5])
    de = o[-1] - y[:5]
    dw = o[-2].T @ de
    print (dw)
    dh = de @ o_layer.T
    o_layer = np.subtract(o_layer, dw * e)

    de = dh * relu(dh)
    dw = o[1].T @ de
    dh = de @ h_layers[1].T
    h_layers[1] = np.subtract(h_layers[1], dw * e)

    de = dh * relu(dh)
    dw = o[0].T @ de 
    dh = de @ h_layers[0].T
    h_layers[0] = np.subtract(h_layers[0], dw * e)

    print ('i did it')
    #plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, edgecolor='k') ; plt.show()

def l_tuple(layers, i):
    try:
        layers[i] = (layers[i], layers[i + 1]) ; return(l_tuple(layers, i + 1))
    except IndexError:
        layers.pop(i) ; return layers
        
def initalize_weights(l, code=0):
    m, n = l
    if code == 0:
        p = np.sqrt(6 / (m + n))
        return np.random.uniform(-p, p, [m, n])

def sgd(n_iter, X, y, h, b, e0=0.5, k=32, t=100):
    depth = len(h) - 1
    data = pd.DataFrame()
    et = e0 / 100
    j = []

    for n in range(n_iter):
        print ('iteration: ', n, '\n\n')
        # forwards
        p = np.random.choice(X.shape[0], (k))
        X_t = np.array([X[i] for i in p])
        y_t = np.array([y[i] for i in p]) ; y_t = get_one_hot(y_t, h[depth].shape[1])

        o = [X_t]
        for l in range(depth):
            o.append(o[l] @ h[l] + b[l])
            o[l + 1] = o[l + 1] * relu(o[l + 1])
        o.append(softmax(o[depth] @ h[depth]))
        
        j.append(cross_entropy(o[depth + 1], y_t))
        
        
        #backwards
        e = n/t ; e = (1 - e) * e0 + e * et 
        gradient = o[depth + 1] - y_t
        print ('output of iteration: ', gradient, '\n\n')
        dw = []
        db = []
        # calculate gradients
        for l in range(depth, -1, -1):
            dw.append(o[l].T @ gradient / k)
            db.append(np.sum(gradient, axis=0) / k)
            gradient = gradient @ h[l].T ; gradient = relu(gradient) * gradient
        
        # use gradients
        dw = dw[::-1]
        db = db[::-1]
        #print ('iteration gradient of output weights: ', dw[0], '\n')
        #print ('iteration gradient of output bias: ', db[0], '\n\n\n')
        for l in range(depth + 1):
            h[l] = h[l] - dw[l] * e
            b[l] = b[l] - db[l] * e
        
    return j




        

X, y = sk.make_classification(n_samples=1000, n_features=6, n_informative=5,  n_redundant=1, n_repeated=0
                              , n_classes=3, n_clusters_per_class=2, flip_y=0.01, class_sep=2)

layers = [6, 8, 8, 4, 3] ; layers = l_tuple(layers, 0)
hidden = []
bias = [np.zeros(l[1]) + 0.1 for l in layers]
for l in layers:
    hidden.append(initalize_weights(l))

for h in hidden:
    print (h, '\n')
print ('\n\n\nand now for the second time: \n')
j = sgd(1000, X, y, hidden, bias, k=5) ; plt.scatter(range(len(j)), j) ; plt.show()
for h in hidden:
    print (h, '\n')
#plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, edgecolor='k') ; plt.show()
