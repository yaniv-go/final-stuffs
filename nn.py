import numpy as np

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



layers = [2, 3, 4, 2]

x = np.array([[1, 0, 0], 
             [1, 0, 0]])
y = np.array([[1, 0, 0],
             [1, 0, 0]])
print (x, y)
print (cross_entropy(x, y))