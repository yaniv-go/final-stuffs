import timeit
import cupy as cp
import numpy as np
from layers import *
from PIL import Image
import pandas as pd
import pickle
import cProfile
import os

def get_batches(x, y, k): 
    p = np.random.permutation(x.shape[0])
    x, y = x[p], y[p]

    print (x[0])
    print(y[0])

    n = x.shape[0] % k
    n = k - n
    x = np.append(x, x[:n], axis=0)
    y = np.append(y, y[:n], axis=0)

    return x, y

f = 'C:\\Users\\yaniv\\Documents\\datasets\\dog-breed\\images.npy'
images = np.load(f)
labels = np.load('C:\\Users\\yaniv\\Documents\\datasets\\dog-breed\\image-labels.npy')

input(4)

print(images.shape)
print(labels.shape)
images, labels = get_batches(images, labels, 16)
print(images.shape)
print(labels.shape)

x = images.reshape(-1, 16, 3, 224, 224)
y = images.reshape(-1, 16)

print (x[0])
print (y[0])
