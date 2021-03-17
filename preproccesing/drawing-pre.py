from PIL import Image, ImageFilter
import numpy as np
import cupy as cp
import pickle
import time
import sys
import os

def get_arrays(path):
    return 

path = '/home/yaniv/drawings/downloads/'

classes = os.listdir(path)
groups_dict = {}

for c, group in enumerate(classes):
    groups_dict[group] = c
    a = np.load(path + group).reshape(-1, 1, 28, 28)
    b = np.zeros((a.shape[0]), dtype=np.uint8) + c
    try:
        x = np.append(x, a, axis=0)
        y = np.append(y, b)
    except NameError:
        x = a
        y = b

for i in range(50):
    p = np.random.permutation(x.shape[0])

    x = x[p]
    y = y[p]

print(y[34333:34666])

np.save('/home/yaniv/drawings/x.npy', np.append(x, x[:1672])
np.save('/home/yaniv/drawings/y.npy', np.append(y,z))