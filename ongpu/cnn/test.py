import timeit
import time
import cupy as cp
import numpy as np
from layers import *
from PIL import Image
import pandas as pd
import pickle
import cProfile
import os
import cnn
import re

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

def get_image_arrays(path, breeds):
    images_path = dataset_path + 'images-resized\\'
    
    images = []
    labels = []
    for folder in os.listdir(images_path):
        breed = folder[10:]
        breed_path = images_path + folder + '\\'
        for filename in os.listdir(breed_path):
            image_path = breed_path + filename
            image = Image.open(image_path).convert('RGB')
            images.append(cp.array(image))
            labels.append(breeds[breed])

    return images, labels

def get_ready_batches(dataset_path):
    images = cp.load(dataset_path + 'base-images.npy')
    labels = cp.load(dataset_path + 'labels.npy')

    images = images.transpose(0, 3, 1, 2)

    for i in range(50):    
        p = cp.random.permutation(labels.shape[0])
        images = images[p]
        labels = labels[p]

    images_and_extra = cp.concatenate((images, images[:156]), axis=0)
    labels_and_extra = cp.concatenate((labels, labels[:156]), axis=0)

    cp.save(dataset_path + "images-and-extra.npy", images_and_extra)
    cp.save(dataset_path + "labels-and-extra", labels_and_extra)

def resize_images(dataset_path, size):
    images_path = dataset_path + 'images-224\\'
    images_resized_path = dataset_path + 'images-112\\'

    os.mkdir(images_resized_path)

    for folder in os.listdir(images_path):
        breed_path = images_path + folder + '\\'
        resized_breed_path = images_resized_path + folder + '\\'
        os.mkdir(resized_breed_path)
        for photo in os.listdir(breed_path):
            image = Image.open(breed_path + photo)
            image = image.resize((size, size))
            image.save(resized_breed_path + photo)


dataset_path = "/home/yaniv/dog-breed/"
reg = re.compile(r'(\d{1,3}) : (.+)')

with open("/home/yaniv/dog-breed/" + 'breed-dict.pickle', 'rb') as f:
    breeds = pickle.load(f)

with open(dataset_path + '7-group-dict.pickle', 'rb') as f:
    groups = pickle.load(f)

with open(dataset_path + 'breeds-7.txt', 'r') as f:
    lines = f.readlines()

breeds_7 = []
for line in lines:
    a = re.match(reg, line)
    breeds_7.append(a.groups())
print(breeds_7)
breeds_7 = {x : y.strip() for x, y in breeds_7}

breeds_7 = {int(x) : groups[y] for x, y in breeds_7.items()}
print(breeds_7)
"""
with open(dataset_path + 'breeds-7.pickle', 'wb') as f:
    pickle.dump(breeds_7, f)

y = np.load(dataset_path + 'all-labels-shuffled.npy')
for c in range(y.shape[0]):
    y[c] = breeds_7[y[c]]

np.save(dataset_path + 'all-labels-grouped-7.npy', y)

print (np.max(y))
"""