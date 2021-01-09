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


dataset_path = "C:\\Users\\yaniv\\Documents\\datasets\\dog-breed\\"

with open(dataset_path + 'breed-dict.pickle', 'rb') as f:
    breeds = pickle.load(f)

images = cp.load(dataset_path + 'base-images.npy')
labels = cp.load(dataset_path + 'labels.npy')

for i in range(50):    
    p = cp.random.permutation(labels.shape[0])
    images = images[p]
    labels = labels[p]

images_and_extra = cp.concatenate((images, images[:156]), axis=0)
labels_and_extra = cp.concatenate((labels, labels[:156]), axis=0)

cp.save(dataset_path + "images-and-extra.npy", images_and_extra)
cp.save(dataset_path + "labels-and-extra", labels_and_extra)


        

