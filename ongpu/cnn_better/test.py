import timeit
import time
import cupy as cp
import numpy as np
import layers_better as layers
#from layers import *
from PIL import Image
import pandas as pd
import pickle
import cProfile
import os
#import cnn

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


dataset_path = "C:\\Users\\yaniv\\Documents\\datasets\\dog-breed\\"


fc = layers.Fc(3 * 3 * 2, 3, 'sgd', 0)
x = cp.arange(5 * 3 * 2 * 3).reshape((5, 2, 3, 3))
o = fc.forward(x)

o[:, 0] = -1
relu = layers.Relu('sgd')
o = relu.forward(o)

do = o
do = relu.bprop(do)
print(do)
do = fc.bprop(do)

fc.optimizer(0.01)
