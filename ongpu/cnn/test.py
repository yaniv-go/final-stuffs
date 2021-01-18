import timeit
import cupy as cp
import numpy as np
from layers import *
from PIL import Image
import pandas as pd
import pickle
import cProfile
import os
import cnn

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

with open(dataset_path + 'breed-dict.pickle', 'rb') as f:
    breeds = pickle.load(f)

x = cp.arange(2 * 3 * 5 * 5).reshape((2, 3, 5, 5))

res_block = ResidualBlock(BN_layer((3, 5, 5)), ConvLayer(amount=3, channels=3), BN_layer((3, 5, 5)), ConvLayer(amount=3, channels=3))

print ('input : ', x)
o = res_block.forward(x)
print('\n\nforward: ', o)
print('\n\nback: ', res_block.backprop(o))