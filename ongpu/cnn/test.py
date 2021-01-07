import timeit
import cupy as cp
import numpy as np
from layers import *
from PIL import Image
import pandas as pd
import os

resized_dataset_path = 'C:\\Users\\yaniv\\Documents\\datasets\\dog-breed\\images-resized'

pictures = []
labels = []
for folder in os.listdir(resized_dataset_path):
    breed_path = resized_dataset_path + '\\' + folder
    for photo in os.listdir(breed_path):
        image = Image.open(breed_path + '\\' + photo)
        pic = cp.asarray(image).transpose(2, 0, 1)
        pictures.append(pic)
        labels.append(folder)
        print (labels[5:])
        print (pictures)
        break
    break