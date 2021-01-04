import timeit
import cupy as cp
import numpy as np
from layers import *
from PIL import Image
import pandas as pd


dataset_path = 'C:\\Users\\yaniv\\Documents\\datasets\\dog-breed-identification'


csv = pd.read_csv(dataset_path + '\\labels.csv')

for filename in csv['id']:
    print (filename)
    image = Image.open(dataset_path + '\\train\\' + filename + '.jpg')
    new_image = image.resize((224, 224))
    new_image.save(dataset_path + '\\edited-train\\' + filename + '.jpg')