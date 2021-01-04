import timeit
import cupy as cp
import numpy as np
from layers import *
from PIL import Image
import pandas as pd
import os

dataset_path = 'C:\\Users\\yaniv\\Documents\\datasets\\dog-breed-identification'


csv = pd.read_csv(dataset_path + '\\labels.csv')

for filename in os.listdir(dataset_path + '\\test'):
    image = Image.open(dataset_path + '\\test\\' + filename)
    new_image = image.resize((224, 224))
    new_image.save(dataset_path + '\\edited-test\\' + filename)