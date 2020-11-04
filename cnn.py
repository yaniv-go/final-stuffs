import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sk
import random

class CNN:
    def __init__(self):
        self.nn = {}
        self.layers = []
        depth = 0
    