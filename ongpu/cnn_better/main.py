from keras.datasets import mnist
from matplotlib import pyplot
from cnn_better import CNN
import numpy as np
import cupy as cp


(xt, yt), (xv, yv) = mnist.load_data()

xt = xt.reshape((-1 , 16 , 1, 28, 28))
xv = xv.reshape((-1 ,16 , 1, 28, 28))

cnn = CNN(input_shape=(1, 28, 28), pre_proc_x='norm-cent', pre_proc_y='one-hot')

cnn.convLayer(8, batch_norm=True)
cnn.maxPool()
cnn.ResidualBlock(8)
cnn.fc(100, batch_norm=True)
cnn.fc(10)
cnn.Softmax()

cnn.nb_classes = 10

cnn.adam_momentum(20, xt, yt, xv, yv, 1e-03)

