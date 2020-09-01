import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from cleverhans.model import Model
class CIFARModel(Model):
    def __init__(self,restore):
        super(CIFARModel, self).__init__()
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(64, (3, 3),
                         input_shape=(32, 32, self.num_channels)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.load_weights(restore)

        self.model = model
    def fprop(self, x, **kwargs):
        return {"logits":self.model(x)}
    def predict(self,x):
        return self.model(x)



