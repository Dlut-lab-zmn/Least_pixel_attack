## contained in the LICENCE file in this directory.

from __future__ import print_function
import keras
from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib

from keras.utils import np_utils
from keras.models import load_model
from cleverhans.model import Model

class CIFAR:
    def __init__(self):
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')/255.-0.5
        x_test = x_test.astype('float32')/255.-0.5
    
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        #self.test_data = np.expand_dims( x_test[:, :, :, 0],3)
        self.test_data = x_test[:, :, :, :]
        self.test_labels = y_test
        
        VALIDATION_SIZE = 5000
        
        
        self.validation_data =x_train[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = y_train[:VALIDATION_SIZE]
        self.train_data =x_train[VALIDATION_SIZE:, :, :, :]
        
        #self.validation_data =np.expand_dims( x_train[:VALIDATION_SIZE, :, :, 0],3)
        #self.validation_labels = y_train[:VALIDATION_SIZE]
        #self.train_data =np.expand_dims( x_train[VALIDATION_SIZE:, :, :, 0],3)
        print(self.train_data.shape)
        self.train_labels = y_train[VALIDATION_SIZE:]


class CIFARModel(Model):
    def __init__(self):
        super(CIFARModel, self).__init__()
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]


        num_classes = 10
        
    
        x_shape = [32,32,3]
    
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same',
                             input_shape=x_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
    
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
    
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.5))
    
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        self.model = model
        self.model.load_weights("./Least pixel attack/models/cifar")

    def fprop(self, x, **kwargs):
        return {"logits":self.model(x)}
    def predict(self,x):
        return self.model(x)

