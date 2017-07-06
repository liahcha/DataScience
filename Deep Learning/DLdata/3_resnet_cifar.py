#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:55:06 2017

@author: drlego
"""

### Step 1: Import modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os

import keras
import keras.backend as K

from keras import layers
from keras.models import Input, Model, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10


## Fix random seed for reproducibility
np.random.seed(20170706)


## Check proper working directory
path = os.getcwd()
os.chdir(path)
if os.getcwd().split('/')[-1] == 'DLdata':
    pass
else:
    path = os.getcwd()+'/DLdata'
    #raise OSError('Check current working directory.\n'
    #              'If not specified as instructed, '
    #              'more errors will occur throught the code.\n'
    #              '- Current working directory: %s' % os.getcwd())
print(path)



####################################################################################


### Step 2: Load & preprocess data


## 2-1. Load
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

## 2-2. Preprocess
# Change data types to 'float32'
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalization
X_train /= 255
X_test /= 255

# Check shapes of train / test data
assert  X_train.shape == (50000, 32, 32, 3) and X_test.shape == (10000, 32, 32, 3)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# Convert class vectors to binary class matrices (one-hot vectors)
Y_train = keras.utils.to_categorical(y_train, 10)
Y_test = keras.utils.to_categorical(y_test, 10)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)


####################################################################################

### Step 3: Build model

## 3-1. Define hyperparameters

epochs = 30
batch_size = 128
num_classes = 10

## 3-2. Define ResNet model customized on MNIST data

# DEFINE INPUT TENSOR
inputs = Input(shape=(32, 32, 3)) # image sahpe = 32*32, channel 3

bn_axis= 3

# TODO: WRITE CODE FOR RESNET (OF YOUR PREFERENCE)
x = Conv2D(64, (3,3), padding='same', strides=(2,2), name='conv1')(inputs)
x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
x = Activation('relu')(x)

x = Conv2D(128, (3,3), padding='same', strides=(2,2), name='conv2')(x)
x = BatchNormalization(axis=bn_axis, name='bn_von2')(x)
x = Activation('relu')(x)

shortcut1 = Conv2D(128, (1,1), strides=(4, 4), name='conv5')(inputs)
shortcut1 = BatchNormalization(axis=bn_axis, name='reshape1')(shortcut1)
x = keras.layers.add([x, shortcut1])

x2 = Conv2D(256, (3,3), padding='same', strides=(2,2), name='conv3')(x)
x2 = BatchNormalization(axis=bn_axis, name='bn_von3')(x2)
x2 = Activation('relu')(x2)

#x2 = Conv2D(512, (3,3), padding='same', strides=(1,1), name='conv4')(x2)
#x2 = BatchNormalization(axis=bn_axis, name='bn_von4')(x2)

shortcut2 = Conv2D(256, (1,1), strides=(2,2), name='conv9')(x)
shortcut2 = BatchNormalization(axis=bn_axis, name='reshape2')(shortcut2)

x2 = keras.layers.add([x2, shortcut2])
x2 = Activation('relu')(x2)
x2 = Flatten(name='flatten')(x2)

# Softmax layer for predictions: (512, ) --> (10, ) 
predictions = Dense(10, activation='softmax', name='prediction')(x2)

# Instantiate model
model = Model(inputs=inputs, outputs=predictions, name='resnet_cifar10')


model.summary()

####################################################################################


### Step 4: Define callbacks

from keras.callbacks import TensorBoard

# List of callbacks
callbacks = []

# Tensorboard for visualization; only available with tensorflow backend
# In the terminal; tensorboard --logdir='/full/path/to/resnet_cifar10_logs/'
if K.backend() == 'tensorflow':
    print('Using tensorboard callback')
    tb_logdir = path+'/resnet_cifar10_logs/'
    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)
    tensorboard = TensorBoard(log_dir=tb_logdir,
                              histogram_freq=1,
                              write_graph=True)
    callbacks.append(tensorboard)
    
    
####################################################################################

### Step 5: Compile & train model


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print(model.summary())


history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_split=0.1)


####################################################################################


### Step 6: Save & load weights

# Save model weights
model.save_weights(path+'/weights/resnet_cifar10_weights.h5')


####################################################################################


### Step 7: Test model performance
test_scores = model.evaluate(X_test, Y_test, verbose=1)
print("Test accuracy: %.2f%%" % (test_scores[1] * 100))
#train_scores = model.evaluate(X_train, Y_train, verbose=1)
#print("Train accuracy: %.2f%%" % (train_scores[1] * 100))


####################################################################################