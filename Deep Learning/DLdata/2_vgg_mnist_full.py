#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 21:04:16 2017

@author: drlego
"""

### Step 1: Import modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import keras
import keras.backend as K

from keras.datasets import mnist
from keras.models import Model, Input, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D


## Fix random seed for reproducibility
np.random.seed(20170704)


## Check proper working directory
#os.chdir('path/to/day_2/')
if os.getcwd().split('/')[-1] == 'day_2':
    pass
else:
    raise OSError('Check current working directory.\n'
                  'If not specified as instructed, '
                  'more errors will occur throught the code.\n'
                  '- Current working directory: %s' % os.getcwd())

if not K.image_data_format() == 'channels_last':
    raise AssertionError("'image_data_format' is not 'channels_last'."
                         "Check '~/.keras/keras.json' file.")

####################################################################################


### Step 2: Load & preprocess data

## 2-1. Load
if os.path.exists('/home/user/.keras/datasets/mnist.npz'): 
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
else:
    raise OSError("'mnist.npz' file does not exist.")

## 2-2. Preprocess
# Change data types to 'float32'
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalization
X_train /= 255
X_test /= 255

# Check shapes of train / test data
assert  X_train.shape == (60000, 28, 28) and X_test.shape == (10000, 28, 28)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# Expand dimensions of X
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
    
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# Convert class vectors to binary class matrices (one-hot vectors)
Y_train = keras.utils.to_categorical(y_train, 10)
Y_test = keras.utils.to_categorical(y_test, 10)


####################################################################################


### Step 3: Build model

## 3-1. Define hyperparameters

epochs = 5
batch_size = 128
num_classes = 10

## 3-2. Define VGG model customized on MNIST data

# TODO: DEFINE INPUT TENSOR (channel last)
inputs = Input(shape=(28, 28, 1))

# Pad zeros to change shape: (28, 28, 1) --> (32, 32, 1)
x = ZeroPadding2D(padding=(2, 2))(inputs)

# Conv --> Conv --> MaxPool: (32, 32, 1) --> ( 32, 32, 8) --> (32, 32, 8) --> (16, 16, 8)
x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv1')(x)
x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv2')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool1')(x)

# TODO: WRITE CODE 
# Conv --> Conv --> MaxPool: (16, 16, 8) --> (16, 16, 16) --> (16, 16, 16) --> (8, 8, 16)
x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv1')(x)
x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv2')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool2')(x)

# TODO: WRITE CODE
# Conv --> Conv --> MaxPool: (8, 8, 16) --> (8, 8, 32) --> (8, 8, 32) --> (4, 4, 32)
x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv1')(x)
x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv2')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool3')(x)

# Flatten (4, 4, 32) --> (512, )
x = Flatten()(x)

# TODO: WRITE CODE
# Fully-connected layer 1: (512, ) --> (256, ), activation: relu, dropout: 0.5
x = Dense(256, activation='relu', name='fc_1')(x)
x = Dropout(0.5)(x)

# TODO: WRITE CODE
# Fully-connected layer 2: (256, ) --> (128, ), activation: relu, dropout: 0.5
x = Dense(128, activation='relu', name='fc_2')(x)
x = Dropout(0.5)(x)

# TODO: WRITE CODE
# Softmax Layer : (256, ) --> (10, )
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

# Instantiate model
model = Model(inputs=inputs, outputs=predictions, name='vgg_mnist')


####################################################################################


### Step 4: Define callbacks

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard

# List of callbacks
callbacks = []

# Model checkpoints
ckpt_path = './practice/vgg_mnist_ckpts/vgg_mnist.{epoch:02d}-{val_acc:.2f}.hdf5'
if not os.path.exists(os.path.dirname(ckpt_path)):
    os.makedirs(os.path.dirname(ckpt_path))

checkpoint = ModelCheckpoint(filepath=ckpt_path,
                             monitor='val_acc',
                             save_best_only=False,
                             verbose=1)
callbacks.append(checkpoint)

# Stop training early
earlystopping = EarlyStopping(monitor='val_loss',
                              patience=5,
                              verbose=1)
callbacks.append(earlystopping)

# Reduce learning rate when learning does not improve
reducelr = ReduceLROnPlateau(monitor='val_loss',
                             factor=0.1, 
                             patience=10,
                             verbose=1)
callbacks.append(reducelr)

# Tensorboard for visualization; only available with tensorflow backend
# In the terminal; tensorboard --logdir='/full/path/to/vgg_mnist_logs/'
if K.backend() == 'tensorflow':
    print('Using tensorboard callback')
    tb_logdir = './practice/vgg_mnist_logs/'
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


### Step 6: Save & load model weights

# Save model weights
model.save_weights('weights/vgg_mnist_weights.h5')

# Load model weights
model.load_weights('weights/vgg_mnist_weights_master.h5')


####################################################################################


### Step 7: Test model performance
test_scores = model.evaluate(X_test, Y_test, verbose=1)
print("Test accuracy: %.2f%%" % (test_scores[1] * 100))
#train_scores = model.evaluate(X_train, Y_train, verbose=1)
#print("Train accuracy: %.2f%%" % (train_scores[1] * 100))


####################################################################################


### Step 8: Using best checkpoint model
best_model_path = './practice/vgg_mnist_ckpts/filename.hdf5' # TODO: change filename
best_model = load_model(best_model_path)
best_model.summary()
test_scores = best_model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy: %.2f%%' %(test_scores[1] * 100))
