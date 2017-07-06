#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:01:40 2017

@author: drlego
"""

### Step 1: Import modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
import keras
import keras.backend as K

from keras.datasets import mnist
from keras.models import Model, Input, load_model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM

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


## Set logging
def set_logging(testlog=False):
    # 1. Make 'logger' instance
    logger = logging.getLogger()
    # 2. Make 'formatter'
    formatter = logging.Formatter(
            '[%(levelname)s:%(lineno)s] %(asctime)s > %(message)s'
            )
    # 3. Make 'streamHandler'
    streamHandler = logging.StreamHandler()
    # 4. Set 'formatter' to 'streamHandler'
    streamHandler.setFormatter(formatter)
    # 5. Add streamHandler to 'logger' instance
    logger.addHandler(streamHandler)
    # 6. Set level of log; DEBUG -> INFO -> WARNING -> ERROR -> CRITICAL
    logger.setLevel(logging.DEBUG)
    # 7. Print test INFO message
    if testlog: # default is 'False'
        logging.info("Stream logging available!")
    
    return logger

_ = set_logging()


####################################################################################


### Step 2: Load & preprocess data

## 2-1. Load
# Data, shuffled and split between train / test sets
# shape of X_train, X_test; (batch_size, height, width)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
logging.info('MNIST data has been loaded.')

## 2-2. Preprocess
# Change data types to 'float32'
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# Normalization
X_train /= 255
X_test /= 255

# Check input shape of data; (batch_size, timesteps, input_dim)
timesteps = 28
input_dim = 28

X_train = X_train.reshape(-1, timesteps, input_dim)
X_test = X_test.reshape(-1, timesteps, input_dim)

logging.info('final training data shape: {}'.format(X_train.shape))
logging.info('final test data shape: {}'.format(X_test.shape))

# Convert class vectors to binary class matrices (one-hot vectors)
num_classes = 10
Y_train = keras.utils.to_categorical(y_train, num_classes)
Y_test = keras.utils.to_categorical(y_test, num_classes)

logging.info('final train label shape: {}'.format(y_train.shape))
logging.info('final test label shape: {}'.format(y_test.shape))


####################################################################################


### Step 3: Build Model

## 3-1. Define hyperparameters
epochs = 10
batch_size = 128
hidden_size = 100
num_classes = 10

## 3-2. Define RNN model with LSTM cells for MNIST data

# TODO: DEFINE INPUT TENSOR (hint; (timesteps, input_dim))
input_sequences = Input(shape=(timesteps, input_dim), # (28, 28)
                        name='input_sequences')

# TODO: DEFINE LSTM LAYER (with hidden_size=100)
x = LSTM(units=hidden_size, name='lstm')(input_sequences)

# TODO: DEFINE DENSE LAYER (with hidden_size=100)
x = Dense(units=100, activation='relu', name='fc')(x)

# TODO: DEFINE SOFTMAX LAYER (10 classes)
prediction = Dense(num_classes, activation='softmax', name='prediction')(x)

# INSTANTIATE MODEL
model = Model(inputs=input_sequences,
              outputs=prediction,
              name='LSTM_mnist')


####################################################################################


### Step 4: Define callbacks

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard

# List of callbacks
callbacks = []

# Model checkpoints
ckpt_path = './practice/lstm_mnist_ckpts/lstm_mnist.{epoch:02d}-{val_acc:.2f}.hdf5'
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
# In the terminal; tensorboard --logdir='/full/path/to/lstm_mnist_logs/'
if K.backend() == 'tensorflow':
    logging.info('Using tensorboard callback')
    tb_logdir = './practice/lstm_mnist_logs/'
    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)
    tensorboard = TensorBoard(log_dir=tb_logdir,
                              histogram_freq=1,
                              write_graph=True)
    callbacks.append(tensorboard)


####################################################################################


### Step 5: Compile & train model

# TODO: COMPILE MODEL
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


print(model.summary())


history = model.fit(X_train, Y_train, 
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=callbacks,
                    verbose=1)


####################################################################################


### Step 6: Save & load model weights

# Save model weights
model.save_weights('weights/lstm_mnist_weights.h5')

# Load model weights
model.load_weights('weights/lstm_mnist_weights_master.h5')


####################################################################################


### Step 7: Test model performance

test_scores = model.evaluate(X_test, Y_test, verbose=1)
logging.info('Test accuracy: %.2f%%' %(test_scores[1] * 100))
#print("Test accuracy: %.2f%%" % (test_scores[1] * 100))

#train_scores = model.evaluate(X_train, y_train, verbose=1)
#print("Train accuracy: %.2f%%" % (train_scores[1] * 100))


####################################################################################


### Step 8: Using best checkpoint model

best_model_path = './practice/lstm_mnist_ckpts/filename.hdf5' # TODO: change filename
best_model = load_model(best_model_path)
best_model.summary()
test_scores = best_model.evaluate(X_test, Y_test, verbose=1)
logging.info('Test accuracy: %.2f%%' %(test_scores[1] * 100))
