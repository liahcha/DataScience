#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:12:07 2017

@author: drlego
"""

### Step 1: Import modules & set logging

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np

import keras.backend as K

from keras.datasets import imdb
from keras.models import Model, Input
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


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


### Step 2: Load, view & preprocess data

## 2-1. Load
# Load dataset, but only keep the top n words
logging.info('Loading imdb dataset...')
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

logging.debug('Shape of train data: {}'.format(X_train.shape))
logging.debug('Shape of test data: {}'.format(X_test.shape))


## 2-2. View
# word2idx, idx2word <- python dictionaries
word2idx = imdb.get_word_index()
idx2word = dict((v, k) for k, v in word2idx.items())
logging.info('Vocabulary size: {}'.format(len(idx2word)))

# View the original review in text
def to_text(X, idx2word):
    text = [' '.join([idx2word[index] for index in review])for review in X]
    return text

text_train = to_text(X_train, idx2word)
text_test = to_text(X_test, idx2word)
logging.info('\n{}\n- {}'.format(text_train[0], ('pos' if y_train[0] == 1 else 'neg')))
logging.info('\n{}\n- {}'.format(text_test[245], ('pos' if y_test[245] == 1 else 'neg')))


## 2-3. Preprocess
# Truncate and pad input sequences
max_review_len = 500
if X_train.shape == (25000, ) and X_test.shape == (25000, ):
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_len,
                                     padding='pre', truncating='pre',
                                     value=0.)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_len,
                                    padding='pre', truncating='pre',
                                    value=0.)

logging.info('Pad sequences shorter than %d with "0"' % max_review_len)
logging.info('Truncate sequences longer than {0} to {0}'.format(max_review_len))
logging.debug('Shape of train data (preprocessed): {}'.format(X_train.shape))
logging.debug('Shape of test data (preprocessed) : {}'.format(X_test.shape))


####################################################################################


### Step 3: Build model

## 3-1. Hyperparameters
epochs = 5
batch_size = 128
hidden_size = 100
embedding_vector_len = 32

## 3-2. Define RNN model with LSTM cells for IMDB data

# Define input (SHAPE IS IMPORTANT!!!)
input_sequence = Input(shape=(max_review_len, ), # max_review_len = 500
                       dtype='int32', 
                       name='input_sequence')


# Define Embedding layer
x = Embedding(input_dim=top_words, # top_words = 5000 
              output_dim=embedding_vector_len, # embedding_vetor_len = 32
              input_length=max_review_len, # max_review_len = 500
              mask_zero=True,
              name='embedding')(input_sequence)


# Define LSTM layer
x = LSTM(units=hidden_size,
         dropout=0.,
         recurrent_dropout=0.,
         kernel_initializer='glorot_uniform',
         recurrent_initializer='orthogonal',
         return_sequences=False,
         name='lstm')(x)


# Define Dense layer
x = Dense(units=100, activation='relu', name='fc')(x)


# Define prediction layer; use sigmoid for binary classification
prediction = Dense(units=1, activation='sigmoid', name='prediction')(x)


# Instantiate model
model = Model(inputs=input_sequence,
              outputs=prediction,
              name='LSTM_imdb')


####################################################################################


### Step 4: Define callbacks

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard

# List of callbacks
callbacks = []

# Model checkpoints
ckpt_path = './demo/lstm_imdb_ckpts/weights.{epoch:02d}-{val_acc:.2f}.hdf5'
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

# Tensorboard for visualization
if K.backend() == 'tensorflow':
    tb_logdir = './demo/lstm_imdb_logs/'
    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)
    tensorboard = TensorBoard(log_dir=tb_logdir,
                              histogram_freq=1,
                              write_graph=True)
    callbacks.append(tensorboard)

####################################################################################


### Step 5: Compile & train model

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


print(model.summary())


history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=callbacks,
                    verbose=1)


####################################################################################


### Step 6: Save & load weights

# Save model weights
model.save_weights('weights/lstm_imdb_weights.h5')

# Load model weights
model.load_weights('weights/lstm_imdb_weights_master.h5')


####################################################################################


### Step 7: Test final model performance

test_scores = model.evaluate(X_test, y_test, verbose=1)
logging.info('Test accuracy: %.2f%%' %(test_scores[1] * 100))
#print("Test accuracy: %.2f%%" % (test_scores[1] * 100))

#train_scores = model.evaluate(X_train, y_train, verbose=1)
#print("Train accuracy: %.2f%%" % (train_scores[1] * 100))
