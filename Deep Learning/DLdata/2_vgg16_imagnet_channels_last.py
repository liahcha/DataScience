#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:56:04 2017

@author: drlego
"""

### Step 1: Import modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K

from keras.models import Model, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import load_img, img_to_array


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
    

####################################################################################


### Step 2: Define VGG16 model
def VGG16(weights_path=None):

    inputs = Input(shape=(224, 224, 3))
    
    # Block 1: (224, 224, 3) --> (224, 224, 64) --> (224, 224, 64) --> (112, 112, 64)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2: (112, 112, 64) --> (112, 112, 128) --> (112, 112, 128) --> (56, 56, 128)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3: (56, 56, 128) --> (56, 56, 256) --> (56, 56, 256) --> (56, 56, 256) --> (28, 28, 256)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4: (28, 28, 256) --> (28, 28, 512) --> (28, 28, 512) --> (28, 28, 512) --> (14, 14, 512)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5: (14, 14, 512) --> (14, 14, 512) --> (14, 14, 512) --> (14, 14, 512) --> (7, 7, 512) 
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(1000, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=predictions, name='vgg16')

    # Load weights
    if weights_path != None:
        if not K.image_data_format() == 'channels_last':
            raise AssertionError("Weights cannot be loaded:"
                                 "'image_data_format' must be set to 'channels_last'"
                                 "in '/.keras/keras.json' file.")
        else:
            model.load_weights(weights_path)
        
    return model


####################################################################################



### Step 3: Load model
weights_path = './models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model = VGG16(weights_path)
print(model.summary())


####################################################################################


### Step 4: Predict the label of a sample imagenet image

sample_images = ['dragonfly_1.jpg',
                 'marimba_1.jpg',
                 'terrier_1.jpg',
                 'toiletpaper_1.jpg']

img_path = os.path.join('imagenet_samples/', sample_images[1])

img = load_img(img_path, target_size=(224, 224))          # .jpg --> PIL instance

plt.imshow(img)                                           # show image

x = img_to_array(img)                                     # PIL --> Numpy array
print('image shape: {}'.format(x.shape))

x = np.expand_dims(x, axis=0)                             # add dimension 0 (channel first)
print('image shape: {}'.format(x.shape))

x = preprocess_input(x)                                   # preprocessing

y_pred = model.predict(x)                                 # make prediction

pprint(decode_predictions(y_pred, top=5))                 # view results
