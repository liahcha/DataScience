### Step 1: Import modules

import os
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

import keras
import keras.backend as K

from keras.models import Model, Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Flatten, Dense, Activation
from keras.layers.normalization import BatchNormalization # for BachNormalization

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import load_img, img_to_array


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


### Step 2: Define Resnet model

## 2-1. Identity block
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    
    assert K.image_data_format() == 'channels_last'
    
    bn_axis = 3
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = keras.layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


## 2-2. Convolution block
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    
    assert K.image_data_format() == 'channels_last'
    
    bn_axis = 3 # index number : batch size, 가로, 세로, channel
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, # strides=2 (resizing as half-size)
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor) # for input output 레이어 shpae 맞춰주기 위함 
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


## 2-3. ResNet50
def ResNet50(weights_path=None):
    
    inputs = Input(shape=(224, 224, 3))
    
    assert K.image_data_format() == 'channels_last'
    
    bn_axis = 3    
    
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x) # Batch Normalization : before activation function
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # number of filter = 64, 64, 256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    
    x = Flatten()(x)
    predictions = Dense(1000, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, predictions, name='resnet50')

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
weights_path = path+'/weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
model = ResNet50(weights_path=weights_path)
print(model.summary())


####################################################################################


### Step 4: Predict the label of a sample imagenet image

sample_images = ['dragonfly_1.jpg',
                 'marimba_1.jpg',
                 'terrier_1.jpg',
                 'toiletpaper_1.jpg',
                 'dog_1.jpg',
                 'dog_2.jpg']

img_path = os.path.join(path+'/imagenet_samples/', sample_images[-2])

img = load_img(img_path, target_size=(224, 224))          # .jpg --> PIL instance

plt.imshow(img)                                           # show image (plot)

x = img_to_array(img)                                     # PIL --> Numpy array
print('image shape: {}'.format(x.shape))

x = np.expand_dims(x, axis=0)                             # add dimension 0 (channel first) - for including batch size
print('image shape: {}'.format(x.shape))
#print : ID, class, probability

# Check whether image shape matches input shape
assert model.input._shape_as_list()[1:] == list(x.shape)[1:]

x = preprocess_input(x)                                   # preprocessing

y_pred = model.predict(x)                                 # make prediction

pprint(decode_predictions(y_pred, top=10))                # view results