### Step 1: Import modules
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import keras
import keras.backend as K

# model import : https://keras.io/applications
# VGG16, VGG19, ResNEt50, InseptionV3
from keras.applications.vgg16 import VGG16 


## Fix random seed for reproducibility
np.random.seed(20170706)

## Set logging
def set_logging(testlog=False):
    '''Custom function for allowing stream logging.'''
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


## Check proper working directory
#os.chdir('path/to/day_3_resnet+visualization/')
#if os.getcwd().split('/')[-1] == 'day_3_resnet+visualization':
#    pass
#else:
#    raise OSError('Check current working directory.\n'
#                  'If not specified as instructed, '
#                  'more errors will occur throught the code.\n'
#                  '- Current working directory: %s' % os.getcwd())
    
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


### Step 2: Import VGG model without 2 fc-layers

## 2-1. Load VGG16 model from 'keras.applications.vgg16'
if K.image_data_format() == 'channels_last': 
    # image shape = (height, width, channel); e.g (224, 224, 3)
    # setting : channel last
    model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3)) 
    # weights='imagenet' weight를 변경하고 싶을 경우
    weights_path = path+'/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5' # notop
    model.load_weights(weights_path) # load weights
    logging.info("Loaded VGG16 model weights, 'channels last'")

# View model architecture
print(model.summary())


####################################################################################


### Step 3: What have convolutional filters learned?

## 3-1. Define parameters

# Change input shape; (None, 224, 224, 3) --> (1, 224, 224, 3)
# batch dimension value 'None' to 1
input_shape = tuple(model.input._shape_as_list())
print('Shape of model input: ', input_shape)

input_shape = (1, ) + input_shape[1:]
print('Shape of model input: ', input_shape)

## 3-2. Visualize filters

# Utility function
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x = (x - x.mean()) / (x.std() + 1e-5) * 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Construct a python dict; with (key, value) = (layer_name, layer)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
print(layer_dict)

# Keep only the convolutional layers
conv_layer_dict = dict()
names_to_keep = ['Conv2D', 'Convolutional2D', 'Conv1D', 'Conv2D']
for layer_name, layer in layer_dict.items():
    if layer.__class__.__name__ in names_to_keep:
        conv_layer_dict[layer_name] = layer

#print(conv_layer_dict)

# Define name of conv layer
conv_layer_names = sorted(conv_layer_dict.keys())


##############################


# TODO: change names to show different features
conv_layer_name = conv_layer_names[0] # choose index from 0  12

kept_filters = []
for filter_idx in range(10): # TODO: change max number of filters
    # output logs 
    logging.info('Processing filter %d' % filter_idx)

    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    input_img = model.input

    layer_output = conv_layer_dict[conv_layer_name].output

    loss = K.mean(layer_output[:, :, :, filter_idx])

    # Define operation for computing gradients of
    # the input image with respect to the loss (gradient calculation)
    grads = K.gradients(loss=loss, variables=input_img)[0]

    # Normalization
    grads = grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # Define submodel that returns
    # the loss and grads given an input image
    submodel = K.function(inputs=[input_img], outputs=[loss, grads])

    # Step size (a.k.a learning rate) for gradient ascent
    step_size = 1.0

    # Randomly generate gray images with some noise
    input_img_data = np.random.random(input_shape)
    assert input_img_data.shape == (1, 224, 224, 3)

    input_img_data = (input_img_data - 0.5) * 20 + 128

    # Run gradient ascent for 20 steps
    # Note that we perform gradient ascent because
    # our objective is to maximize the loss function (sum of filter activations)
    for i in range(20):
        loss_value, grads_value = submodel([input_img_data])
        input_img_data += grads_value * step_size
        if loss_value <= 0.0:
            # skip some filters whose loss get stuck to 0.0
            logging.info('Skip filter %d' % filter_idx)
            break
        
    # decode the resulting input image
    if loss_value > 0:
        result = deprocess_image(input_img_data)
        kept_filters.append([result, loss_value])

    ###### kept_filters(list) ######
    #    [result, loss_value]      #
    #    [result, loss_value]      #
    #           ...                #
    #    [result, loss_value]      #
    ################################


# Number of filters to visualize
n_filters = 5 # TODO: change to
assert n_filters <= len(kept_filters)

# Visualize
try:
    fig, axes = plt.subplots(nrows=1, ncols=n_filters, 
                             figsize=(n_filters * 3, 3))
    for i in range(n_filters):
        img_array = np.squeeze(kept_filters[i][0], axis=0)
        if K.image_data_format() == 'channels_first':
            img_array = img_array.transpose(1, 2, 0)
        axes[i].imshow(img_array)
        #axes[i].set_title('filter %d' % (i + 1))
    fig.suptitle("What filters of '%s' see" % conv_layer_name)
    plt.show(fig)
    #plt.savefig('filters_{}.png'.format(layer_name))
except Exception as e:
    print(str(e))