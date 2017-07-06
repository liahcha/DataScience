### Step 1: Import modules
import os
from pprint import pprint

import keras
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K

from keras.datasets import cifar10
from keras.models import Model, Input, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import load_img, img_to_array


## Fix random seed for reproducibility
np.random.seed(20170704)

#path = os.getcwd()+'/DLdata'
#os.chdir(path)


#path = 'home/user/DataScience/DataScience/Study Note/Deep Learning/DLdata'
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


### Step 2: Define VGG16 model

## 2-1. Load
if os.path.exists('/home/user/.keras/datasets/cifar-10-batches-py.tar.gz'): 
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
else:
    raise OSError("file does not exist, must be downloaded.")
    
## 2-2. Preprocess
# Change data types to 'float32'
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Convert class vectors to binary class matrices (one-hot vectors)
Y_train = keras.utils.to_categorical(y_train, 10)
Y_test = keras.utils.to_categorical(y_test, 10)

# Normalization
X_train /= 255
X_test /= 255

# Check shapes of train / test data
assert  X_train.shape == (50000, 32, 32, 3) and X_test.shape == (10000, 32, 32, 3)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

####################################################################################


### Step 3: Build model
epochs = 5
batch_size = 128
num_classes = 10
    

# TODO: DEFINE INPUT TENSOR (channel last)
inputs = Input(shape=(32, 32, 3))

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
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)



# Flatten (4, 4, 32) --> (512, )
x = Flatten(name='flatten')(x)

# TODO: WRITE CODE
# Fully-connected layer 1: (512, ) --> (256, ), activation: relu, dropout: 0.5
x = Dense(256, activation='relu', name='fc1')(x)

# TODO: WRITE CODE
# Fully-connected layer 2: (256, ) --> (128, ), activation: relu, dropout: 0.5
x = Dense(128, activation='relu', name='fc2')(x)

# TODO: WRITE CODE
# Softmax Layer : (256, ) --> (10, )
predictions = Dense(10, activation='softmax', name='predictions')(x)

# Instantiate model
model = Model(inputs=inputs, outputs=predictions, name='vgg_cifar')

####################################################################################


### Step 4: Define callbacks

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard

# List of callbacks
callbacks = []

# Model checkpoints
ckpt_path = path+'/vgg_cifar10_ckpts/vgg_cifar10.{epoch:02d}-{val_acc:.2f}.hdf5'
if not os.path.exists(os.path.dirname(ckpt_path)):
    os.makedirs(os.path.dirname(ckpt_path))

checkpoint = ModelCheckpoint(filepath=ckpt_path,
                             monitor='val_acc',
                             save_best_only=True,
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
# In the terminal; tensorboard --logdir='/full/path/to/vgg_cifar10_logs/'
if K.backend() == 'tensorflow':
    print('Using tensorboard callback')
    tb_logdir = path+'/vgg_cifar10_logs/'
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
model.save_weights(path+'/weights/vgg_cifar10_weights.h5')

# Load model weights
model.load_weights(path+'/weights/vgg_cifar10_weights.h5')


####################################################################################


### Step 7: Test model performance
test_scores = model.evaluate(X_test, Y_test, verbose=1)
print("Test accuracy: %.2f%%" % (test_scores[1] * 100))
#train_scores = model.evaluate(X_train, Y_train, verbose=1)
#print("Train accuracy: %.2f%%" % (train_scores[1] * 100))


####################################################################################


### Step 8: Using best checkpoint model
# file list load
best_model_path = path+'/vgg_cifar10_ckpts/vgg_cifar10.04-0.69.hdf5' # must change filename
best_model = load_model(best_model_path)
best_model.summary()
test_scores = best_model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy: %.2f%%' %(test_scores[1] * 100))