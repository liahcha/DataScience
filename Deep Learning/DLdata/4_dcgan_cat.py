import os,random
import numpy as np

from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.activations import *
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras import backend as K
K.image_data_format()

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.misc import imread

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


def plot_loss(losses):
#        display.clear_output(wait=True)
#        display.display(plt.gcf())
        plt.figure(figsize=(5,4))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()

def plot_gen(n_ex=25,dim=(5,5), figsize=(5,5)):
    noise = np.random.normal(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)
    generated_images = ((generated_images * 127.5) + 127.5).astype(np.uint8)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,:,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_real(n_ex=25,dim=(5,5), figsize=(5,5) ):

    idx = np.random.randint(0,X_train.shape[0],n_ex)
    generated_images = X_train[idx,:,:,:]
    generated_images = ((generated_images * 127.5) + 127.5).astype(np.uint8)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,:,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


## load data
filedir = path+'/datasets/cats64/'
filenames = os.listdir(filedir)

X_train = []
for filename in filenames:
    img = imread(os.path.join(filedir, filename))
    X_train.append(img)

X_train = np.array(X_train)
X_train = X_train.astype('float32')
X_train = (X_train - 127.5) / 127.5

print(np.min(X_train), np.max(X_train))
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')



shp = X_train.shape[1:]
opt = Adam(lr = 0.0002, beta_1 = .5, clipnorm=1.)
dopt = Adam(lr = 0.0001, beta_1 = .5, clipnorm=1.)


# Build Generative model ...
g_input = Input(shape=[100])
H = Dense(512*4*4,kernel_initializer='glorot_normal')(g_input)
H = LeakyReLU(0.2)(H)
H = BatchNormalization()(H)

H = Reshape([4,4,512])(H) # dimension setting

H = Conv2D(512,(3,3),padding = 'same', kernel_initializer='glorot_normal')(H)
H = LeakyReLU(0.2)(H)
H = BatchNormalization()(H)

H = UpSampling2D()(H)
H = Conv2D(256,(3,3),padding='same',kernel_initializer='glorot_normal')(H)
H = LeakyReLU(0.2)(H)
H = BatchNormalization()(H)

H = UpSampling2D()(H)
H = Conv2D(128,(4,4),padding='same',kernel_initializer='glorot_normal')(H)
H = LeakyReLU(0.2)(H)
H = BatchNormalization()(H)

H = UpSampling2D()(H)
H = Conv2D(64,(5,5),padding='same',kernel_initializer='glorot_normal')(H)
H = LeakyReLU(0.2)(H)
H = BatchNormalization()(H)

H = UpSampling2D()(H)
H = Conv2D(3,(5,5),padding='same',kernel_initializer='glorot_normal')(H)
g_V = Activation('tanh')(H)

generator = Model(inputs = g_input, outputs = g_V)
generator.compile(loss = 'binary_crossentropy', optimizer = opt)
generator.summary()

# Build Discriminative model ...
d_input = Input(shape=shp)

H = Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer='glorot_normal')(d_input)
H = LeakyReLU(0.2)(H)

H = Conv2D(128,(4,4),strides=(2,2),padding='same',kernel_initializer='glorot_normal')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(0.5)(H)

H = Conv2D(256,(4,4),strides=(2,2),padding='same',kernel_initializer='glorot_normal')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(0.5)(H)

H = Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer='glorot_normal')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(0.5)(H)

H = Flatten()(H) # for 1 classification
d_V = Dense(1, activation = 'sigmoid')(H)

discriminator = Model(d_input,d_V)
discriminator.compile(loss = 'binary_crossentropy', optimizer=dopt)

## combine model
discriminator.trainable = False
GAN_input= Input(shape = (100,))
gen_sample = generator(GAN_input)
GAN_output = discriminator(gen_sample)

GAN = Model(inputs = GAN_input, outputs = GAN_output)
GAN.compile(loss = 'binary_crossentropy', optimizer= opt)

##############
## train GAN model

epoch = 10000
batch_size = 32
freq = 1

batch_count = X_train.shape[0]//batch_size

losses = {"d":[], "g":[]}
for i in range(epoch):
    for j in tqdm(range(batch_count)):
        noise_input = np.random.normal(0, 1, size= (batch_size, 100))

        real_image = X_train[np.random.randint(0, X_train.shape[0],
                                                size = batch_size)]

        fake_image = generator.predict(noise_input, batch_size=batch_size)

        X = np.concatenate([fake_image,real_image])
        y_discriminator = [0]*batch_size + [1] * batch_size

        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y_discriminator)
        losses["d"].append(d_loss)


        noise_input = np.random.normal(0, 1, size= (batch_size, 100))
        y_generator = [1] * batch_size
        discriminator.trainable = False
        g_loss = GAN.train_on_batch(noise_input, y_generator)
        losses["g"].append(g_loss)

    if i%freq==freq-1:
       plot_loss(losses)
       plot_gen()

    if i%10==10-1:
       print("###############################################")
       print("iteration = ", i+1)




#plot_gen()
#plot_real()