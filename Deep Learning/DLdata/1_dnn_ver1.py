# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:46:29 2017

@author: HQ
"""
## load modulas
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input,Dense,Activation
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import to_categorical

## parameter
batch_size = 128
num_classes = 10
epoch = 10

## load mnist data
(x_train,y_train),(x_test,y_test) = mnist.load_data()

## preprocessing (feature scaling)

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
## 나누기 연산이 들어가므로 uint8 -> float32로 변경

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

## convert class vectors 
# Lable의 categorical 값을 One-hot 형태로 변환 
# 예를 들어 [1, 3, 2, 0] 를 
# [[ 0., 1., 0., 0.], 
# [ 0., 0., 0., 1.], 
# [ 0., 0., 1., 0.], 
# [ 1., 0., 0., 0.]] 
# 로 변환하는 것을 One-hot 형태라고 함

y_train_cat = to_categorical(y_train,num_classes)
y_test_cat = to_categorical(y_test,num_classes)


#### deep neural network modeling 
info_input = Input(shape=(28*28,))

layer1 = Dense(512)(info_input)
layer1_act = Activation('relu')(layer1)

layer2 = Dense(512)(layer1_act)
layer2_act = Activation('relu')(layer2)

layer3 = Dense(10)(layer2_act)
layer3_act = Activation('softmax')(layer3)

model = Model(inputs=[info_input], 
              outputs = [layer3_act])

### model structure
model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(),
              metrics = ['accuracy'])

model.fit(x_train, y_train_cat, batch_size = batch_size, epochs = epoch,
          verbose = 1,
          validation_data = (x_test, y_test_cat)) ## validation loss

## model Save & load 
model.save('path/filename.h5') ## save model
model = load_model('path/filename.h5') ## load model
         
          
## case1          
score = model.evaluate(x_test,y_test_cat,verbose=0)          
print('Total loss on Testing set:', score[0])
print('Accuracyt of Testing set:', score[1])
        
## cacs2
pred = model.predict(x_test)
classify = []
for i in range(0,pred.shape[0]):
    classify.append(np.argmax(pred[i]))
classify =  np.asarray(classify)

result = [y_test,classify] ## fin table
result = np.matrix(result)


'''          
## checkpoint
from keras.callbacks import ModelCheckpoint
filepath = 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5 ### checkpoint neural network model improvements

filepath = 'weight-best.hdf5' ## check point best NN model Only 

checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]           

model.fit(x_train,y_train_cat, validation_split=0.2, epochs = epoch, batch_size = batch_size,
          callbacks = callbacks_list, verbose = 1 )

## load weights
model.load_weights("weight)

############ parameters .. 

mode type : max // auto // min 

if monitor = val_acc then mode = max
if monitor = val_loss then mode = min
if mode = auto then automatically inferred from the nmae of the monitored quantity

'''

'''
## early stopping
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 2, mode = 'auto')
callbacks_es = [earlystop]

model.fit(x_train,y_train_cat, validation_split=0.2, epochs = epoch, batch_size = batch_size,
          callbacks = callbacks_es, verbose = 1 )

#### patience 
when the loss on validation set doesn't improve for 2 epochs  ==> stop training

'''

''' 
### learning rate & weight decay & momentum

from keras.optimizers import *

optim1 = SGD(lr = 0.01, momentum = 0.0, decay = 0.0)
optim2 = Adagrad(lr = 0.01, decay = 0.0)
optim3 = Adam(lr = 0.0001, decay = 0.0)
....
## reference site 
## http://keras.io/optimizers/ 

'''
          
'''
## dropout example
#### deep neural network modeling  with dropout

from keras.layers import Dropout

info_input = Input(shape=(28*28,))

layer0 = Dropout(0.5)(info_input)
layer1 = Dense(512)(layer0)
layer1_act = Activation('relu')(layer1)

layer2 = Dropout(0.5)(layer1_act)

layer3 = Dense(512)(layer2)
layer3_act = Activation('relu')(layer3)

layer4 = Dense(10)(layer3_act)
layer4_act = Activation('softmax')(layer4)

model = Model(inputs=[info_input], 
              outputs = [layer4_act])

model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = optim1,
              metrics = ['accuracy'])
'''











