import numpy as np
from keras.datasets import reuters
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

## parameter 
max_words = 1000
batch_size = 32
epochs = 5

## load data 
print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


print('Building model...')
input_word = Input(shape = (max_words,))

layer1 = Dense(512)(input_word)
layer1_act = Activation('relu')(layer1)

layer2 = Dense(256)(layer1)
layer2_act = Activation('relu')(layer2)

layer3 = Dense(num_classes)(layer2_act)
layer3_act = Activation('softmax')(layer3)

model = Model(inputs = input_word, outputs = layer3_act)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

fit = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])