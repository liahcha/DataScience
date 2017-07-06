import numpy as np
from keras.datasets import reuters
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.layers import Dropout

## parameter 
max_words = 1000
batch_size = 32
epochs = 20

## load data 
print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)

## word index check
word2idx = reuters.get_word_index()
idx2word = dict([(v, k) for k, v in word2idx.items()])


from collections import Counter
count = Counter()
count.update(y_train)
count.update(y_test)
count.most_common()  # topic, data index

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

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# -------------------------------------------
# train ( x_train / y_train )
# test ( x_test / y_test )

# hidden layers (dense 512, 256 , 128 , ... ) 
# activation function : relu for each layer 
# the activation function of last layer = softmax ? sigmoid ?

info_input = Input(shape=(1000,))

layer0 = Dropout(0.5)(info_input)
layer1 = Dense(512)(layer0)
layer1_act = Activation('relu')(layer1)

layer2 = Dropout(0.5)(layer1_act)

layer3 = Dense(512)(layer2)
layer3_act = Activation('relu')(layer3)

layer4 = Dropout(0.5)(layer3_act)

layer5 = Dense(46)(layer4)
layer5_act = Activation('softmax')(layer5)

model = Model(inputs=[info_input], outputs = [layer5_act])

### model structure
model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(),
              metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
          verbose = 1, 
          validation_split = 0.2) ## validation loss

# --------------------------------------------

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


