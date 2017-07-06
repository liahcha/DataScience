import os
#import sputnik
import cv2, spacy, numpy as np
from spacy.en import English
from keras.models import model_from_json, Model
from keras.optimizers import SGD
from keras import backend as K
from keras.layers.merge import Concatenate
from sklearn.externals import joblib
from keras.layers import Input,Reshape, Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.activations import *
from keras.optimizers import *
'''
##
## open-cv2 install
#pip install opencv-python
c#onda install libgcc
#pip install spacy
'''
K.set_image_data_format("channels_first") ##

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


## weight path
VQA_model_file_name      = path+'/models/VQA/VQA_MODEL.json'
VQA_weights_file_name   = path+'/models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name  = path+'/models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = path+'/models/CNN/vgg16_weights_th_dim_ordering_th_kernels.h5'


'''
## download VGG16 weight file
https://github.com/fchollet/deep-learning-models/releases
filename : vgg16_weights_th_dim_ordering_th_kernels.h5

## download pre-train word2vec model
$sputnik --name spacy --repository-url http://index.spacy.io install en_glove_cc_300_1m_vectors

## copy
/home/user/anaconda3/lib/python3.6/site-packages/spacy/data
'''

### Compile the model
from models.CNN.VGG import VGG_16
image_model = VGG_16(CNN_weights_file_name)
    # this is standard VGG 16 without the last two layers
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
image_model.summary()


### Extract Image features from URL
def get_image_features(image_file_name, CNN_weights_file_name):
    image_features = np.zeros((1, 4096)) # 4096 dimensional 1 image
    im = cv2.resize(cv2.imread(image_file_name), (224, 224)) # (w*h)
    # img loading using opencv library 

    ## for URL
    #from skimage import io
    #im = cv2.resize(io.imread(image_file_name),(224,224))
    # dimension setting (224,224,3) to (1,224,224,3) idx1 : 1 image
    im = im.transpose((2,0,1)) # convert the image to RGBA
    im = np.expand_dims(im, axis=0)

    image_features[0,:] = image_model.predict(im)[0]
    return image_features


### word embedding
def get_question_features(question):
    #spacy.set_lang_class('en_glove_cc_300_1m_vectors', 'vectors')
    word_embeddings = spacy.load('en',vectors = 'en_glove_cc_300_1m_vectors')
    tokens = word_embeddings(question)

    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor

### try the embedding

## for windows
#spacy.set_lang_class('en_glove_cc_300_1m_vectors', 'vectors')
word_embeddings = spacy.load('en',vectors = 'en_glove_cc_300_1m_vectors')

obama = word_embeddings(u"obama")
putin = word_embeddings(u"putin")
banana = word_embeddings(u"banana")
monkey = word_embeddings(u"monkey")

obama.similarity(putin)
obama.similarity(banana)
banana.similarity(monkey)


### VQA model
# Image model
image_input = Input(shape=(4096,))
#model_image = Reshape([4096,])(image_input)

# Language Model
language_input = Input(shape = (30, 300,)) 
# (sequence, dimension: word 2 output vector- 300, )
# sequence (length: input word coount)
model_language = LSTM(512, return_sequences= True)(language_input)
model_language = LSTM(512, return_sequences= True)(model_language)
model_language = LSTM(512, return_sequences= False)(model_language)

# combine model
vqa_input = Concatenate()([model_language, image_input])

vqa_model = Dense(1024, kernel_initializer='glorot_normal')(vqa_input)
vqa_model = Activation('tanh')(vqa_model)
vqa_model = Dropout(0.5)(vqa_model)

vqa_model = Dense(1024, kernel_initializer='glorot_normal')(vqa_model)
vqa_model = Activation('tanh')(vqa_model)
vqa_model = Dropout(0.5)(vqa_model)

vqa_model = Dense(1024, kernel_initializer='glorot_normal')(vqa_model)
vqa_model = Activation('tanh')(vqa_model)
vqa_model = Dropout(0.5)(vqa_model)

vqa_model = Dense(1000)(vqa_model)
vqa_value = Activation('softmax')(vqa_model)

vqa_model = Model(inputs = [language_input, image_input], outputs = vqa_value)
vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

vqa_model.load_weights(VQA_weights_file_name)
vqa_model.summary()


############################################################################################

#image_file_name = path+"/test.jpg"
image_file_name = path+"/img_sample/food.jpeg"
question = u'what is vehicle in the picture?'

## get the image features
image_features = get_image_features(image_file_name, CNN_weights_file_name)

## get the question features
question_features = get_question_features(question)

labelencoder = joblib.load(label_encoder_file_name)

y_output = vqa_model.predict([question_features,image_features])
for label in reversed(np.argsort(y_output)[0,-10:]):
    print (str(round(y_output[0,label]*100,2)),"% ",
                     labelencoder.inverse_transform(label))


