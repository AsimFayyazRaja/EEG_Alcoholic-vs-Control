import glob
from tqdm import tqdm
import numpy as np
import os

import pickle
import copy
from keras.layers import Input,LeakyReLU,Conv1D, Dense, SeparableConv1D, MaxPooling1D, Flatten,Dropout, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from scipy.fftpack import fft, dct, ifft2, idct

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

with open('../Conv_perchannel/perchannel_conv_features_full', 'rb') as fp:
    X=pickle.load(fp)

with open('../Conv_perchannel/perchannel_conv_labels_full', 'rb') as fp:
    labels=pickle.load(fp)

X=X.transpose(1,2,3,0)
X=np.reshape(X,(len(X),16,16,64))
print(X.shape)

features=X
newfeatures=[]



#X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.1,shuffle=True)

model=load_model('autoencoder.h5')
layer_name = 'encoding1'
intermediate_layer_model = Model(inputs=model.input,
                            outputs=model.get_layer(layer_name).output)
newfeatures.append(np.array(intermediate_layer_model.predict(X)))

newfeatures=np.array(newfeatures[0])
print(newfeatures.shape)

with open('ae_features_full', 'wb') as fp:
    pickle.dump(newfeatures, fp)

with open('ae_labels_full', 'wb') as fp:
    pickle.dump(labels, fp)

'''
newfeatures=[]
for f in features:
    newfeatures.append(np.transpose(f))
    
newfeatures=np.array(newfeatures)
features=newfeatures
print(features.shape)
'''