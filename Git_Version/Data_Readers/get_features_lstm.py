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


'''
features=np.expand_dims(features,axis=2)

for i in range(5):

    model=load_model('Signal_Processing/LSTM_full_features'+str(i)+'.h5')
    layer_name = 'dense4'
    intermediate_layer_model = Model(inputs=model.input,
                                outputs=model.get_layer(layer_name).output)
    newfeatures=np.array(intermediate_layer_model.predict(features))

    with open('Signal_Processing/LSTM_intermediate_features_peaks'+str(i), 'wb') as fp:
        pickle.dump(newfeatures, fp)

    with open('Signal_Processing/LSTM_intermediate_labels_peaks'+str(i), 'wb') as fp:
        pickle.dump(labels, fp)
'''

i=0
for i in range(5):
    with open('X_train'+str(i), 'rb') as fp:
        X_train=pickle.load(fp)
    with open('X_test'+str(i), 'rb') as fp:
        X_test=pickle.load(fp)

    model=load_model('LSTM_full_features'+str(i)+'.h5')
    layer_name = 'dense4'
    intermediate_layer_model = Model(inputs=model.input,
                                outputs=model.get_layer(layer_name).output)
    newfeatures=np.array(intermediate_layer_model.predict(X_train))

    with open('X_train_LSTM_intermediate_features_peaks'+str(i), 'wb') as fp:
        pickle.dump(newfeatures, fp)

    intermediate_layer_model = Model(inputs=model.input,
                                outputs=model.get_layer(layer_name).output)

    newfeatures=np.array(intermediate_layer_model.predict(X_test))

    with open('X_test_LSTM_intermediate_features_peaks'+str(i), 'wb') as fp:
        pickle.dump(newfeatures, fp)

'''
with tqdm(total=5) as pbar:
    for i in range(5):
        model=load_model('Signal_Processing/LSTM_full_peaky_features.h5)
        layer_name = 'dense7'
        intermediate_layer_model = Model(inputs=model.input,
                                    outputs=model.get_layer(layer_name).output)
        newfeatures=np.array(intermediate_layer_model.predict(features))
        print(newfeatures.shape)
        print(labels.shape)

        with open('Signal_Processing/LSTM_intermediate_features_peaks'+str(i), 'wb') as fp:
            pickle.dump(newfeatures, fp)

        with open('Signal_Processing/LSTM_intermediate_labels_peaks'+str(i), 'wb') as fp:
            pickle.dump(labels, fp)
        try:
            del(model)
            K.clear_session()
        except:
            print("pass")
        pbar.update(1)
newfeatures=[]
for f in features:
    newfeatures.append(np.transpose(f))
    
newfeatures=np.array(newfeatures)
features=newfeatures
print(features.shape)
'''