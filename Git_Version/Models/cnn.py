import glob
from tqdm import tqdm
import numpy as np
import os

import pickle

from keras.layers import Input, Dense, SeparableConv2D, MaxPooling2D, Flatten,Dropout, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from scipy.fftpack import fft, dct, ifft2

#getting data
with open('Signal_Processing/LSTM_intermediate_full_peakfeatures', 'rb') as fp:
    features=pickle.load(fp)

with open('Signal_Processing/LSTM_intermediate_full_labels', 'rb') as fp:
    labels=pickle.load(fp)

features=np.reshape(features,(len(features),32,32,1))

print(features.shape)


from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(features)

print(kf)
i=0

for train_index, test_index in kf.split(features):
    print("Pass number: "+str(i))
    X_train, X_test=features[train_index], features[test_index]
    y_train, y_test=labels[train_index], labels[test_index]
    input_img = Input(shape=(32,32,1))  # adapt this if using `channels_first` image data format

    x = SeparableConv2D(64, (5,5), strides=1, activation='relu', padding='same', name='conv1')(input_img)
    x=MaxPooling2D()(x)

    x = SeparableConv2D(128, (5,5),strides=1, activation='relu', padding='same', name='conv2')(x)
    x=MaxPooling2D()(x)

    x = SeparableConv2D(256, (5,5),strides=1, activation='relu', padding='same')(x)
    x=MaxPooling2D()(x)

    x=Flatten()(x)
    x=Dense(64, activation='relu')(x)

    x=Dropout(0.4)(x)
    x=Dense(128, activation='relu')(x)

    x=Dropout(0.4)(x)
    x=Dense(256, activation='relu', name='dense1')(x)

    x=Dropout(0.4)(x)
    x=Dense(512, activation='relu', name='dense2')(x)

    x=Dropout(0.4)(x)
    x=Dense(1024, activation='relu', name='dense3')(x)

    output=Dense(2, activation='softmax')(x)
    model = Model(input_img, output)

    adam=Adam(lr=0.0005)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

    model.summary()

    es=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',baseline=None, restore_best_weights=True)

    model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    callbacks=[es])

    model.save('CNN_models/convsep2d_after_lstm_peakfeatures'+str(i)+'.h5')
    try:
        with open('Validation_indices/convsep2d_after_lstm_peakfeatures'+str(i), 'wb') as fp:
                    pickle.dump(test_index, fp)
    except:
        print("val_indices not saved")
    del(model)
    K.clear_session()

    i+=1

'''
from sklearn.model_selection import KFold

kf = KFold(n_splits=3, shuffle=True)
kf.get_n_splits(features)

print(kf)
i=0

for train_index, test_index in kf.split(features):
    print("Pass number: "+str(i))
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test=features[train_index], features[test_index]
    y_train, y_test=labels[train_index], labels[test_index]

    input_img = Input(shape=(16,16,64))  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (5,5), strides=1, activation='relu', padding='same', name='conv1')(input_img)
    x=MaxPooling2D()(x)

    x = Conv2D(128, (5,5),strides=1, activation='relu', padding='same', name='conv2')(x)
    x=MaxPooling2D()(x)

    x = Conv2D(256, (5,5),strides=1, activation='relu', padding='same')(x)
    x=MaxPooling2D()(x)

    x=Flatten()(x)
    x=Dense(64, activation='relu')(x)

    x=Dropout(0.6)(x)
    x=Dense(128, activation='relu')(x)

    x=Dropout(0.6)(x)
    x=Dense(256, activation='relu', name='dense1')(x)

    x=Dropout(0.6)(x)
    x=Dense(512, activation='relu', name='dense2')(x)

    x=Dropout(0.6)(x)
    x=Dense(1024, activation='relu', name='dense3')(x)

    x=Dense(2048, activation='relu', name='dense4')(x)
    
    output=Dense(2, activation='softmax')(x)

    model = Model(input_img, output)

    adam=Adam(lr=0.0003)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

    model.summary()

    from keras.callbacks import TensorBoard

    tbcb =TensorBoard(log_dir='tmp/cnn_sep', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(X_train, y_train,
                    epochs=15,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    callbacks=[tbcb])
    i+=1
    model.save('CNN_models/cnn'+str(i)+'.h5')
    del(model)
    K.clear_session()
'''