import glob
from tqdm import tqdm
import numpy as np
import os

import pickle

from keras.layers import Input, Reshape,LSTM, Dense, SeparableConv2D, SeparableConv1D, MaxPooling2D, MaxPooling1D, Flatten,Dropout, BatchNormalization
from keras.models import Model
from keras.layers import LeakyReLU 
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from scipy.fftpack import fft, dct, ifft2, idct

#getting data
with open('smoote_features', 'rb') as fp:
    features=pickle.load(fp)

with open('smoote_labels', 'rb') as fp:
    labels=pickle.load(fp)

'''
newfeatures=[]
for f in features:
    newfeatures.append(np.transpose(f))

newfeatures=np.array(newfeatures)
features=newfeatures
print(features.shape)
'''

#features=np.reshape(features,(len(features),16*16,64))

print(features.shape)

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf.get_n_splits(features)

print(kf)
i=0

print(features.shape)
print(labels.shape)
'''
features=np.reshape(features,(len(features),16,16,64))
'''

for train_index, test_index in kf.split(features):
    print("Pass number: "+str(i))
    X_train, X_test=features[train_index], features[test_index]
    y_train, y_test=labels[train_index], labels[test_index]

    input_img = Input(shape=(256,64))  # adapt this if using `channels_first` image data format

    x = SeparableConv1D(128,5, strides=1, padding='same', name='conv1')(input_img)
    x=LeakyReLU()(x)
    x=MaxPooling1D()(x)

    x = SeparableConv1D(256,5,strides=1, padding='same', name='conv2')(x)
    x=LeakyReLU()(x)
    x=MaxPooling1D()(x)
    
    r=Reshape((64,16,16))(x)
    x = SeparableConv2D(128,5,strides=1, padding='same', name='conv3')(r)
    x=LeakyReLU()(x)
    x=MaxPooling2D()(x)

    x = SeparableConv2D(256,5,strides=1, padding='same', name='conv4')(x)
    x=LeakyReLU()(x)
    x=MaxPooling2D()(x)

    x=Flatten()(x)
    x=Dense(256, activation='relu')(x)

    x=Dropout(0.4)(x)
    x=Dense(512, activation='relu')(x)

    x=Dropout(0.4)(x)
    x=Dense(1024, activation='relu')(x)

    x=Dropout(0.4)(x)
    x=Dense(2048, activation='relu', name='dense001')(x)

    output=Dense(2, activation='softmax')(x)

    model = Model(input_img, output)

    adam=Adam(lr=0.0001)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

    model.summary()

    from keras.callbacks import TensorBoard

    tbcb =TensorBoard(log_dir='tmp/cnn_sep', histogram_freq=0, write_graph=True, write_images=True)
    es=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',baseline=None, restore_best_weights=True)

    model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=64,
                    shuffle=False,
                    validation_data=(X_test, y_test),
                    callbacks=[tbcb,es])
    i+=1
    model.save('CNN_sep_models/cnn_sep1d_2d_smoote'+str(i)+'.h5')
    try:
        with open('Validation_indices/cnn_sep1d_2d_smoote'+str(i), 'wb') as fp:
                    pickle.dump(test_index, fp)
    except:
        print("val_indices not saved")
    del(model)
    K.clear_session()