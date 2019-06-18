import glob
from tqdm import tqdm
import numpy as np
import os

import pickle

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

#getting data
with open('butter_features', 'rb') as fp:
    features=pickle.load(fp)

with open('labels', 'rb') as fp:
    labels=pickle.load(fp)

features=np.reshape(features,(len(features),64,256))

print(features.shape)

newfeatures=[]
for f in features:
    newfeatures.append(np.transpose(f))

newfeatures=np.array(newfeatures)
features=newfeatures
print(features.shape)


from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(features)

print(kf)
i=0

'''
print(features.transpose(0,2,1).shape)
'''

for train_index, test_index in kf.split(features):
    print("Pass number: "+str(i))
    X_train, X_test=features[train_index], features[test_index]
    y_train, y_test=labels[train_index], labels[test_index]

    input_img = Input(shape=(256,64))  # adapt this if using `channels_first` image data format

    x = SeparableConv1D(128, 5, strides=1, padding='same', name='conv1')(input_img)
    x=LeakyReLU()(x)
    x=MaxPooling1D()(x)
    
    x = SeparableConv1D(256, 5,strides=1, padding='same', name='conv2')(x)
    x=LeakyReLU()(x)
    x=MaxPooling1D()(x)
    
    x = SeparableConv1D(512, 5,strides=1, padding='same')(x)
    x=LeakyReLU()(x)
    x=MaxPooling1D()(x)

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

    x=Dense(2048, activation='relu', name='dense41')(x)

    x=Dense(64, activation='relu', name='encoding')(x)

    output=Dense(2, activation='softmax')(x)

    model = Model(input_img, output)

    adam=Adam(lr=0.0005)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

    model.summary()

    from keras.callbacks import TensorBoard

    es=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',baseline=None, restore_best_weights=True)
    tbcb =TensorBoard(log_dir='tmp/cnn_sep', histogram_freq=0, write_graph=True, write_images=True)
    
    model.fit(X_train, y_train,
                    epochs=35,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    callbacks=[tbcb,es])
    i+=1
    model.save('CNN_sep_models/cnn_delta'+str(i)+'.h5')
    try:
        with open('Validation_indices/cnn_delta'+str(i), 'wb') as fp:
                    pickle.dump(test_index, fp)
    except:
        print("val_indices not saved")
    del(model)
    K.clear_session()