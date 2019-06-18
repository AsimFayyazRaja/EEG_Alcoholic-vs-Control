import glob
from tqdm import tqdm
import numpy as np
import os

import pickle

from keras.layers import Input, LeakyReLU, Dense,Activation, LSTM, MaxPooling1D, Flatten,Dropout,Reshape, SeparableConv1D
from keras.models import Model
from keras import backend as K
from keras.layers.recurrent import RNN,GRU, SimpleRNN
from keras.models import load_model
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.optimizers import Adam


#getting data
with open('Signal_Processing/overall_full_features', 'rb') as fp:
    features=pickle.load(fp)

with open('Signal_Processing/overall_full_labels', 'rb') as fp:
    labels=pickle.load(fp)

y=labels
print(features.shape)
print(y.shape)

features=np.reshape(features,(len(features),-1))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler=scaler.fit(features)

features=scaler.transform(features)

features=np.expand_dims(features,axis=2)

from keras.callbacks import TensorBoard
from sklearn.model_selection import KFold

kf = KFold(n_splits=5,shuffle=True,random_state=42)
kf.get_n_splits(features)

print(kf)
i=0
'''
for train_index, test_index in kf.split(features):

    with open('Signal_Processing/SRNN_train_index'+str(i), 'wb') as fp:
        pickle.dump(test_index, fp)

    with open('Signal_Processing/SRNN_test_index'+str(i), 'wb') as fp:
        pickle.dump(train_index, fp)
    
    i+=1
'''
i=0

for i in range(5):

    print("Pass number: "+str(i))

    with open('Signal_Processing/SRNN_train_index'+str(i), 'rb') as fp:
        train_index=pickle.load(fp)

    with open('Signal_Processing/SRNN_test_index'+str(i), 'rb') as fp:
        test_index=pickle.load(fp)

    X_train, y_train=features[train_index], labels[train_index]
    X_test, y_test=features[test_index], labels[test_index]

    input_img = Input(shape=(3904,1))  # adapt this if using `channels_first` image data format

    x=SimpleRNN(32,return_sequences=True)(input_img)
    #x=MaxPooling1D()(x)

    x=SimpleRNN(64,return_sequences=True)(x)
    #x=MaxPooling1D()(x)

    x=Flatten()(x)

    x=Dense(256)(x)
    x=LeakyReLU()(x)

    x=Dense(512, name='dense3')(x)
    x=LeakyReLU()(x)
    x=Dropout(0.4)(x)

    x=Dense(1024, name='dense23')(x)
    x=LeakyReLU()(x)
    x=Dropout(0.4)(x)

    x=Dense(2048, name='dense4')(x)
    x=LeakyReLU()(x)
    x=Dropout(0.4)(x)

    x=Dense(1024, name='dense7')(x)
    x=LeakyReLU()(x)
    output=Dense(2, activation='softmax')(x)

    model = Model(input_img, output)

    adam=Adam(lr=0.00001)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

    model.summary()

    from keras.callbacks import TensorBoard

    es=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto',baseline=None, restore_best_weights=True)
    
    model.fit(X_train, y_train,
                    epochs=40,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    callbacks=[es])
    model.save('Signal_Processing/SRNN_full_features'+str(i)+'.h5')
    del(model)
    K.clear_session()
