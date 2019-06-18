import glob
from tqdm import tqdm
import numpy as np
import os

import pickle

from keras.layers import Input, LeakyReLU, Dense,Activation, LSTM, MaxPooling1D, Flatten,Dropout,Reshape, SeparableConv1D
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.models import load_model
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.optimizers import Adam
from keras.regularizers import l2

'''
#getting data
with open('Signal_Processing/overall_full_features', 'rb') as fp:
    features=pickle.load(fp)

with open('Signal_Processing/overall_full_labels', 'rb') as fp:
    labels=pickle.load(fp)

features=np.array(features)
labels=np.array(labels)
features=np.reshape(features,(len(features),-1))
features=np.expand_dims(features,axis=2)

from keras.callbacks import TensorBoard
from sklearn.model_selection import KFold

kf = KFold(n_splits=5,shuffle=True,random_state=42)
kf.get_n_splits(features)

print(kf)
i=0

for train_index, test_index in kf.split(features):

    X_train, X_test=features[train_index], features[test_index]
    y_train, y_test=labels[train_index], labels[test_index]

    with open('Signal_Processing/X_train'+str(i), 'wb') as fp:
        pickle.dump(X_train, fp)

    with open('Signal_Processing/X_test'+str(i), 'wb') as fp:
        pickle.dump(X_test, fp)
    
    with open('Signal_Processing/y_train'+str(i), 'wb') as fp:
        pickle.dump(y_train, fp)
    
    with open('Signal_Processing/y_test'+str(i), 'wb') as fp:
        pickle.dump(y_test, fp)

    i+=1
'''


i=0

for i in range(5):
    print("Pass number: "+str(i))
    
    with open('Signal_Processing/X_train'+str(i), 'rb') as fp:
        X_train=pickle.load(fp)

    with open('Signal_Processing/X_test'+str(i), 'rb') as fp:
        X_test=pickle.load(fp)

    with open('Signal_Processing/y_train'+str(i), 'rb') as fp:
        y_train=pickle.load(fp)

    with open('Signal_Processing/y_test'+str(i), 'rb') as fp:
        y_test=pickle.load(fp)
    m=0
    #X_train=np.array(X_train,dtype=np.float)
    #X_test=np.array(X_test,dtype=np.float)
    
    input_img = Input(shape=(4032,1))
    x=LSTM(16, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), return_sequences=True)(input_img)
    x=MaxPooling1D()(x)
    
    x=LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), return_sequences=True)(x)
    x=MaxPooling1D()(x)
    x=Flatten()(x)

    x=Dense(1024, name='dense1')(x)
    x=LeakyReLU()(x)
    x=Dropout(0.6)(x)

    output=Dense(2, activation='softmax')(x)

    model = Model(input_img, output)

    adam=Adam(lr=0.0001)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

    #model.summary()

    from keras.callbacks import TensorBoard

    es=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',baseline=None, restore_best_weights=True)

    history=model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    callbacks=[es])
    model.save('Signal_Processing/LSTM_full_features'+str(i)+'.h5')
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('lstm_acc'+str(i)+'.png')
    # "Loss"

    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('lstm_loss'+str(i)+'.png')
    plt.close()

    i+=1
    del(model)
    K.clear_session()
