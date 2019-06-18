import glob
from tqdm import tqdm
import numpy as np
import os

import pickle

from keras.layers import Input, Dense, Reshape, ConvLSTM2D, MaxPooling2D, Flatten, Dropout
from keras.layers import Conv2D, LSTM, LeakyReLU, MaxPooling3D
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
import keras
from sklearn.model_selection import train_test_split



#getting data
with open('trial_features', 'rb') as fp:
    features=pickle.load(fp)

with open('trial_labels', 'rb') as fp:
    labels=pickle.load(fp)

print(features.shape)
'''
newfeatures=[]
for f in features:
    newfeatures.append(np.transpose(f))
    
newfeatures=np.array(newfeatures)
features=newfeatures
print(features.shape)
'''
features=np.reshape(features,(len(features),30,64,16,16))

print(features.shape)

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(features)

print(kf)
i=0
i=0
for train_index, test_index in kf.split(features):
    print("Pass number: "+str(i))
    X_train, X_test=features[train_index], features[test_index]
    y_train, y_test=labels[train_index], labels[test_index]

    input_img = Input(shape=(30,64,16,16))  # adapt this if using `channels_first` image data format

    conv=ConvLSTM2D(64,5,activation='relu', return_sequences=True)(input_img)
    maxi=MaxPooling3D()(conv)

    conv=ConvLSTM2D(128,5,activation='relu')(maxi)
    maxi=MaxPooling2D()(conv)

    x=Flatten()(maxi)
    x=Dense(64, activation='relu')(x)

    x=Dropout(0.2)(x)
    x=Dense(128, activation='relu')(x)

    x=Dropout(0.2)(x)
    x=Dense(256, activation='relu', name='dense1')(x)

    x=Dropout(0.2)(x)
    x=Dense(512, activation='relu', name='dense2')(x)

    x=Dropout(0.2)(x)
    x=Dense(1024, activation='relu', name='dense3')(x)


    output=Dense(2, activation='softmax')(x)
    adam=Adam(lr=0.0005)
    model = Model(input_img, output)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    es=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',baseline=None, restore_best_weights=True)
    model.summary()
    model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    callbacks=[es])   
    i+=1
    model.save('ConvLSTM_models/conv_lstm_2d'+str(i)+'.h5')
    try:
        with open('Validation_indices/conv_lstm_2d'+str(i), 'wb') as fp:
                    pickle.dump(test_index, fp)
    except:
        print("val_indices not saved")
    del(model)
    K.clear_session()