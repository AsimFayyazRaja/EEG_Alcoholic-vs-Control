import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import csv
import string
from collections import Counter
from tqdm import tqdm
import collections, re
import random
from random import randint
import glob
from PIL import Image
from skimage import transform
import copy
from random import shuffle
import os
import time
import imageio
from skimage.io import imsave
from skimage import img_as_ubyte
from skimage import img_as_float
import cv2
import keras

import pickle

from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras.layers import Input, Activation, Conv2DTranspose, Flatten, Reshape
from keras import backend as K

from keras.models import load_model
from keras.optimizers import Adam
from sklearn.preprocessing import Normalizer

with open('../Conv_perchannel/perchannel_conv_features_full', 'rb') as fp:
    X=pickle.load(fp)


X=X.transpose(1,2,3,0)
X=np.reshape(X,(len(X),16,16,64))
print(X.shape)

input_img = Input(shape=(16, 16, 64))  # adapt this if using `channels_first` image data format

x = Conv2D(64, (3, 3), activation='relu', padding='valid')(input_img)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
x = MaxPooling2D((2, 2))(x)


x=Flatten()(x)
x=Dense(3,activation='relu', name="encoding")(x)        #Encodings here


x=Dense(256,activation='relu', name='encoding1')(x)
x=Reshape((16,16,1))(x)

x = Conv2DTranspose(128, (3, 3), padding='valid')(x)
x=Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
#16x16

x = Conv2DTranspose(32, (3, 3), padding='valid')(x)
x = Activation('relu')(x)
#32x32

x=Flatten()(x)
x=Dense(16*16,activation='linear', name='encoding2')(x)
x=Reshape((16,16,1))(x)
decoded = Conv2DTranspose(64, (3, 3), activation='linear', padding='same', name='out')(x)



adam=Adam(lr=0.0003)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=adam, loss='binary_crossentropy')

autoencoder.summary()

model=autoencoder
'''
transformer=Normalizer()

X=transformer.fit_transform(X)


X=np.reshape(X,(len(X),64,64,1))
'''

print(X.shape)


x_train=X[:10000]
x_test=X[10000:]

print(x_train.shape)
print(x_test.shape)

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, x_test))

autoencoder.save('autoencoder.h5')