import glob
from tqdm import tqdm
import numpy as np
import os

import pickle

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier



for i in range(5):
    #getting data
    with open('Signal_Processing/LSTM_intermediate_features_peaks'+str(i), 'rb') as fp:
        features=pickle.load(fp)

    with open('Signal_Processing/LSTM_intermediate_labels_peaks'+str(i), 'rb') as fp:
        labels=pickle.load(fp)
    '''
    with open('Signal_Processing/X_train'+str(i), 'rb') as fp:
        X_train=pickle.load(fp)

    with open('Signal_Processing/X_test'+str(i), 'rb') as fp:
        X_test=pickle.load(fp)

    with open('Signal_Processing/y_train'+str(i), 'rb') as fp:
        y_train=pickle.load(fp)

    with open('Signal_Processing/y_test'+str(i), 'rb') as fp:
        y_test=pickle.load(fp)
    '''

    '''
    features=features.transpose(1,2,3,0)
    features=np.reshape(features,(len(features),4,32,64))
    print(features.shape)
    '''

    y=[]
    for l in labels:
        if np.argmax(l)==1:    #alco
            y.append(1)
        else:
            y.append(2)     #control

    y=np.array(y)
    print(y.shape)

    X=features

    X=np.reshape(X,(len(X),-1))

    print(X.shape)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    for learning_rate in learning_rates:
        gb = GradientBoostingClassifier(n_estimators=400, learning_rate = learning_rate, max_features=100, max_depth = 10, random_state = 0)
        gb.fit(X_train, y_train)
        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
        print()
    break