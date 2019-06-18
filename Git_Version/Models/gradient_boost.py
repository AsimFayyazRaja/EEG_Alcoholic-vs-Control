import glob
from tqdm import tqdm
import numpy as np
import os

import pickle

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier

for i in range(5):
    #getting data
    with open('Signal_Processing/LSTM_intermediate_features_peaks'+str(i), 'rb') as fp:
        features=pickle.load(fp)

    with open('Signal_Processing/LSTM_intermediate_labels_peaks'+str(i), 'rb') as fp:
        labels=pickle.load(fp)
    
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

    '''
    scaler = StandardScaler()
    scaler=scaler.fit(X)

    X=scaler.transform(X)
    '''
    clf = GradientBoostingClassifier(n_estimators=400)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    clf.fit(X_train,y_train)

    print(X_test.shape)

    print("Testing acc:",clf.score(X_test,y_test)*100)

    print("Training acc:",clf.score(X_train,y_train)*100)