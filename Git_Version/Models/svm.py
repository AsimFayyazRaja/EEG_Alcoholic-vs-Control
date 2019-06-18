import glob
from tqdm import tqdm
import numpy as np
import os

import pickle

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

'''
for i in range(5):
    #getting data
    with open('Signal_Processing/LSTM_intermediate_features_peaks'+str(i), 'rb') as fp:
        features=pickle.load(fp)

    with open('Signal_Processing/LSTM_intermediate_features_peaks'+str(i), 'rb') as fp:
        labels=pickle.load(fp)
    
    with open('Signal_Processing/X_train'+str(i), 'rb') as fp:
        X_train=pickle.load(fp)

    with open('Signal_Processing/X_test'+str(i), 'rb') as fp:
        X_test=pickle.load(fp)

    with open('Signal_Processing/y_train'+str(i), 'rb') as fp:
        y_train=pickle.load(fp)

    with open('Signal_Processing/y_test'+str(i), 'rb') as fp:
        y_test=pickle.load(fp)
    features=features.transpose(1,2,3,0)
    features=np.reshape(features,(len(features),4,32,64))
    print(features.shape)
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
    
    
    
    clf = svm.SVC(kernel='rbf', C=100, gamma=0.1, max_iter=-1, cache_size=200, coef0=0.0
    , decision_function_shape='ovr', degree=3, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    clf.fit(X_train,y_train)

    print(X_test.shape)

    print("Testing acc:",clf.score(X_test,y_test)*100)

    print("Training acc:",clf.score(X_train,y_train)*100)

    
    clf = svm.SVC(kernel='linear', C=100)

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

    clf_grid1 = GridSearchCV(svm.SVC(), param_grid, verbose=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    clf_grid1.fit(X_train, y_train)
    print("Best Parameters:\n", clf_grid1.best_params_)

    print("Best Estimators:\n", clf_grid1.best_estimator_)
'''

for i in range(5):
    
    with open('Signal_Processing/LSTM_intermediate_features_peaks_X_train'+str(i), 'rb') as fp:
        X_train=pickle.load(fp)

    with open('Signal_Processing/LSTM_intermediate_features_peaks_X_test'+str(i), 'rb') as fp:
        X_test=pickle.load(fp)

    with open('Signal_Processing/LSTM_intermediate_labels_peaks_y_train'+str(i), 'rb') as fp:
        y_train=pickle.load(fp)

    with open('Signal_Processing/LSTM_intermediate_labels_peaks_y_test'+str(i), 'rb') as fp:
        y_test=pickle.load(fp)

    y1=[]
    y2=[]
    for l in y_train:
        if np.argmax(l)==0:    #alco
            y1.append(0)
        else:
            y1.append(1)     #control
    
    for l in y_test:
        if np.argmax(l)==0:    #alco
            y2.append(0)
        else:
            y2.append(1)     #control
    
    y_train=np.array(y1)
    y_test=np.array(y2)
    
    clf = svm.SVC(kernel='rbf', C=100, gamma=0.1, max_iter=-1, cache_size=200, coef0=0.0
    , decision_function_shape='ovr', degree=3, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)


    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    clf.fit(X_train,y_train)

    print(X_test.shape)

    print("Testing acc:",clf.score(X_test,y_test)*100)
    preds=clf.predict(X_test)
    from sklearn.metrics import confusion_matrix

    conf=confusion_matrix(y_test, preds)
    labs=["Alcoholic", "Control"]
    df_cm = pd.DataFrame(conf, index = [i for i in labs],
                  columns = [i for i in labs])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig("Conf_Matrix_SVM-"+str(i)+".png")

    from sklearn.metrics import roc_auc_score
    print("ROC: ",roc_auc_score(y_test, preds))


'''
clf = svm.SVC(kernel='linear', C=100)

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

clf_grid1 = GridSearchCV(svm.SVC(), param_grid, verbose=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
clf_grid1.fit(X_train, y_train)
print("Best Parameters:\n", clf_grid1.best_params_)

print("Best Estimators:\n", clf_grid1.best_estimator_)
'''