from sklearn.cluster import DBSCAN
import glob
from tqdm import tqdm
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import pickle

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

clf = DBSCAN(eps=0.3, min_samples=2, n_jobs=6)


for i in range(5):
    with open('Signal_Processing/X_train'+str(i), 'rb') as fp:
        X_train=pickle.load(fp)

    with open('Signal_Processing/X_test'+str(i), 'rb') as fp:
        X_test=pickle.load(fp)

    with open('Signal_Processing/y_train'+str(i), 'rb') as fp:
        y_train=pickle.load(fp)

    with open('Signal_Processing/y_test'+str(i), 'rb') as fp:
        y_test=pickle.load(fp)
    X_train=np.reshape(X_train,(len(X_train),-1))
    X_test=np.reshape(X_test,(len(X_test),-1))
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    clf.fit(X_train)

    labels = clf.labels_
    print(np.unique(labels))
    print("Silhouette Coefficient: %0.3f"
      % silhouette_score(X_train, labels))

    #print("Training acc:",clf.score(X_train,y_train)*100)