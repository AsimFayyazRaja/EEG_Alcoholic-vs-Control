import glob
from tqdm import tqdm
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#getting data
with open('Signal_Processing/overall_full_features', 'rb') as fp:
    features=pickle.load(fp)

with open('Signal_Processing/overall_full_labels', 'rb') as fp:
    labels=pickle.load(fp)

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


scaler = StandardScaler()
scaler=scaler.fit(X)

X=scaler.transform(X)

print(X.shape)

clf =LogisticRegression(random_state=0, solver='sag',multi_class='ovr',max_iter=50000,n_jobs=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
clf.fit(X_train,y_train)

print(X_test.shape)

print("Testing acc=",clf.score(X_test,y_test)*100)

print("Training acc=",clf.score(X_train,y_train)*100)
