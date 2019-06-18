import glob
from tqdm import tqdm
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#getting data
with open('features', 'rb') as fp:
    features=pickle.load(fp)

with open('labels', 'rb') as fp:
    labels=pickle.load(fp)

alco=[]
control=[]
i=0
for l in labels:
    if np.argmax(l)==0:
        alco.append(features[i])
    else:
        control.append(features[i])
    i+=1

newfeats=[]
newlabs=[]

for i in range(len(control)):
    newfeats.append(control[i])
    newlabs.append(np.array([0,1]))

r=np.random.randint(0,len(alco),size=len(control))

for ind in r:
    newfeats.append(alco[ind])
    newlabs.append(np.array([1,0]))

newlabs=np.array(newlabs)
newfeats=np.array(newfeats)


newfeats, X_test, newlabs, y_test = train_test_split(newfeats,newlabs,test_size=0.3, random_state=42, shuffle=True)

print(newfeats.shape)
print(newlabs.shape)

with open('balanced_features', 'wb') as fp:
    pickle.dump(newfeats, fp)

with open('balanced_labels', 'wb') as fp:
    pickle.dump(newlabs, fp)