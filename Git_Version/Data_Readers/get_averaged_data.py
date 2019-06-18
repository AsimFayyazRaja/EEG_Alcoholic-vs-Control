import glob
from tqdm import tqdm
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#getting data
with open('4_features', 'rb') as fp:
    features=pickle.load(fp)

with open('4_labels', 'rb') as fp:
    labels=pickle.load(fp)

'''
newfeatures=[]
newlabels=[]
j=0
with tqdm(total=len(features)) as pbar:
    for feature in features:            #making set of 4 features
        i=0
        sublist=[]
        for f in feature:
            if i==0:
                i+=1
                sublist.append(f)
                continue
            else:
                if i%4==0:
                    newlabels.append(labels[j])
                    newfeatures.append(np.array(sublist))
                    i=0
                    sublist=[]
                else:
                    sublist.append(f)
                    i+=1
        j+=1
        pbar.update(1)

newfeatures=np.array(newfeatures)
newlabels=np.array(newlabels)

print(newfeatures.shape)
print(newlabels.shape)
with open('4_features', 'wb') as fp:
    pickle.dump(newfeatures, fp)

with open('4_labels', 'wb') as fp:
    pickle.dump(newlabels, fp)
'''

newfeatures=[]
for f in features:
    newfeatures.append(np.mean(f,axis=(0)))

newfeatures=np.array(newfeatures)
print(newfeatures.shape)

with open('averaged_features', 'wb') as fp:
    pickle.dump(newfeatures, fp)

with open('averaged_labels', 'wb') as fp:
    pickle.dump(labels, fp)