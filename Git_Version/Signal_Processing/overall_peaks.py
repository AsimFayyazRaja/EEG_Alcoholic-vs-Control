from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
import pickle
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from scipy.ndimage.filters import laplace
import copy
from tqdm import tqdm


#getting data
with open('updatedFeatures', 'rb') as fp:
    features=pickle.load(fp)

with open('../smoote_labels', 'rb') as fp:
    labels=pickle.load(fp)


print(features.shape)
print(labels.shape)



from scipy.signal import argrelmax
from scipy.signal import find_peaks

index_features=[]

alco_features=[]
control_features=[]
i=0

minlimit=0      #controls min height limit
maxlimit=15     #controls max height limit

minwidth=3      #controls min width
maxwidth=8     #controls max width

newfeatures=[]

alco_indices=[]
control_indices=[]

#gets data's num of peaks depending on height and width 
with tqdm(total=len(features)) as pbar:
    for f in features:
        sublist=[]
        flag=False
        peaks=0
        for channel in f:
            x=np.array(find_peaks(channel,height=[minlimit,maxlimit],
            width=[minwidth,maxwidth]))
            x=x[0]
            peaks+=len(x)
        if np.argmax(labels[i])==0:
            alco_indices.append(i)
            alco_features.append(peaks)      #alco per channel peaks 
        else:
            control_features.append(peaks)       #control per channel peaks
            control_indices.append(i)
        newfeatures.append(peaks)
        pbar.update(1)
        i+=1

newfeatures=np.array(newfeatures)
print(newfeatures.shape)

print(len(alco_features))
print(len(control_features))

plt.scatter(alco_indices,alco_features,marker='o')
plt.scatter(control_indices,control_features,marker='*')

plt.xlabel("Sample number")
plt.ylabel("Total peaks in all channels")
plt.legend(['Alcoholic', 'Control'], loc='upper right')

plt.show()
