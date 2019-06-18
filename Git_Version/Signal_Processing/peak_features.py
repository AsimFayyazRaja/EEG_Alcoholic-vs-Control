from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
import pickle
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from scipy.ndimage.filters import laplace
import copy
from tqdm import tqdm

from sklearn.model_selection import train_test_split


#getting data
with open('../averaged_features', 'rb') as fp:
    features=pickle.load(fp)

with open('../averaged_labels', 'rb') as fp:
    labels=pickle.load(fp)


print(features.shape)
print(labels.shape)



X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.1,shuffle=True)


with open('X_test', 'wb') as fp:
    pickle.dump(X_test, fp)

with open('y_test', 'wb') as fp:
    pickle.dump(y_test, fp)


from scipy.signal import argrelmax
from scipy.signal import find_peaks, peak_prominences

index_features=[]

alco_indices=[]
control_indices=[]
i=0

minlimit=0      #controls min height limit
maxlimit=15     #controls max height limit

minwidth=3      #controls min width
maxwidth=8     #controls max width

alco_peaks=[]
control_peaks=[]

newfeatures=[]

'''
#gets data's num of peaks depending on height and width 
with tqdm(total=len(features)) as pbar:
    for f in features:
        sublist=[]
        for channel in f:
            x=np.array(find_peaks(channel,height=[minlimit,maxlimit],
            width=[minwidth,maxwidth]))
            x=x[0]
            index=np.arange(256)
            temp=np.setdiff1d(index,x)
            c=copy.deepcopy(channel)
            c[temp]=0      #all indices except peaks are made 0
            sublist.append(c)
        newfeatures.append(sublist)
        pbar.update(1)
        i+=1
'''

#gets data's num of peaks depending on height and width 
with tqdm(total=len(X_train)) as pbar:
    for f in X_train:
        sublist=[]
        flag=False
        for channel in f:
            arr=[]
            peaks,_=find_peaks(channel)
            prominences = peak_prominences(channel, peaks)[0]
            contour_heights = channel[peaks] - prominences
            contour_heights=np.abs(contour_heights)
            try:
                arr.append(np.sum(contour_heights))
            except:
                flag=True
            if flag:
                break
            else:
                sublist.append(np.array(arr))
        if flag==False:
            if np.argmax(y_train[i]==0):
                alco_indices.append(np.array(sublist))
            else:
                control_indices.append(np.array(sublist))
        pbar.update(1)
        i+=1

alco_indices=np.array(alco_indices)
control_indices=np.array(control_indices)

print(alco_indices.shape)
print(control_indices.shape)


alco_means=[]
control_means=[]

for i in range(64):
    alco_means.append(np.mean(alco_indices[:,i]))
    control_means.append(np.mean(control_indices[:,i]))

alco_means=np.array(alco_means)
control_means=np.array(control_means)

print(alco_means.shape)


with open('alco_means_contour', 'wb') as fp:
    pickle.dump(alco_means, fp)

with open('control_means_contour', 'wb') as fp:
    pickle.dump(control_means, fp)


alco_std=[]
control_std=[]

for i in range(64):
    alco_std.append(np.std(alco_indices[:,i]))
    control_std.append(np.std(control_indices[:,i]))

alco_std=np.array(alco_std)
control_std=np.array(control_std)

print(alco_std.shape)

with open('alco_std_contour', 'wb') as fp:
    pickle.dump(alco_std, fp)

with open('control_std_contour', 'wb') as fp:
    pickle.dump(control_std, fp)