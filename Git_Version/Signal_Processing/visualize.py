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
from scipy.signal import argrelmax
from scipy.signal import find_peaks

#getting data
with open('../features', 'rb') as fp:
    features=pickle.load(fp)

with open('../labels', 'rb') as fp:
    labels=pickle.load(fp)


index_features=[]

alco_indices=[]
control_indices=[]
i=0

minlimit=1      #controls min height limit
maxlimit=18     #controls max height limit

minwidth=2      #controls min width
maxwidth=6     #controls max width

alco_peaks=[]
control_peaks=[]

train_features=[]

with tqdm(total=len(features)) as pbar:
    for f in features:
        sublist=[]
        flag=False
        sublist2=[]
        sublist3=[]
        p=0
        for channel in f:
            x=np.array(find_peaks(channel,height=[minlimit,maxlimit],
            width=[minwidth,maxwidth]))
            x=x[0]
            sublist.append(len(x))
        
        if np.argmax(labels[i])==0:
            alco_peaks.append(1)
            alco_indices.append(np.array(sublist))      #alco per channel peaks 
        else:
            control_peaks.append(1)
            control_indices.append(np.array(sublist))       #control per channel peaks
        i+=1
        pbar.update(1)
        
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
alco_std=[]
control_std=[]

for i in range(64):
    alco_std.append(np.std(alco_indices[:,i]))
    control_std.append(np.std(control_indices[:,i]))

alco_std=np.array(alco_std)
control_std=np.array(control_std)


x=np.arange(0,64)

plt.plot(x,alco_means,'bo')

plt.plot(x,control_means,'r+')
plt.xlabel("Channels")
plt.ylabel("Average num of peaks")

plt.title("Avg num of peaks in each channel, height:"+str(minlimit)+"-"+str(maxlimit)
+"width: "+str(minwidth)+"-"+str(maxwidth))

plt.legend(['Alcoholic', 'Control'], loc='upper right')
plt.show()


x=np.arange(0,64)

plt.plot(x,alco_std,'bo')

plt.plot(x,control_std,'r+')
plt.xlabel("Channels")
plt.ylabel("Satndard Deviation")

plt.title("Std dev in each channel, height:"+str(minlimit)+"-"+str(maxlimit)
+"width: "+str(minwidth)+"-"+str(maxwidth))

plt.legend(['Alcoholic', 'Control'], loc='upper right')
plt.show()