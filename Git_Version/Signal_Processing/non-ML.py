from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
import pickle
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from scipy.ndimage.filters import laplace
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

#getting data
with open('alco_means', 'rb') as fp:
    alco_means=pickle.load(fp)

with open('control_means', 'rb') as fp:
    control_means=pickle.load(fp)

with open('alco_std', 'rb') as fp:
    alco_std=pickle.load(fp)

with open('control_std', 'rb') as fp:
    control_std=pickle.load(fp)


with open('alco_weights', 'rb') as fp:
    alco_weights=pickle.load(fp)

with open('control_weights', 'rb') as fp:
    control_weights=pickle.load(fp)

with open('X_test', 'rb') as fp:
    X_test=pickle.load(fp)

with open('y_test', 'rb') as fp:
    y_test=pickle.load(fp)

print(X_test.shape)
print(y_test.shape)

alco=0
control=0


y_true=[]
for y in y_test:
    y_true.append(np.argmax(y))
    if np.argmax(y)==0:
        alco+=1
    else:
        control+=1
y=np.array(y)

print(control_std.shape)

print(alco_means.shape)

print("Total alco are: ", alco)
print("Total control are: ", control)

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

tests=[]

#gets data's num of peaks depending on height and width 
with tqdm(total=len(X_test)) as pbar:
    for f in X_test:
        sublist=[]
        flag=False
        sublist2=[]
        for channel in f:
            x=np.array(find_peaks(channel,height=[minlimit,maxlimit],
            width=[minwidth,maxwidth]))
            x=x[0]
            try:
                cntr=len(x)
            except:
                flag=True
            if flag:
                break
            else:
                sublist.append(cntr)
        if flag==False:
            tests.append(np.array(sublist))
        else:
            y_true=np.delete(y_true,i)
        pbar.update(1)
        i+=1

tests=np.array(tests)

print(tests.shape)

y_pred=[]

for test in tests:
    alco_counts=0
    control_counts=0
    i=0
    for channel in test:
        if (channel>=alco_means[i]-alco_std[i]) and (channel<=alco_means[i]+alco_std[i]):
            alco_counts+=alco_weights[i]
        if (channel>=control_means[i]-control_std[i]) and (channel<=control_means[i]+control_std[i]):
            control_counts+=control_weights[i]
        i+=1
    if alco_counts>=control_counts:
        y_pred.append(0)
    else:
        y_pred.append(1)

y_pred=np.array(y_pred)

print("F1 score with peaks is: ",f1_score(y_true, y_pred, average='weighted'))