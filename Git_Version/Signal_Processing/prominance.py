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
with open('../averaged_features', 'rb') as fp:
    features=pickle.load(fp)

with open('../averaged_labels', 'rb') as fp:
    labels=pickle.load(fp)


print(features.shape)
print(labels.shape)




from scipy.signal import find_peaks, peak_prominences

index_features=[]

alco_indices=[]
control_indices=[]
i=0

minlimit=15      #controls min height limit
maxlimit=30     #controls max height limit

minwidth=2      #controls min width
maxwidth=4     #controls max width

#gets data's num of peaks depending on height and width 
with tqdm(total=len(features)) as pbar:
    for f in features:
        sublist=[]
        flag=False
        for channel in f:
            arr=[]
            peaks,_=find_peaks(channel)
            prominences = peak_prominences(channel, peaks)[0]
            '''
            contour_heights = channel[peaks] - prominences
            contour_heights=np.abs(contour_heights)
            try:
                for m in range(10):
                    temp=np.argmax(contour_heights)
                    arr.append(contour_heights[temp])
                    contour_heights=np.delete(contour_heights,temp)
            except:
                flag=True
            try:
                for m in range(10):
                    temp=np.argmax(prominences)
                    if prominences[temp]>18:
                        arr.append(prominences[temp])
                        prominences=np.delete(prominences,temp)
                    else:
                        arr.append(0)
            except:
                flag=True
            '''
            try:
                s=float(np.sum(prominences))
                sublist.append(s)
            except Exception as e:
                print(e)
                flag=True
            if flag:
                break

        if flag==False:
            if np.argmax(labels[i]==0):
                alco_indices.append(np.array(sublist))
            else:
                control_indices.append(np.array(sublist))
        
        pbar.update(1)
        i+=1
            
alco_indices=np.array(alco_indices)
control_indices=np.array(control_indices)

with open('prominance_alco', 'wb') as fp:
    pickle.dump(alco_indices, fp)

with open('prominance_control', 'wb') as fp:
    pickle.dump(control_indices, fp)


#getting data
with open('prominance_alco', 'rb') as fp:
    alco_indices=pickle.load(fp)

with open('prominance_control', 'rb') as fp:
    control_indices=pickle.load(fp)

print(alco_indices.shape)
print(control_indices.shape)

alco_means=[]
control_means=[]

print(alco_indices[:,5].shape)

for i in range(64):
    alco_means.append(np.mean(alco_indices[:,i]))
    control_means.append(np.mean(control_indices[:,i]))

x=np.arange(0,64)

plt.plot(x,alco_means,'bo')

plt.plot(x,control_means,'r+')
plt.xlabel("Channels")
plt.ylabel("Average prominence sum")
plt.title("Avg prominence sum for each channel")
plt.legend(['Alcoholic', 'Control'], loc='upper right')
plt.show()

alco_std=[]
control_std=[]

for i in range(64):
    alco_std.append(np.std(alco_indices[:,i]))
    control_std.append(np.std(control_indices[:,i]))

alco_std=np.array(alco_std)
control_std=np.array(control_std)

print(alco_std.shape)
print(control_std.shape)

plt.plot(x,alco_std,'bo')

plt.plot(x,control_std,'r+')
plt.xlabel("Channels")
plt.ylabel("Std dev of prominence sums")
plt.title("Std dev of prominence sums of all channels")
plt.legend(['Alcoholic', 'Control'], loc='upper right')
plt.show()
