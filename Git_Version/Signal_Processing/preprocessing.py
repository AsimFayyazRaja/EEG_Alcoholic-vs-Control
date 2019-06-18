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
with open('../features', 'rb') as fp:
    features=pickle.load(fp)

with open('../labels', 'rb') as fp:
    labels=pickle.load(fp)


print(features.shape)
print(labels.shape)


X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.1,shuffle=True)


with open('X_test', 'wb') as fp:
    pickle.dump(X_test, fp)

with open('y_test', 'wb') as fp:
    pickle.dump(y_test, fp)


from scipy.signal import argrelmax
from scipy.signal import find_peaks


#gets data's num of peaks depending on height and width 
with tqdm(total=64) as pbar:
    for channel_to_check in range(64):
        index_features=[]

        alco_indices=[]
        control_indices=[]
        i=0

        minlimit=0      #controls min height limit
        maxlimit=30     #controls max height limit

        minwidth=3      #controls min width
        maxwidth=8     #controls max width

        alco_peaks=[]
        control_peaks=[]

        train_features=[]

        for f in X_train:
            sublist=[]
            flag=False
            sublist2=[]
            sublist3=[]
            p=0
            for channel in f:
                if p==channel_to_check:
                    x=np.array(find_peaks(channel,height=[minlimit,maxlimit],
                    width=[minwidth,maxwidth]))
                    x=x[0]
                p+=1

            train_features.append(np.array(sublist3))
            if np.argmax(y_train[i])==0:
                alco_peaks.append(1)
                alco_indices.append(len(x))      #alco per channel peaks 
            else:
                control_peaks.append(1)
                control_indices.append(len(x))       #control per channel peaks
            i+=1
            

        train_features=np.array(train_features)

        alco_indices=np.array(alco_indices)
        control_indices=np.array(control_indices)

        with open('alco_indices', 'wb') as fp:
            pickle.dump(alco_indices, fp)

        with open('control_indices', 'wb') as fp:
            pickle.dump(control_indices, fp)

        alco_peaks=np.array(alco_peaks)
        control_peaks=np.array(control_peaks)

        with open('alco_peaks', 'wb') as fp:
            pickle.dump(alco_peaks, fp)

        with open('control_peaks', 'wb') as fp:
            pickle.dump(control_peaks, fp)

        x=np.arange(0,len(alco_indices))


        fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figure

        ax2.plot(x,alco_indices,'bo', c='r')

        ax2.set_xlabel("Num of Alcoholic samples")
        ax2.set_ylabel("Num of peaks 0-30, 3-8")
        ax2.set_title("Channel "+str(channel_to_check)+" only")

        #plt.savefig('Perchannel_peaks/averaged_alcoholic_channel'+str(channel_to_check))

        x=np.arange(0,len(control_indices))

        ax3.plot(x,control_indices,'bo')

        ax3.set_xlabel("Num of Control samples")
        ax3.set_ylabel("Num of peaks 0-30, 3-8")


        plt.savefig('Perchannel_peaks/full_channel'+str(channel_to_check)+'.png')
        plt.close()
        pbar.update(1)


'''
with open('alco_means', 'wb') as fp:
    pickle.dump(alco_means, fp)

with open('control_means', 'wb') as fp:
    pickle.dump(control_means, fp)


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
plt.ylabel("Std dev in num of peaks")
plt.title("Std dev of peaks of " + str(minlimit) +
          " to " + str(maxlimit) + " height limit"
          + str(minwidth) + " to "+str(maxwidth) + " width limit")
plt.legend(['Alcoholic', 'Control'], loc='upper right')
plt.show()


with open('alco_std', 'wb') as fp:
    pickle.dump(alco_std, fp)

with open('control_std', 'wb') as fp:
    pickle.dump(control_std, fp)

alco_channels_wrong=np.zeros(64)          #each channel predicting wrong class info
control_channels_wrong=np.zeros(64)       #each channel predicting wrong class info

j=0
#adjusting weights to give to each channel here
for f in train_features:
    i=0
    sublist=[]
    sublist2=[]
    for channel in f:
        if (channel>alco_means[i]-alco_std[i]) and (channel<alco_means[i]+alco_std[i]):
            if (np.argmax(y_train[j])==0):        
                alco_channels_wrong[i]+=1   #if true prediction then add some weight
            else:           #if wrong prediction then penalize
                alco_channels_wrong[i]=alco_channels_wrong[i]-1
        elif (channel>control_means[i]-control_std[i]) and (channel<control_means[i]+control_std[i]):
            if (np.argmax(y_train[j])==1):
                control_channels_wrong[i]+=1   #if true prediction then add some weight
            else:           #if wrong prediction then penalize
                control_channels_wrong[i]=control_channels_wrong[i]-1
        i+=1    
    j+=1

alco_channels_wrong=alco_channels_wrong/len(alco_peaks)
control_channels_wrong=control_channels_wrong/len(control_peaks)


print(alco_channels_wrong.shape)
print(alco_channels_wrong)

print(control_channels_wrong.shape)
print(control_channels_wrong)


with open('alco_weights', 'wb') as fp:
    pickle.dump(alco_channels_wrong, fp)

with open('control_weights', 'wb') as fp:
    pickle.dump(control_channels_wrong, fp)
'''