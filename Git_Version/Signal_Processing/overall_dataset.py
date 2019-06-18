from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
import pickle
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from scipy.ndimage.filters import laplace
import copy
from tqdm import tqdm

from scipy.signal import chirp, find_peaks, peak_widths

from sklearn.model_selection import train_test_split

#getting data
with open('../features', 'rb') as fp:
    features=pickle.load(fp)

with open('../labels', 'rb') as fp:
    labels=pickle.load(fp)


from scipy.signal import argrelmax, argrelmin, argrelextrema
from scipy.signal import find_peaks,peak_prominences


index_features=[]

i=0
'''
minlimit=0      #controls min height limit
maxlimit=30     #controls max height limit

minwidth=2      #controls min width
maxwidth=6     #controls max width
'''

minlimit=1      #controls min height limit
maxlimit=18     #controls max height limit

minwidth=2      #controls min width
maxwidth=6     #controls max width

train_features=[]


#gets data's num of peaks depending on height and width 
with tqdm(total=len(features)) as pbar:
    for f in features:
        sublist=[]
        flag=False
        sublist2=[]
        sublist3=[]
        p=0
        for channel in f:
            perchannel=[]
            
            x=np.array(find_peaks(channel,height=[minlimit,maxlimit],
            width=[minwidth,maxwidth]))     #indices of peaks of desired height and width
            '''
            minl=10
            maxl=100
            x=np.array(find_peaks(channel,height=[minl,maxl]))     #indices of peaks of desired height and width
            '''
            x=x[0]
            perchannel.append(len(x))       #length is added i.e. how much peaks are there
            
            '''
            for j in range(10):         #first 15 peaks values are added
                try:
                    perchannel.append(channel[x[j]])
                except:
                    perchannel.append(0)
            '''
            for j in range(15):         #first 15 peaks locations are added
                try:
                    perchannel.append(x[j])
                except:
                    perchannel.append(0)
            
            peaks,_=find_peaks(channel)
            '''
            widths = peak_widths(channel, peaks)[0]

            for j in range(15):         #first 15 peaks locations are added
                try:
                    perchannel.append(widths[j])
                except:
                    perchannel.append(0)
            '''
            prominences = peak_prominences(channel, peaks)[0]
            for j in range(15):     #15 prominence of peaks are added
                try:
                    perchannel.append(prominences[j])
                except:
                    perchannel.append(0)
            x=argrelmax(channel)
            x=x[0]
            for j in range(15):         #15 relative maxima positions are added
                try:
                    perchannel.append(x[j])
                except:
                    perchannel.append(0)
            x=argrelmin(channel)
            x=x[0]
            for j in range(15):         #15 relative minima positions are added
                try:
                    perchannel.append(x[j])
                except:
                    perchannel.append(0)
            
            perchannel.append(np.mean(channel))
            perchannel.append(np.std(channel))
            sublist.append(np.array(perchannel))
        train_features.append(np.array(sublist))
        i+=1
        pbar.update(1)


train_features=np.array(train_features)

print(train_features.shape)


with open('overall_full_features', 'wb') as fp:
    pickle.dump(train_features, fp)

with open('overall_full_labels', 'wb') as fp:
    pickle.dump(labels, fp)