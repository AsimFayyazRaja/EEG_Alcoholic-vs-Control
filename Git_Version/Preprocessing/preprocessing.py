from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
import pickle
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from scipy.ndimage.filters import laplace
import copy

#getting data
with open('features', 'rb') as fp:
    features=pickle.load(fp)

with open('labels', 'rb') as fp:
    labels=pickle.load(fp)

from scipy.fftpack import dct, fft, fft2, idct, ifft,ifft2

print(features.shape)

newfeatures=[]
for f in features:
    sublist=[]
    for c in f:
        x=dct(c)                                #taking dct
        temp=[]
        for value in x:
            if value>3 and value <9:
                temp.append(value)
            else:
                temp.append(0)
        sublist.append(idct(temp))              #back to time domain
    newfeatures.append(np.array(sublist))

newfeatures=np.array(newfeatures)
print(newfeatures.shape)

with open('Preprocessed_Data/delta', 'wb') as fp:
    pickle.dump(newfeatures, fp)

'''
alco=[]
control=[]
i=0
print(features.shape)
print(labels.shape)
for l in labels:
    if np.argmax(l)==0:
        alco.append(np.transpose(features[i]))
    else:
        control.append(np.transpose(features[i]))
    i+=1
alco=np.array(alco)
control=np.array(control)

alco_means1=[]
control_means1=[]
for i in range(256):
    alco_means1.append(alco[:,i].mean())
    control_means1.append(control[:,i].mean())

alco_means1=np.array(alco_means1)
control_means1=np.array(control_means1)

point_means={}
for i in range(256):
    point_means[i]=np.abs(alco_means1[i]-control_means1[i])

point_means = sorted(point_means.items(), key=lambda kv: kv[1])
point_means.reverse()
#print(point_means)

with open('Preprocessed_Data/sorted_means_perpoint', 'wb') as fp:
    pickle.dump(point_means, fp)


newfeatures=[]
for f in features:
    newfeatures.append(np.transpose(f))
    
newfeatures=np.array(newfeatures)
features=copy.deepcopy(newfeatures)
print(features.shape)

i=0
features1=[]
for k,v in dict(point_means).items():
    if i >15:
        break
    features1.append(features[:,k])
    i+=1

features1=np.array(features1)

features1=features1.transpose(1,0,2)

print(features1.shape)


with open('Preprocessed_Data/perpoint_mean_features', 'wb') as fp:
    pickle.dump(features1, fp)
'''
'''
print(features.shape)
channel=features[:,0]
print(channel.shape)

channel_means=[]
for i in range(64):
    channel=features[:,i]       #getting channel num 1,2,3..64
    channel_means.append(channel.mean())
channel_means=np.array(channel_means)
print(channel_means.shape)


newfeatures=[]
i=0
for f in features:
    sublist=[]
    i=0
    for c in f:     #subtracting channel's mean from each element of that channel and squaring
        sublist.append(np.square(c-channel_means[i]))
        i+=1
    newfeatures.append(np.array(sublist))

newfeatures=np.array(newfeatures)
print(newfeatures.shape)

with open('Preprocessed_Data/variance', 'wb') as fp:
    pickle.dump(newfeatures, fp)
#mean stuff
'''

'''
import operator

alco=[]
control=[]
i=0
print(features.shape)
print(labels.shape)
for l in labels:
    if np.argmax(l)==0:
        alco.append(features[i])
    else:
        control.append(features[i])
    i+=1
alco=np.array(alco)
control=np.array(control)
print(alco.shape)
print(control.shape)

alco_means=[]
control_means=[]

for i in range(64):
    alco_means.append(alco[:,i].mean())
    control_means.append(control[:,i].mean())

alco_means=np.array(alco_means)
control_means=np.array(control_means)


with open('Preprocessed_Data/alcoholic_means', 'wb') as fp:
    pickle.dump(alco_means, fp)

with open('Preprocessed_Data/control_means', 'wb') as fp:
    pickle.dump(control_means, fp)

means={}
for i in range(64):
    means[i]=np.abs(alco_means[i]-control_means[i])

means = sorted(means.items(), key=lambda kv: kv[1])
means.reverse()
print(means)

with open('Preprocessed_Data/sorted_means', 'wb') as fp:
    pickle.dump(means, fp)
'''

'''
x=np.arange(0,64)

ann=[]
for j in range(2):
    ann.append([x[j],alco_means[j]])

ann=np.array(ann)
print(ann.shape)

p1=plt.scatter(x,alco_means, c='r',marker='o')
p2=plt.scatter(x,control_means, c='b',marker='x')
plt.legend((p1,p2),("Alcoholic", "Control"))
plt.title("Mean values per single channel calculated over all examples")
plt.annotate(x[0],(0,-0.56216753))
plt.show()
'''
'''
plt.plot(alco_means,control_means,'bo')

plt.xlabel("Alcoholic means per channel")
plt.ylabel("Control means per channel")
plt.title("Mean values of alcoholic and control per channel calculated over training set")

for i in range(64):
    plt.annotate(i+1,(alco_means[i],control_means[i]))

plt.show()
'''
'''
with open('Preprocessed_Data/laplace_filter', 'wb') as fp:
    pickle.dump(new_features, fp)
'''

'''
N=16
new_features=[]
for f in features:
    sublist=[]
    for channel in f:
        temp=np.square(channel)
        sublist.append(np.convolve(temp, np.ones((N,))/N, mode='valid'))
    new_features.append(np.array(sublist))

new_features=np.array(new_features)
print(new_features.shape)

with open('Preprocessed_Data/moving_avg', 'wb') as fp:
    pickle.dump(new_features, fp)


transformer = FastICA(n_components=64,max_iter=400,random_state=0)
transformed_features = transformer.fit_transform(features)


with open('ICA.pkl', 'wb') as fid:
    cPickle.dump(transformer, fid)   


print(transformed_features.shape)


with open('ICA_features', 'wb') as fp:
    pickle.dump(transformed_features, fp)

from scipy.fftpack import dst

from scipy.fftpack.convolve import convolve


transformed_features=dst(features, type=2)

print(transformed_features.shape)

with open('DCT_features', 'wb') as fp:
    pickle.dump(transformed_features, fp)
'''

'''
from scipy.fftpack import fft, dct

transformed_features=fft(features)

print(transformed_features.shape)

with open('DCT_features', 'wb') as fp:
    pickle.dump(transformed_features, fp)
'''