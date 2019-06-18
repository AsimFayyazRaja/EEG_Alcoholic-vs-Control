from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
import pickle
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
from scipy.ndimage.filters import laplace
import copy
from tqdm import tqdm

from collections import Counter

#getting data
with open('../averaged_features', 'rb') as fp:
    features=pickle.load(fp)

with open('../averaged_labels', 'rb') as fp:
    labels=pickle.load(fp)


print(features.shape)
print(labels.shape)




from scipy.signal import argrelmax
from scipy.signal import find_peaks

index_features=[]

alco_indices=[]
control_indices=[]
i=0

minlimit=0      #controls min height limit
maxlimit=15     #controls max height limit

minwidth=3      #controls min width
maxwidth=8     #controls max width

newfeatures=[]

channel_to_check=0

alcpos=[]
controlpos=[]

alco_file = open("alco_peak_positions.txt", "w")
alco_file.write('Channel'+'\t'+'\t'+'Positions')
alco_file.write('\n \n')

control_file = open("control_peak_positions.txt", "w")
control_file.write('Channel'+'\t'+'\t'+'Positions')
control_file.write('\n \n')

alc_pos=[]
con_pos=[]

#gets data's num of peaks depending on height and width 
with tqdm(total=64) as pbar:
    for channel_to_check in range(64):
        index_features=[]

        alco_indices=[]
        control_indices=[]
        i=0

        minlimit=0      #controls min height limit
        maxlimit=15     #controls max height limit

        minwidth=3      #controls min width
        maxwidth=8     #controls max width


        for f in features:
            sublist=[]
            flag=False
            p=0
            for channel in f:
                if p==channel_to_check:
                    x=np.array(find_peaks(channel,height=[minlimit,maxlimit],
                    width=[minwidth,maxwidth]))
                    x=x[0]
                    if np.argmax(labels[i])==0:
                        alco_indices.append(np.array(x))
                    else:
                        control_indices.append(np.array(x))
                p+=1
            i+=1
        

        alco_positions = [item for sublist in alco_indices for item in sublist]
        control_positions = [item for sublist in control_indices for item in sublist]

        alco_positions=np.array(alco_positions)
        control_positions=np.array(control_positions)

        a=Counter(alco_positions).most_common(10)
        alc_pos.append(a)
        alco_file.write(str(channel_to_check)+'\t'+str(Counter(alco_positions).most_common(10)))
        alco_file.write('\n \n')

        c=Counter(control_positions).most_common(10)
        con_pos.append(c)
        control_file.write(str(channel_to_check)+'\t'+str(Counter(control_positions).most_common(10)))
        control_file.write('\n \n')

        alcpos.append(a)
        controlpos.append(c)
        pbar.update(1)
alco_file.close()
control_file.close()

with open('alco_positions', 'wb') as fp:
    pickle.dump(alc_pos, fp)

with open('control_positions', 'wb') as fp:
    pickle.dump(con_pos, fp)


'''
alcpos=np.array(alcpos)
controlpos=np.array(controlpos)

print(alcpos.shape)
print(controlpos.shape)

x=np.arange(0,64)

plt.plot(x,alcpos,'bo')

plt.plot(x,controlpos,'r+')

plt.xlabel("Channels")
plt.ylabel("Position")
plt.title("Position where most of the time peak occured")
plt.legend(['Alcoholic', 'Control'], loc='upper right')
plt.show()
'''
