import glob
from tqdm import tqdm
import numpy as np
import os

import pickle

files=os.listdir('Test_data/')

i=0
features=[]
labels=[]
with tqdm(total=len(files)) as pbar:
    for file in files:
        r=0
        file1='Test_data/'+file
        f=open(file1,'r')
        x=f.readline()
        x=x.split()
        x=x[1]
        lab=x[3]
        if lab=='a':
            label=np.array([1,0])       #alcoholic
        elif lab=='c':
            label=np.array([0,1])       #control
        j=0
        sublist=[]
        feature = []
        flag=False
        while(x):
            if flag:
                flag=False
                x = f.readline()
                continue
            if j>4:
                x=x.split()
                num=int(x[2])
                value = float(x[3])
                if num==255:
                    sublist.append(value)
                    feature.append(np.array(sublist))
                    sublist=[]
                    flag=True
                else:
                    sublist.append(value)
            x=f.readline()
            j+=1
        r+=1
        feature=np.array(feature)
        if feature.shape ==(64,256):
            features.append(feature)
            labels.append(label)
        else:
            print("passed")
        
        i+=1
        pbar.update(1)

labels = np.array(labels)
features = np.array(features)
print(labels.shape)
print(features.shape)

with open('test_features', 'wb') as fp:
    pickle.dump(features, fp)

with open('test_labels', 'wb') as fp:
    pickle.dump(labels, fp)
